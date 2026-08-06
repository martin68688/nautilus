import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import AutoModel, AutoProcessor

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================
# 1. Load data
# ============================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ============================================
# 2. Separate features and target
# ============================================
feature_cols = [
    col for col in train_df.columns if col.startswith(("margin", "shape", "texture"))
]
print(f"Number of feature columns: {len(feature_cols)}")

# Rename columns to have consistent naming with underscore (margin_1 instead of margin1)
rename_map = {}
for col in feature_cols:
    if "_" not in col:
        match = re.match(r"([a-zA-Z]+)(\d+)", col)
        if match:
            prefix, num = match.groups()
            new_col = f"{prefix}_{num}"
            rename_map[col] = new_col

train_df = train_df.rename(columns=rename_map)
test_df = test_df.rename(columns=rename_map)

feature_cols = [
    col for col in train_df.columns if col.startswith(("margin_", "shape_", "texture_"))
]
print(f"After rename - feature columns: {len(feature_cols)}")

# ============================================
# 3. Handle missing values (if any)
# ============================================
if train_df[feature_cols].isnull().sum().sum() > 0:
    for col in feature_cols:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

# ============================================
# 4. Encode species labels
# ============================================
label_encoder = LabelEncoder()
train_df["species_encoded"] = label_encoder.fit_transform(train_df["species"])
print(f"Number of unique species: {len(label_encoder.classes_)}")

class_names = list(label_encoder.classes_)
submission_cols = sample_sub.columns.tolist()[1:]
assert (
    len(submission_cols) == len(class_names) == 99
), f"Column count mismatch: {len(submission_cols)} vs {len(class_names)}"

# ============================================
# 5. Train/Validation Split (Stratified)
# ============================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(
    iter(skf.split(train_df[feature_cols].values, train_df["species_encoded"].values))
)

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)
test_set = test_df.reset_index(drop=True)

assert len(set(train_idx) & set(val_idx)) == 0, "Train/Val overlap detected!"
print(
    f"Train samples: {len(train_set)}, Val samples: {len(val_set)}, Test samples: {len(test_set)}"
)

# ============================================
# 6. Feature Scaling (Fit on train only)
# ============================================
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_set[feature_cols].values)
val_features_scaled = scaler.transform(val_set[feature_cols].values)
test_features_scaled = scaler.transform(test_set[feature_cols].values)

# ============================================
# 7. Prepare data tensors
# ============================================
margin_cols = [c for c in feature_cols if "margin_" in c]
shape_cols = [c for c in feature_cols if "shape_" in c]
texture_cols = [c for c in feature_cols if "texture_" in c]

train_ids = train_set["id"].values
val_ids = val_set["id"].values
test_ids = test_set["id"].values

images_base = "./input/images"
train_paths = [f"{images_base}/{i}.jpg" for i in train_ids]
val_paths = [f"{images_base}/{i}.jpg" for i in val_ids]
test_paths = [f"{images_base}/{i}.jpg" for i in test_ids]

for path in train_paths[:3] + test_paths[:3]:
    assert os.path.exists(path), f"Image not found: {path}"

# ============================================
# 8. Image Feature Extraction with SigLIP2 (cached)
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CACHE_FILE = "./working/siglip2_features_cache.pkl"


def load_siglip2():
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-256")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, processor


def extract_image_features_batch(image_paths, model, processor, batch_size=32):
    all_features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                features = model.get_image_features(**inputs)
        all_features.append(features.detach().cpu().numpy())
    return np.vstack(all_features)


if os.path.exists(CACHE_FILE):
    print("Loading cached image features...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    train_img_feats = cache["train"]
    val_img_feats = cache["val"]
    test_img_feats = cache["test"]
else:
    print("Extracting image features with SigLIP2...")
    model, processor = load_siglip2()
    train_img_feats = extract_image_features_batch(train_paths, model, processor)
    val_img_feats = extract_image_features_batch(val_paths, model, processor)
    test_img_feats = extract_image_features_batch(test_paths, model, processor)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(
            {"train": train_img_feats, "val": val_img_feats, "test": test_img_feats}, f
        )

print(
    f"Image features - train: {train_img_feats.shape}, val: {val_img_feats.shape}, test: {test_img_feats.shape}"
)

# ============================================
# 9. Prepare Data Tensors
# ============================================
X_train_margin = train_set[margin_cols].values.astype(np.float32)
X_train_shape = train_set[shape_cols].values.astype(np.float32)
X_train_texture = train_set[texture_cols].values.astype(np.float32)
y_train = train_set["species_encoded"].values.astype(np.int64)

X_val_margin = val_set[margin_cols].values.astype(np.float32)
X_val_shape = val_set[shape_cols].values.astype(np.float32)
X_val_texture = val_set[texture_cols].values.astype(np.float32)
y_val = val_set["species_encoded"].values.astype(np.int64)

X_test_margin = test_set[margin_cols].values.astype(np.float32)
X_test_shape = test_set[shape_cols].values.astype(np.float32)
X_test_texture = test_set[texture_cols].values.astype(np.float32)


# ============================================
# 10. Define Dataset and DataLoader
# ============================================
class LeafDataset(Dataset):
    def __init__(self, margin, shape, texture, img_feats, labels=None):
        self.margin = torch.FloatTensor(margin)
        self.shape = torch.FloatTensor(shape)
        self.texture = torch.FloatTensor(texture)
        self.img_feats = torch.FloatTensor(img_feats)
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.margin)

    def __getitem__(self, idx):
        if self.labels is not None:
            return (
                self.margin[idx],
                self.shape[idx],
                self.texture[idx],
                self.img_feats[idx],
                self.labels[idx],
            )
        else:
            return (
                self.margin[idx],
                self.shape[idx],
                self.texture[idx],
                self.img_feats[idx],
            )


train_dataset = LeafDataset(
    X_train_margin, X_train_shape, X_train_texture, train_img_feats, y_train
)
val_dataset = LeafDataset(
    X_val_margin, X_val_shape, X_val_texture, val_img_feats, y_val
)
test_dataset = LeafDataset(X_test_margin, X_test_shape, X_test_texture, test_img_feats)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)


# ============================================
# 11. Model Definition
# ============================================
class MultiModalLeafClassifier(nn.Module):
    def __init__(
        self,
        n_margin=64,
        n_shape=64,
        n_texture=64,
        img_feat_dim=1152,
        hidden_dim=256,
        n_classes=99,
        dropout=0.3,
    ):
        super().__init__()
        self.margin_encoder = nn.Sequential(
            nn.Linear(n_margin, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.shape_encoder = nn.Sequential(
            nn.Linear(n_shape, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.texture_encoder = nn.Sequential(
            nn.Linear(n_texture, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(img_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        fusion_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 1.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, margin_feats, shape_feats, texture_feats, image_feats):
        margin_emb = self.margin_encoder(margin_feats)
        shape_emb = self.shape_encoder(shape_feats)
        texture_emb = self.texture_encoder(texture_feats)
        image_emb = self.image_projection(image_feats)
        fused = torch.cat([margin_emb, shape_emb, texture_emb, image_emb], dim=1)
        logits = self.classifier(fused)
        return logits


model = MultiModalLeafClassifier(
    n_margin=64,
    n_shape=64,
    n_texture=64,
    img_feat_dim=1152,
    hidden_dim=256,
    n_classes=99,
    dropout=0.3,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")

# ============================================
# 12. Loss, Optimizer, Scheduler
# ============================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = AdamW(
    [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "image_projection" in n and p.requires_grad
            ],
            "lr": 5e-4,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "image_projection" not in n and p.requires_grad
            ],
            "lr": 1e-3,
        },
    ],
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)

total_epochs = 30
warmup_epochs = 2


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None


# ============================================
# 13. Training Loop
# ============================================
def compute_log_loss(y_true, probs, n_classes=99):
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_onehot = np.eye(n_classes)[y_true]
    loss = -np.sum(y_onehot * np.log(probs)) / len(y_true)
    return loss


best_val_loss = float("inf")
best_epoch = -1
patience = 15
patience_counter = 0
model_save_path = "./working/best_model.pth"

print("\nStarting training...")
print(f"{'Epoch':<6} {'Train Loss':<12} {'Val LogLoss':<12} {'Val Acc':<8} {'Time':<8}")
print("-" * 60)

for epoch in range(total_epochs):
    start_time = time.time()
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for margin_b, shape_b, texture_b, img_b, labels_b in train_loader:
        margin_b = margin_b.to(device)
        shape_b = shape_b.to(device)
        texture_b = texture_b.to(device)
        img_b = img_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(margin_b, shape_b, texture_b, img_b)
            loss = criterion(logits, labels_b)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item() * len(margin_b)
        _, preds = torch.max(logits, 1)
        train_correct += (preds == labels_b).sum().item()
        train_total += len(labels_b)

    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    model.eval()
    val_probs = []
    val_labels_list = []

    with torch.no_grad():
        for margin_b, shape_b, texture_b, img_b, labels_b in val_loader:
            margin_b = margin_b.to(device)
            shape_b = shape_b.to(device)
            texture_b = texture_b.to(device)
            img_b = img_b.to(device)
            with torch.cuda.amp.autocast():
                logits = model(margin_b, shape_b, texture_b, img_b)
                probs = F.softmax(logits, dim=1)
            val_probs.append(probs.cpu().numpy())
            val_labels_list.append(labels_b.numpy())

    val_probs = np.vstack(val_probs)
    val_labels = np.concatenate(val_labels_list)
    val_logloss = compute_log_loss(val_labels, val_probs)
    val_preds = np.argmax(val_probs, axis=1)
    val_acc = np.mean(val_preds == val_labels)

    elapsed = time.time() - start_time
    print(
        f"{epoch+1:<6d} {avg_train_loss:<12.4f} {val_logloss:<12.4f} {val_acc:<8.4f} {elapsed:<8.2f}s"
    )

    scheduler.step()

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_logloss": val_logloss,
            },
            model_save_path,
        )
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(
    f"\nBest model from epoch {best_epoch} with validation log loss: {best_val_loss:.4f}"
)

checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Final validation inference
val_probs = []
with torch.no_grad():
    for margin_b, shape_b, texture_b, img_b, _ in val_loader:
        margin_b = margin_b.to(device)
        shape_b = shape_b.to(device)
        texture_b = texture_b.to(device)
        img_b = img_b.to(device)
        with torch.cuda.amp.autocast():
            logits = model(margin_b, shape_b, texture_b, img_b)
            probs = F.softmax(logits, dim=1)
        val_probs.append(probs.cpu().numpy())

val_probs = np.vstack(val_probs)
final_val_logloss = compute_log_loss(y_val, val_probs)
print(f"Final validation log loss: {final_val_logloss:.4f}")

# ============================================
# 14. Test Inference and Submission
# ============================================
test_probs = []
with torch.no_grad():
    for margin_b, shape_b, texture_b, img_b in test_loader:
        margin_b = margin_b.to(device)
        shape_b = shape_b.to(device)
        texture_b = texture_b.to(device)
        img_b = img_b.to(device)
        with torch.cuda.amp.autocast():
            logits = model(margin_b, shape_b, texture_b, img_b)
            probs = F.softmax(logits, dim=1)
        test_probs.append(probs.cpu().numpy())

test_probs = np.vstack(test_probs)
print(f"Test predictions shape: {test_probs.shape}")

submission = pd.DataFrame(test_probs, columns=class_names)
submission.insert(0, "id", test_ids)

expected_cols = sample_sub.columns.tolist()
actual_cols = submission.columns.tolist()
assert actual_cols == expected_cols, f"Column mismatch!"

submission_path = "./submission/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Submission shape: {submission.shape}")

print(f"\nFinal Validation Score: {final_val_logloss}")
