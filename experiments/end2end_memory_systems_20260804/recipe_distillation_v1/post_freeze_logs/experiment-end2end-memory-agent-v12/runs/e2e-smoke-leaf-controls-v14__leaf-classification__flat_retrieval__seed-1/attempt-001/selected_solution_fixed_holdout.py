import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from PIL import Image
import os
import time
import gc
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Setup device and reproducibility
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"Using device: {device}")

# ============================================================
# 1. Load Data
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")


# Dynamically detect feature group columns (handles margin1 vs margin_1)
def get_feature_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix) or c.startswith(prefix + "_")]
    return sorted(cols, key=lambda x: int("".join(filter(str.isdigit, x))))


margin_cols = get_feature_cols(train_df, "margin")
shape_cols = get_feature_cols(train_df, "shape")
texture_cols = get_feature_cols(train_df, "texture")
all_feat_cols = margin_cols + shape_cols + texture_cols
print(
    f"Feature columns found: {len(margin_cols)} margin, {len(shape_cols)} shape, {len(texture_cols)} texture"
)

# Prepare labels
species_cols = sample_sub.columns[1:].tolist()
label_encoder = LabelEncoder()
label_encoder.fit(train_df["species"].values)
train_labels = label_encoder.transform(train_df["species"].values)

# ============================================================
# 2. Split Data FIRST (prevents leakage)
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(np.zeros(len(train_df)), train_labels)))
assert len(set(train_idx) & set(val_idx)) == 0, "Train/Val overlap detected!"

train_ids = train_df["id"].values[train_idx]
val_ids = train_df["id"].values[val_idx]
test_ids = test_df["id"].values

# Raw features for model branches
train_raw_all = train_df[all_feat_cols].values.astype(np.float32)
test_raw_all = test_df[all_feat_cols].values.astype(np.float32)

X_train_raw = train_raw_all[train_idx]
X_val_raw = train_raw_all[val_idx]
X_test_raw = test_raw_all
y_train = train_labels[train_idx]
y_val = train_labels[val_idx]

print(
    f"Train shape: {X_train_raw.shape}, Val shape: {X_val_raw.shape}, Test shape: {X_test_raw.shape}"
)

# Extract raw margin, shape, texture for each split
margin_start = 0
shape_start = len(margin_cols)
texture_start = len(margin_cols) + len(shape_cols)

X_train_margin = X_train_raw[:, margin_start:shape_start]
X_train_shape = X_train_raw[:, shape_start:texture_start]
X_train_texture = X_train_raw[:, texture_start:]

X_val_margin = X_val_raw[:, margin_start:shape_start]
X_val_shape = X_val_raw[:, shape_start:texture_start]
X_val_texture = X_val_raw[:, texture_start:]

X_test_margin = X_test_raw[:, margin_start:shape_start]
X_test_shape = X_test_raw[:, shape_start:texture_start]
X_test_texture = X_test_raw[:, texture_start:]

# ============================================================
# 3. Extract image features using SigLIP2
# ============================================================
from transformers import AutoProcessor, AutoModel

print("Loading SigLIP2 model for image feature extraction...")
siglip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
siglip_processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-256")
siglip_model = siglip_model.to(device)
siglip_model.eval()

image_cache_dir = "./working/img_features"
os.makedirs(image_cache_dir, exist_ok=True)
os.makedirs("./working", exist_ok=True)


def extract_image_features(img_ids, cache_name):
    cache_path = os.path.join(image_cache_dir, f"{cache_name}.npy")
    if os.path.exists(cache_path):
        print(f"Loading cached image features for {cache_name}")
        return np.load(cache_path)

    features = []
    batch_size = 16
    id_list = list(img_ids)

    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i : i + batch_size]
        images = []
        for img_id in batch_ids:
            img_path = f"./input/images/{int(img_id)}.jpg"
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception:
                img = Image.new("RGB", (256, 256), (255, 255, 255))
                images.append(img)

        if images:
            inputs = siglip_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    pooled = siglip_model.get_image_features(**inputs)
            features.append(pooled.float().cpu().numpy())

    features = (
        np.vstack(features)
        if features
        else np.zeros((len(id_list), 1152), dtype=np.float32)
    )
    np.save(cache_path, features)
    print(f"Extracted and cached {len(features)} image features for {cache_name}")
    return features


X_train_img = extract_image_features(train_ids, "train_img")
X_val_img = extract_image_features(val_ids, "val_img")
X_test_img = extract_image_features(test_ids, "test_img")

del siglip_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 4. Model Architecture
# ============================================================
class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embed_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x):
        return self.net(x)


class ImageProjector(nn.Module):
    def __init__(self, input_dim=1152, embed_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x):
        return self.net(x)


class FusionClassifier(nn.Module):
    def __init__(
        self,
        margin_dim=64,
        shape_dim=64,
        texture_dim=64,
        image_dim=1152,
        tab_hidden=128,
        tab_embed=64,
        img_embed=128,
        fusion_hidden=256,
        n_classes=99,
        dropout=0.35,
    ):
        super().__init__()
        self.margin_encoder = TabularEncoder(margin_dim, tab_hidden, tab_embed, dropout)
        self.shape_encoder = TabularEncoder(shape_dim, tab_hidden, tab_embed, dropout)
        self.texture_encoder = TabularEncoder(
            texture_dim, tab_hidden, tab_embed, dropout
        )
        self.image_projector = ImageProjector(image_dim, img_embed, dropout)

        fusion_input_dim = 3 * tab_embed + img_embed
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(fusion_hidden // 2, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, margin, shape, texture, image_features=None):
        emb_margin = self.margin_encoder(margin)
        emb_shape = self.shape_encoder(shape)
        emb_texture = self.texture_encoder(texture)
        tab_emb = torch.cat([emb_margin, emb_shape, emb_texture], dim=1)

        if image_features is not None:
            img_emb = self.image_projector(image_features)
            fused = torch.cat([tab_emb, img_emb], dim=1)
        else:
            fused = torch.cat(
                [tab_emb, torch.zeros(tab_emb.size(0), 128, device=tab_emb.device)],
                dim=1,
            )

        logits = self.classifier(fused)
        return logits


def build_model(
    margin_dim=64,
    shape_dim=64,
    texture_dim=64,
    image_dim=1152,
    n_classes=99,
    tab_hidden=128,
    tab_embed=64,
    img_embed=128,
    fusion_hidden=256,
    dropout=0.35,
):
    return FusionClassifier(
        margin_dim=margin_dim,
        shape_dim=shape_dim,
        texture_dim=texture_dim,
        image_dim=image_dim,
        tab_hidden=tab_hidden,
        tab_embed=tab_embed,
        img_embed=img_embed,
        fusion_hidden=fusion_hidden,
        n_classes=n_classes,
        dropout=dropout,
    )


def create_criterion(n_classes=99, label_smoothing=0.1):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def create_optimizer(model, lr=1e-3, weight_decay=5e-4):
    return AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def create_scheduler(optimizer, total_epochs=80, warmup_epochs=3):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 5. Build DataLoaders
# ============================================================
batch_size = 32
num_workers = 2 if torch.cuda.is_available() else 0

X_train_margin_t = torch.FloatTensor(X_train_margin)
X_train_shape_t = torch.FloatTensor(X_train_shape)
X_train_texture_t = torch.FloatTensor(X_train_texture)
X_train_img_t = torch.FloatTensor(X_train_img)
y_train_t = torch.LongTensor(y_train)

X_val_margin_t = torch.FloatTensor(X_val_margin)
X_val_shape_t = torch.FloatTensor(X_val_shape)
X_val_texture_t = torch.FloatTensor(X_val_texture)
X_val_img_t = torch.FloatTensor(X_val_img)
y_val_t = torch.LongTensor(y_val)

X_test_margin_t = torch.FloatTensor(X_test_margin)
X_test_shape_t = torch.FloatTensor(X_test_shape)
X_test_texture_t = torch.FloatTensor(X_test_texture)
X_test_img_t = torch.FloatTensor(X_test_img)

train_dataset = TensorDataset(
    X_train_margin_t, X_train_shape_t, X_train_texture_t, X_train_img_t, y_train_t
)
val_dataset = TensorDataset(
    X_val_margin_t, X_val_shape_t, X_val_texture_t, X_val_img_t, y_val_t
)
test_dataset = TensorDataset(
    X_test_margin_t, X_test_shape_t, X_test_texture_t, X_test_img_t
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

print(
    f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
)

# ============================================================
# 6. Training Setup
# ============================================================
model = build_model(
    margin_dim=64,
    shape_dim=64,
    texture_dim=64,
    image_dim=1152,
    n_classes=99,
    tab_hidden=128,
    tab_embed=64,
    img_embed=128,
    fusion_hidden=256,
    dropout=0.35,
).to(device)

criterion = create_criterion(n_classes=99, label_smoothing=0.1)
optimizer = create_optimizer(model, lr=1e-3, weight_decay=5e-4)
total_epochs = 80
scheduler = create_scheduler(optimizer, total_epochs=total_epochs, warmup_epochs=3)

best_val_score = float("inf")
best_model_state = None
patience = 15
patience_counter = 0
scaler_amp = torch.cuda.amp.GradScaler()

# ============================================================
# 7. Training Loop
# ============================================================
print("Starting training...")
start_time = time.time()

for epoch in range(total_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        margin_b, shape_b, texture_b, img_b, y_b = batch
        margin_b, shape_b, texture_b, img_b, y_b = (
            margin_b.to(device),
            shape_b.to(device),
            texture_b.to(device),
            img_b.to(device),
            y_b.to(device),
        )

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(margin_b, shape_b, texture_b, img_b)
            loss = criterion(logits, y_b)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        train_loss += loss.item() * margin_b.size(0)
        _, preds = torch.max(logits, 1)
        train_total += y_b.size(0)
        train_correct += (preds == y_b).sum().item()

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            margin_b, shape_b, texture_b, img_b, y_b = batch
            margin_b, shape_b, texture_b, img_b, y_b = (
                margin_b.to(device),
                shape_b.to(device),
                texture_b.to(device),
                img_b.to(device),
                y_b.to(device),
            )
            with torch.cuda.amp.autocast():
                logits = model(margin_b, shape_b, texture_b, img_b)
                probs = torch.softmax(logits, dim=1)
                loss = criterion(logits, y_b)

            val_loss += loss.item() * margin_b.size(0)
            all_val_preds.append(probs.cpu().numpy())
            all_val_labels.append(y_b.cpu().numpy())

    val_preds = np.vstack(all_val_preds)
    val_labels = np.concatenate(all_val_labels)

    eps = 1e-15
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_logloss = log_loss(val_labels, val_preds_clipped, labels=range(99))
    train_loss_avg = train_loss / train_total
    train_acc = train_correct / train_total

    print(
        f"Epoch {epoch+1}/{total_epochs} | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | Val LogLoss: {val_logloss:.4f}"
    )

    if val_logloss < best_val_score:
        best_val_score = val_logloss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(best_model_state)
print(f"Best validation log loss: {best_val_score:.6f}")

# ============================================================
# 8. Test Inference
# ============================================================
model.eval()
test_preds = []

with torch.no_grad():
    for batch in test_loader:
        margin_b, shape_b, texture_b, img_b = batch
        margin_b, shape_b, texture_b, img_b = (
            margin_b.to(device),
            shape_b.to(device),
            texture_b.to(device),
            img_b.to(device),
        )
        with torch.cuda.amp.autocast():
            logits = model(margin_b, shape_b, texture_b, img_b)
            probs = torch.softmax(logits, dim=1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.vstack(test_preds)
print(f"Test predictions shape: {test_preds.shape}")

# ============================================================
# 9. Create Submission
# ============================================================
submission_cols = sample_sub.columns[1:].tolist()
label_to_idx = {species: idx for idx, species in enumerate(label_encoder.classes_)}
submission_order_idx = [label_to_idx[sp] for sp in submission_cols]

test_preds_reordered = test_preds[:, submission_order_idx]
eps = 1e-15
test_preds_clipped = np.clip(test_preds_reordered, eps, 1 - eps)
row_sums = test_preds_clipped.sum(axis=1, keepdims=True)
test_preds_normalized = test_preds_clipped / row_sums

submission_df = pd.DataFrame(test_preds_normalized, columns=submission_cols)
submission_df.insert(0, "id", test_ids.astype(int))
submission_df = submission_df[sample_sub.columns]

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

# ============================================================
# 10. Final Validation Score
# ============================================================
model.eval()
all_val_final_preds = []
all_val_final_labels = []

with torch.no_grad():
    for batch in val_loader:
        margin_b, shape_b, texture_b, img_b, y_b = batch
        margin_b, shape_b, texture_b, img_b, y_b = (
            margin_b.to(device),
            shape_b.to(device),
            texture_b.to(device),
            img_b.to(device),
            y_b.to(device),
        )
        with torch.cuda.amp.autocast():
            logits = model(margin_b, shape_b, texture_b, img_b)
            probs = torch.softmax(logits, dim=1)
        all_val_final_preds.append(probs.cpu().numpy())
        all_val_final_labels.append(y_b.cpu().numpy())

val_final_preds = np.vstack(all_val_final_preds)
val_final_labels = np.concatenate(all_val_final_labels)
val_final_preds_clipped = np.clip(val_final_preds, eps, 1 - eps)
val_row_sums = val_final_preds_clipped.sum(axis=1, keepdims=True)
val_final_preds_normalized = val_final_preds_clipped / val_row_sums

final_val_score = log_loss(
    val_final_labels, val_final_preds_normalized, labels=range(99)
)
print(f"Final Validation Score: {final_val_score}")
