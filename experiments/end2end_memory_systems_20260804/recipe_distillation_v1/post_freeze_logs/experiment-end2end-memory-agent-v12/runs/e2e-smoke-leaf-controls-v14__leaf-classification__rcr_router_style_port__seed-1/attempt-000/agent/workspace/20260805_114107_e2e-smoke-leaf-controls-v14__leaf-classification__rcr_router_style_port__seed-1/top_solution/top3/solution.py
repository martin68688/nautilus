import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# ============================================================
# 2. IDENTIFY FEATURE GROUPS
# ============================================================
margin_cols = [f"margin{i}" for i in range(1, 65)]
shape_cols = [f"shape{i}" for i in range(1, 65)]
texture_cols = [f"texture{i}" for i in range(1, 65)]
all_feature_cols = margin_cols + shape_cols + texture_cols

# ============================================================
# 3. ENCODE TARGET LABELS
# ============================================================
label_encoder = LabelEncoder()
train_df["species_encoded"] = label_encoder.fit_transform(train_df["species"])
num_classes = len(label_encoder.classes_)

# ============================================================
# 4. CREATE TRAIN/VALIDATION SPLIT (Stratified, Index-Safe)
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(train_df, train_df["species_encoded"])))

# Keep original indices - do NOT reset_index
train_set = train_df.iloc[train_idx].copy()
val_set = train_df.iloc[val_idx].copy()

assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap detected!"


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================
def create_engineered_features(
    df,
    margin_cols,
    shape_cols,
    texture_cols,
    is_train=True,
    scaler=None,
    pca_models=None,
    raw_scaler=None,
):
    result = pd.DataFrame(index=df.index)
    group_dict = {"margin": margin_cols, "shape": shape_cols, "texture": texture_cols}

    for group_name, cols in group_dict.items():
        group_data = df[cols].values
        result[f"{group_name}_mean"] = np.mean(group_data, axis=1)
        result[f"{group_name}_std"] = np.std(group_data, axis=1)
        result[f"{group_name}_min"] = np.min(group_data, axis=1)
        result[f"{group_name}_max"] = np.max(group_data, axis=1)
        result[f"{group_name}_range"] = (
            result[f"{group_name}_max"] - result[f"{group_name}_min"]
        )
        centered = group_data - result[f"{group_name}_mean"].values[:, None]
        std_safe = np.maximum(result[f"{group_name}_std"].values, 1e-10)[:, None]
        skew = np.mean((centered / std_safe) ** 3, axis=1)
        result[f"{group_name}_skew"] = skew

    result["margin_shape_ratio"] = result["margin_mean"] / (
        np.abs(result["shape_mean"]) + 1e-10
    )
    result["margin_texture_ratio"] = result["margin_mean"] / (
        np.abs(result["texture_mean"]) + 1e-10
    )
    result["shape_texture_ratio"] = result["shape_mean"] / (
        np.abs(result["texture_mean"]) + 1e-10
    )

    if is_train:
        pca_models = {}
    for group_name, cols in group_dict.items():
        group_data = df[cols].values
        if is_train:
            pca = PCA(n_components=32, random_state=42)
            pca_components = pca.fit_transform(group_data)
            pca_models[group_name] = pca
        else:
            pca = pca_models[group_name]
            pca_components = pca.transform(group_data)
        for i in range(pca_components.shape[1]):
            result[f"{group_name}_pca_{i+1}"] = pca_components[:, i]

    engineered_feature_cols = [c for c in result.columns if c != "id"]
    if is_train:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(result[engineered_feature_cols])
    else:
        scaled_values = scaler.transform(result[engineered_feature_cols])
    scaled_df = pd.DataFrame(
        scaled_values, columns=engineered_feature_cols, index=result.index
    )

    raw_feature_df = df[all_feature_cols].copy()
    if is_train:
        raw_scaler = StandardScaler()
        raw_scaled = raw_scaler.fit_transform(raw_feature_df.values)
        raw_feature_scaled_df = pd.DataFrame(
            raw_scaled, columns=all_feature_cols, index=raw_feature_df.index
        )
    else:
        raw_scaled = raw_scaler.transform(raw_feature_df.values)
        raw_feature_scaled_df = pd.DataFrame(
            raw_scaled, columns=all_feature_cols, index=raw_feature_df.index
        )

    final_features = pd.concat([raw_feature_scaled_df, scaled_df], axis=1)
    final_features["image_path"] = df["id"].apply(
        lambda x: (
            os.path.join("./input/images", f"{x}.jpg")
            if os.path.exists(os.path.join("./input/images", f"{x}.jpg"))
            else None
        )
    )
    return final_features, scaler, pca_models, raw_scaler


X_train_engineered, scaler, pca_models, raw_scaler = create_engineered_features(
    train_set, margin_cols, shape_cols, texture_cols, is_train=True
)
X_val_engineered, _, _, _ = create_engineered_features(
    val_set,
    margin_cols,
    shape_cols,
    texture_cols,
    is_train=False,
    scaler=scaler,
    pca_models=pca_models,
    raw_scaler=raw_scaler,
)
X_test_engineered, _, _, _ = create_engineered_features(
    test_df,
    margin_cols,
    shape_cols,
    texture_cols,
    is_train=False,
    scaler=scaler,
    pca_models=pca_models,
    raw_scaler=raw_scaler,
)

# ============================================================
# 6. PREPARE FEATURE ARRAYS
# ============================================================
train_image_paths = X_train_engineered["image_path"].values
val_image_paths = X_val_engineered["image_path"].values
test_image_paths = X_test_engineered["image_path"].values

feature_cols_final = [
    c for c in X_train_engineered.columns if c not in ["image_path", "id"]
]
X_train = X_train_engineered[feature_cols_final].values.astype(np.float32)
X_val = X_val_engineered[feature_cols_final].values.astype(np.float32)
X_test = X_test_engineered[feature_cols_final].values.astype(np.float32)

y_train = train_set["species_encoded"].values
y_val = val_set["species_encoded"].values
test_ids = test_df["id"].values

print(
    f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
)
print(f"Number of classes: {num_classes}")


# ============================================================
# 7. MODEL ARCHITECTURE
# ============================================================
class BranchEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TabularImageFusionModel(nn.Module):
    def __init__(
        self,
        margin_dim=64,
        shape_dim=64,
        texture_dim=64,
        image_dim=1152,
        num_classes=99,
        hidden_dim=256,
        dropout=0.3,
    ):
        super().__init__()
        self.margin_encoder = BranchEncoder(
            margin_dim, hidden_dim=hidden_dim // 2, dropout=dropout
        )
        self.shape_encoder = BranchEncoder(
            shape_dim, hidden_dim=hidden_dim // 2, dropout=dropout
        )
        self.texture_encoder = BranchEncoder(
            texture_dim, hidden_dim=hidden_dim // 2, dropout=dropout
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        total_dim = (hidden_dim // 2) * 3 + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, margin, shape, texture, image_feat):
        margin_emb = self.margin_encoder(margin)
        shape_emb = self.shape_encoder(shape)
        texture_emb = self.texture_encoder(texture)
        image_emb = self.image_proj(image_feat)
        fused = torch.cat([margin_emb, shape_emb, texture_emb, image_emb], dim=1)
        return self.fusion(fused)


# ============================================================
# 8. TRAINING SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TabularImageFusionModel(
    num_classes=num_classes, hidden_dim=256, dropout=0.3
).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model created with {total_params:,} trainable parameters")

# ============================================================
# 9. IMAGE FEATURE EXTRACTION (SigLIP2, Frozen)
# ============================================================
from transformers import AutoProcessor, AutoModel

siglip_processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-256")
siglip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
siglip_model.to(device)
siglip_model.eval()
for p in siglip_model.parameters():
    p.requires_grad = False


def extract_image_features(image_paths, batch_size=16):
    all_feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        valid_idx = []
        for j, p in enumerate(batch_paths):
            if p is not None and os.path.exists(p):
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_idx.append(j)
        if not images:
            all_feats.append(np.zeros((len(batch_paths), 1152), dtype=np.float32))
            continue
        inputs = siglip_processor(
            images=images,
            return_tensors="pt",
            size={"height": 256, "width": 256},
            do_resize=True,
            do_center_crop=True,
        ).to(device)
        with torch.no_grad():
            feats = siglip_model.get_image_features(**inputs).float().cpu().numpy()
        full_batch = np.zeros((len(batch_paths), 1152), dtype=np.float32)
        for j, f in zip(valid_idx, feats):
            full_batch[j] = f
        all_feats.append(full_batch)
    return np.concatenate(all_feats, axis=0)


print("Extracting SigLIP2 image features...")
train_img_feats = extract_image_features(train_image_paths)
val_img_feats = extract_image_features(val_image_paths)
test_img_feats = extract_image_features(test_image_paths)

del siglip_model, siglip_processor
torch.cuda.empty_cache()
print(
    f"Image features shape: train={train_img_feats.shape}, val={val_img_feats.shape}, test={test_img_feats.shape}"
)

# ============================================================
# 10. DATA TENSOR PREPARATION
# ============================================================
raw_feature_dim = 192
X_train_raw = X_train[:, :raw_feature_dim]
X_val_raw = X_val[:, :raw_feature_dim]
X_test_raw = X_test[:, :raw_feature_dim]

train_margin = torch.tensor(X_train_raw[:, 0:64], dtype=torch.float32)
train_shape = torch.tensor(X_train_raw[:, 64:128], dtype=torch.float32)
train_texture = torch.tensor(X_train_raw[:, 128:192], dtype=torch.float32)
train_image = torch.tensor(train_img_feats, dtype=torch.float32)

val_margin = torch.tensor(X_val_raw[:, 0:64], dtype=torch.float32)
val_shape = torch.tensor(X_val_raw[:, 64:128], dtype=torch.float32)
val_texture = torch.tensor(X_val_raw[:, 128:192], dtype=torch.float32)
val_image = torch.tensor(val_img_feats, dtype=torch.float32)

test_margin = torch.tensor(X_test_raw[:, 0:64], dtype=torch.float32)
test_shape = torch.tensor(X_test_raw[:, 64:128], dtype=torch.float32)
test_texture = torch.tensor(X_test_raw[:, 128:192], dtype=torch.float32)
test_image = torch.tensor(test_img_feats, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(
    train_margin, train_shape, train_texture, train_image, y_train_tensor
)
val_dataset = TensorDataset(val_margin, val_shape, val_texture, val_image, y_val_tensor)
test_dataset = TensorDataset(test_margin, test_shape, test_texture, test_image)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)


# ============================================================
# 11. LOG LOSS METRIC
# ============================================================
def compute_log_loss(y_true, y_pred_proba, eps=1e-15):
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    n = len(y_true)
    loss = 0.0
    for i in range(n):
        loss += -np.log(y_pred_proba[i, y_true[i]])
    return loss / n


# ============================================================
# 12. TRAINING LOOP
# ============================================================
best_val_loss = float("inf")
best_epoch = 0
patience = 15
patience_counter = 0
num_epochs = 60
model_save_path = "./working/best_model.pt"
os.makedirs("./working", exist_ok=True)

scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-5
)

print("Starting training...")
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for margin_b, shape_b, texture_b, image_b, y_b in train_loader:
        margin_b, shape_b, texture_b, image_b, y_b = (
            margin_b.to(device),
            shape_b.to(device),
            texture_b.to(device),
            image_b.to(device),
            y_b.to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(margin_b, shape_b, texture_b, image_b)
            loss = criterion(logits, y_b)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * len(y_b)
        preds = logits.argmax(dim=1)
        train_correct += (preds == y_b).sum().item()
        train_total += len(y_b)

    scheduler.step()

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for margin_b, shape_b, texture_b, image_b, y_b in val_loader:
            margin_b, shape_b, texture_b, image_b = (
                margin_b.to(device),
                shape_b.to(device),
                texture_b.to(device),
                image_b.to(device),
            )
            with torch.cuda.amp.autocast():
                logits = model(margin_b, shape_b, texture_b, image_b)
                loss = criterion(logits, y_b.to(device))
                probs = torch.softmax(logits, dim=1)
            val_loss += loss.item() * len(y_b)
            preds = logits.argmax(dim=1)
            val_correct += (preds == y_b.to(device)).sum().item()
            val_total += len(y_b)
            all_val_probs.append(probs.float().cpu().numpy())
            all_val_labels.append(y_b.numpy())

    val_loss_avg = val_loss / val_total
    val_acc = val_correct / val_total
    val_probs_np = np.concatenate(all_val_probs, axis=0)
    val_labels_np = np.concatenate(all_val_labels, axis=0)
    val_logloss = compute_log_loss(val_labels_np, val_probs_np)

    print(
        f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss/train_total:.4f} | Val Loss: {val_loss_avg:.4f} | Val LogLoss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), model_save_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"Best validation logloss: {best_val_loss:.4f} at epoch {best_epoch}")

# ============================================================
# 13. LOAD BEST MODEL & TEST INFERENCE
# ============================================================
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

all_val_probs = []
with torch.no_grad():
    for margin_b, shape_b, texture_b, image_b, _ in val_loader:
        margin_b, shape_b, texture_b, image_b = (
            margin_b.to(device),
            shape_b.to(device),
            texture_b.to(device),
            image_b.to(device),
        )
        with torch.cuda.amp.autocast():
            logits = model(margin_b, shape_b, texture_b, image_b)
            probs = torch.softmax(logits, dim=1)
        all_val_probs.append(probs.float().cpu().numpy())

val_probs_final = np.concatenate(all_val_probs, axis=0)
val_probs_final = np.clip(val_probs_final, 1e-15, 1 - 1e-15)
val_scores_final = compute_log_loss(y_val, val_probs_final)

all_test_probs = []
with torch.no_grad():
    for margin_b, shape_b, texture_b, image_b in test_loader:
        margin_b, shape_b, texture_b, image_b = (
            margin_b.to(device),
            shape_b.to(device),
            texture_b.to(device),
            image_b.to(device),
        )
        with torch.cuda.amp.autocast():
            logits = model(margin_b, shape_b, texture_b, image_b)
            probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.float().cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

# ============================================================
# 14. SUBMISSION FILE
# ============================================================
species_cols = [c for c in sample_sub.columns if c != "id"]
submission_df = pd.DataFrame(test_probs, columns=species_cols)
submission_df.insert(0, "id", test_ids)
submission_df = submission_df[["id"] + species_cols]

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(
    f"Submission saved to ./submission/submission.csv with shape {submission_df.shape}"
)

# ============================================================
# 15. FINAL VERIFICATION
# ============================================================
assert submission_df.shape == (
    len(test_ids),
    100,
), f"Unexpected submission shape: {submission_df.shape}"
assert list(submission_df.columns) == ["id"] + species_cols, "Column mismatch!"
assert not submission_df.isnull().any().any(), "NaN in submission!"
assert (submission_df.iloc[:, 1:].values >= 0).all(), "Negative probabilities!"
assert (submission_df.iloc[:, 1:].values <= 1).all(), "Probabilities > 1!"
row_sums = submission_df.iloc[:, 1:].sum(axis=1)
assert np.allclose(
    row_sums, 1.0, atol=1e-3
), f"Row sums not 1: min={row_sums.min()}, max={row_sums.max()}"

print(f"Submission verified: {len(test_ids)} rows × {len(species_cols)} species")
print(f"Final Validation Score: {val_scores_final}")
