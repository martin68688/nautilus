import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats
import os
import json
import warnings

warnings.filterwarnings("ignore")

# ===== Configuration =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Load Data =====
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# Feature column names (actual CSV structure: no underscores)
margin_cols = [f"margin{i}" for i in range(1, 65)]
shape_cols = [f"shape{i}" for i in range(1, 65)]
texture_cols = [f"texture{i}" for i in range(1, 65)]
all_feature_cols = margin_cols + shape_cols + texture_cols

# Species columns for submission
species_cols = sample_sub.columns[1:].tolist()


# ===== Feature Engineering =====
def engineer_tabular_features(df):
    df = df.copy()
    for prefix, cols in [
        ("margin", margin_cols),
        ("shape", shape_cols),
        ("texture", texture_cols),
    ]:
        group_data = df[cols].values
        df[f"{prefix}_mean"] = group_data.mean(axis=1)
        df[f"{prefix}_std"] = group_data.std(axis=1)
        df[f"{prefix}_max"] = group_data.max(axis=1)
        df[f"{prefix}_min"] = group_data.min(axis=1)
        df[f"{prefix}_range"] = df[f"{prefix}_max"] - df[f"{prefix}_min"]
        df[f"{prefix}_skew"] = stats.skew(group_data, axis=1)
        df[f"{prefix}_kurtosis"] = stats.kurtosis(group_data, axis=1)
        df[f"{prefix}_median"] = np.median(group_data, axis=1)
        df[f"{prefix}_p25"] = np.percentile(group_data, 25, axis=1)
        df[f"{prefix}_p75"] = np.percentile(group_data, 75, axis=1)
        df[f"{prefix}_l2"] = np.linalg.norm(df[cols].values, axis=1)

    df["margin_shape_l2_ratio"] = df["margin_l2"] / (df["shape_l2"] + 1e-8)
    df["margin_texture_l2_ratio"] = df["margin_l2"] / (df["texture_l2"] + 1e-8)
    df["shape_texture_l2_ratio"] = df["shape_l2"] / (df["texture_l2"] + 1e-8)
    return df


train_engineered = engineer_tabular_features(train_df)
test_engineered = engineer_tabular_features(test_df)

additional_feats = [
    c for c in train_engineered.columns if c not in ["id", "species"] + all_feature_cols
]

# PCA on each feature group (fit on train only)
pca_features_train = []
pca_features_test = []
for prefix, cols in [
    ("margin", margin_cols),
    ("shape", shape_cols),
    ("texture", texture_cols),
]:
    pca = PCA(n_components=16, random_state=42)
    pca_train = pca.fit_transform(train_engineered[cols].values)
    pca_test = pca.transform(test_engineered[cols].values)
    pca_features_train.append(
        pd.DataFrame(pca_train, columns=[f"{prefix}_pca_{i+1}" for i in range(16)])
    )
    pca_features_test.append(
        pd.DataFrame(pca_test, columns=[f"{prefix}_pca_{i+1}" for i in range(16)])
    )

# Combine features
train_X_raw = np.hstack(
    [
        train_engineered[all_feature_cols].values,
        train_engineered[additional_feats].values,
        np.hstack([pca.values for pca in pca_features_train]),
    ]
)
test_X_raw = np.hstack(
    [
        test_engineered[all_feature_cols].values,
        test_engineered[additional_feats].values,
        np.hstack([pca.values for pca in pca_features_test]),
    ]
)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(train_df["species"])

# Stratified split (avoid index bug)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(train_X_raw, y_encoded)))

X_train_raw = train_X_raw[train_idx]
X_val_raw = train_X_raw[val_idx]
y_train = y_encoded[train_idx]
y_val = y_encoded[val_idx]

# Standardize (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(test_X_raw)

# Image paths
train_img_paths = [
    f"./input/images/{img_id}.jpg"
    for img_id in train_df["id"].iloc[train_idx].astype(str)
]
val_img_paths = [
    f"./input/images/{img_id}.jpg"
    for img_id in train_df["id"].iloc[val_idx].astype(str)
]
test_img_paths = [
    f"./input/images/{img_id}.jpg" for img_id in test_df["id"].astype(str)
]
test_ids = test_df["id"].values

assert len(set(train_idx) & set(val_idx)) == 0, "Data leakage detected!"
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ===== Image Preprocessing =====
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

# ===== Load SigLIP2 Model =====
print("Loading SigLIP2 model...")
from transformers import AutoModel

siglip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
vision_model = (
    siglip_model.vision_model if hasattr(siglip_model, "vision_model") else siglip_model
)
for param in vision_model.parameters():
    param.requires_grad = False
vision_model.to(device)
vision_model.eval()
IMAGE_FEATURE_DIM = 1152


# ===== Extract Image Features =====
def load_image_features(paths, batch_size=32):
    all_features = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(transform(img))
            except:
                batch_images.append(torch.zeros(3, 256, 256))
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                features = vision_model(pixel_values=batch_tensor)
                if hasattr(features, "pooler_output"):
                    feat = features.pooler_output
                elif hasattr(features, "last_hidden_state"):
                    feat = features.last_hidden_state[:, 0]
                else:
                    feat = features
                if isinstance(feat, tuple):
                    feat = feat[0]
                all_features.append(feat.cpu().numpy())
    return np.concatenate(all_features, axis=0)


print("Extracting image features...")
train_img_features = load_image_features(train_img_paths)
val_img_features = load_image_features(val_img_paths)
test_img_features = load_image_features(test_img_paths)
print(
    f"Image features - Train: {train_img_features.shape}, Val: {val_img_features.shape}, Test: {test_img_features.shape}"
)


# ===== Model Definition =====
class MultiModalLeafClassifier(nn.Module):
    def __init__(self, n_features, n_classes=99, image_dim=1152, hidden_dim=256):
        super().__init__()
        self.tabular_encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, tabular_features, image_features=None):
        tab_out = self.tabular_encoder(tabular_features)
        if image_features is not None:
            img_out = self.image_proj(image_features)
            fused = torch.cat([tab_out, img_out], dim=1)
            fused = self.fusion(fused)
        else:
            fused = self.fusion(torch.cat([tab_out, tab_out], dim=1))
        return self.classifier(fused)


# ===== Initialize Model =====
n_features = X_train.shape[1]
model = MultiModalLeafClassifier(
    n_features=n_features, n_classes=99, image_dim=IMAGE_FEATURE_DIM, hidden_dim=256
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)

num_epochs = 150
patience = 15
best_val_loss = float("inf")
best_epoch = 0
no_improve = 0
scaler_amp = torch.cuda.amp.GradScaler()

# ===== Data Loaders =====
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.FloatTensor(train_img_features),
    torch.LongTensor(y_train),
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val),
    torch.FloatTensor(val_img_features),
    torch.LongTensor(y_val),
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test), torch.FloatTensor(test_img_features)
)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)

# ===== Training Loop =====
print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_tab, batch_img, batch_labels in train_loader:
        batch_tab, batch_img, batch_labels = (
            batch_tab.to(device),
            batch_img.to(device),
            batch_labels.to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(batch_tab, batch_img)
            loss = criterion(logits, batch_labels)
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    # Validation
    model.eval()
    val_loss_sum = 0
    val_batches = 0
    all_val_probs = []

    with torch.no_grad():
        for batch_tab, batch_img, batch_labels in val_loader:
            batch_tab, batch_img = batch_tab.to(device), batch_img.to(device)
            with torch.cuda.amp.autocast():
                logits = model(batch_tab, batch_img)
                loss = criterion(logits, batch_labels.to(device))
                probs = F.softmax(logits, dim=1)
            val_loss_sum += loss.item()
            val_batches += 1
            all_val_probs.append(probs.cpu().numpy())

    avg_train_loss = total_loss / n_batches
    avg_val_loss = val_loss_sum / val_batches
    val_probs_all = np.concatenate(all_val_probs, axis=0)
    val_probs_clipped = np.clip(val_probs_all, 1e-15, 1 - 1e-15)
    val_logloss = -np.mean(np.log(val_probs_clipped[np.arange(len(y_val)), y_val]))

    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val LogLoss: {val_logloss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}"
    )

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_epoch = epoch + 1
        no_improve = 0
        torch.save(model.state_dict(), "./working/best_model.pth")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}, best epoch was {best_epoch}")
            break

# ===== Load Best Model =====
print(
    f"Loading best model from epoch {best_epoch} with val log loss {best_val_loss:.4f}"
)
model.load_state_dict(torch.load("./working/best_model.pth"))
model.eval()

# ===== Final Validation =====
val_probs_list = []
with torch.no_grad():
    for batch_tab, batch_img, _ in val_loader:
        batch_tab, batch_img = batch_tab.to(device), batch_img.to(device)
        with torch.cuda.amp.autocast():
            logits = model(batch_tab, batch_img)
            probs = F.softmax(logits, dim=1)
        val_probs_list.append(probs.cpu().numpy())
val_probs_final = np.concatenate(val_probs_list, axis=0)
val_probs_clipped_final = np.clip(val_probs_final, 1e-15, 1 - 1e-15)
val_logloss_final = -np.mean(
    np.log(val_probs_clipped_final[np.arange(len(y_val)), y_val])
)

# ===== Test Inference =====
test_probs_list = []
with torch.no_grad():
    for batch_tab, batch_img in test_loader:
        batch_tab, batch_img = batch_tab.to(device), batch_img.to(device)
        with torch.cuda.amp.autocast():
            logits = model(batch_tab, batch_img)
            probs = F.softmax(logits, dim=1)
        test_probs_list.append(probs.cpu().numpy())
test_probs = np.concatenate(test_probs_list, axis=0)

# ===== Generate Submission =====
test_probs_clipped = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs_norm = test_probs_clipped / test_probs_clipped.sum(axis=1, keepdims=True)

submission = pd.DataFrame(test_probs_norm, columns=species_cols)
submission.insert(0, "id", test_ids)
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv with shape {submission.shape}")

print(f"Final Validation Score: {val_logloss_final}")
