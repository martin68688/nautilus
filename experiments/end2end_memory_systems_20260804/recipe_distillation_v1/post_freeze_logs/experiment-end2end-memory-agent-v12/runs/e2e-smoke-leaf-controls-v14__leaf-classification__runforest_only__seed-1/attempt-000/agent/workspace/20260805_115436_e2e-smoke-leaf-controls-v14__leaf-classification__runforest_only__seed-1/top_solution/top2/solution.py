import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import pickle
import time

# =====================================================
# 1. DATA PROCESSING & FEATURE ENGINEERING
# =====================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# Feature column groups (no underscores!)
margin_cols = [f"margin{i}" for i in range(1, 65)]
shape_cols = [f"shape{i}" for i in range(1, 65)]
texture_cols = [f"texture{i}" for i in range(1, 65)]
all_feature_cols = margin_cols + shape_cols + texture_cols
species_cols = sample_sub.columns[1:].tolist()

# Encode labels
label_encoder = LabelEncoder()
train_df["species_encoded"] = label_encoder.fit_transform(train_df["species"])


def engineer_features(df):
    """Create enhanced features from raw margin/shape/texture features."""
    features = {}

    # Raw features
    for col in all_feature_cols:
        features[col] = df[col].values

    # Per-modality statistics
    for group_name, cols in [
        ("margin", margin_cols),
        ("shape", shape_cols),
        ("texture", texture_cols),
    ]:
        data = df[cols].values
        features[f"{group_name}_mean"] = data.mean(axis=1)
        features[f"{group_name}_std"] = data.std(axis=1)
        features[f"{group_name}_skew"] = skew(data, axis=1)
        features[f"{group_name}_kurtosis"] = kurtosis(data, axis=1)
        features[f"{group_name}_min"] = data.min(axis=1)
        features[f"{group_name}_max"] = data.max(axis=1)
        features[f"{group_name}_median"] = np.median(data, axis=1)
        features[f"{group_name}_p25"] = np.percentile(data, 25, axis=1)
        features[f"{group_name}_p75"] = np.percentile(data, 75, axis=1)
        features[f"{group_name}_range"] = data.max(axis=1) - data.min(axis=1)
        features[f"{group_name}_energy"] = np.sum(data**2, axis=1)
        features[f"{group_name}_entropy"] = -np.sum(
            data * np.log(np.clip(data, 1e-10, None)), axis=1
        )

    # Cross-modality correlations
    margin_data = df[margin_cols].values
    shape_data = df[shape_cols].values
    texture_data = df[texture_cols].values

    for name, (d1, d2) in [
        ("margin_shape", (margin_data, shape_data)),
        ("margin_texture", (margin_data, texture_data)),
        ("shape_texture", (shape_data, texture_data)),
    ]:
        d1_centered = d1 - d1.mean(axis=1, keepdims=True)
        d2_centered = d2 - d2.mean(axis=1, keepdims=True)
        numerator = (d1_centered * d2_centered).sum(axis=1)
        denom1 = np.sqrt((d1_centered**2).sum(axis=1))
        denom2 = np.sqrt((d2_centered**2).sum(axis=1))
        features[f"corr_{name}"] = numerator / (denom1 * denom2 + 1e-10)

    # Ratios of group means
    for g1, g2 in [("margin", "shape"), ("margin", "texture"), ("shape", "texture")]:
        features[f"ratio_{g1}_{g2}"] = features[f"{g1}_mean"] / (
            features[f"{g2}_mean"] + 1e-10
        )

    return pd.DataFrame(features)


# Apply feature engineering
X_train_eng = engineer_features(train_df)
X_test_eng = engineer_features(test_df)

# Add PCA components per modality (fit on train only)
for group_name, cols in [
    ("margin", margin_cols),
    ("shape", shape_cols),
    ("texture", texture_cols),
]:
    pca = PCA(n_components=16, random_state=42)
    train_pca = pca.fit_transform(train_df[cols].values)
    test_pca = pca.transform(test_df[cols].values)
    for i in range(16):
        X_train_eng[f"pca_{group_name}_{i}"] = train_pca[:, i]
        X_test_eng[f"pca_{group_name}_{i}"] = test_pca[:, i]

# Split training data (stratified)
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df["species_encoded"],
)

# Get labels directly from split indices (NO INDEX BUG)
y_train = train_df["species_encoded"].values[train_idx]
y_val = train_df["species_encoded"].values[val_idx]

# Get features for split
X_train = X_train_eng.iloc[train_idx].values
X_val = X_train_eng.iloc[val_idx].values
X_test = X_test_eng.values

# Scale features (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Verify no leakage
assert len(set(train_idx) & set(val_idx)) == 0, "Data leakage detected!"

print(f"Train shape: {X_train_scaled.shape}")
print(f"Val shape: {X_val_scaled.shape}")
print(f"Test shape: {X_test_scaled.shape}")
print(f"Number of classes: {len(label_encoder.classes_)}")


# =====================================================
# 2. MODEL DEFINITION
# =====================================================
class ModalityEncoder(nn.Module):
    """Encoder for a single feature modality."""

    def __init__(self, input_dim, hidden_dim=128, output_dim=64, dropout=0.3):
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
            nn.Linear(hidden_dim, output_dim),
        )
        self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class LeafClassifier(nn.Module):
    """Tabular-only leaf classifier using first 192 raw features."""

    def __init__(self, n_classes=99, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.margin_encoder = ModalityEncoder(64, hidden_dim=128, output_dim=64)
        self.shape_encoder = ModalityEncoder(64, hidden_dim=128, output_dim=64)
        self.texture_encoder = ModalityEncoder(64, hidden_dim=128, output_dim=64)

        fusion_dim = 64 * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, n_classes),
        )

    def forward(self, margin, shape, texture):
        m_emb = self.margin_encoder(margin)
        s_emb = self.shape_encoder(shape)
        t_emb = self.texture_encoder(texture)
        fused = torch.cat([m_emb, s_emb, t_emb], dim=1)
        return self.fusion(fused)


# =====================================================
# 3. TRAINING & EVALUATION
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Use only raw features (first 192) for the modality encoders
X_train_raw = X_train_scaled[:, :192]
X_val_raw = X_val_scaled[:, :192]
X_test_raw = X_test_scaled[:, :192]

# Split into margin, shape, texture
X_train_margin, X_train_shape, X_train_texture = (
    X_train_raw[:, 0:64],
    X_train_raw[:, 64:128],
    X_train_raw[:, 128:192],
)
X_val_margin, X_val_shape, X_val_texture = (
    X_val_raw[:, 0:64],
    X_val_raw[:, 64:128],
    X_val_raw[:, 128:192],
)
X_test_margin, X_test_shape, X_test_texture = (
    X_test_raw[:, 0:64],
    X_test_raw[:, 64:128],
    X_test_raw[:, 128:192],
)

# Model setup
model = LeafClassifier(n_classes=99).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=5e-6)

# Create DataLoaders
train_dataset = TensorDataset(
    torch.tensor(X_train_margin, dtype=torch.float32),
    torch.tensor(X_train_shape, dtype=torch.float32),
    torch.tensor(X_train_texture, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_margin, dtype=torch.float32),
    torch.tensor(X_val_shape, dtype=torch.float32),
    torch.tensor(X_val_texture, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long),
)
test_dataset = TensorDataset(
    torch.tensor(X_test_margin, dtype=torch.float32),
    torch.tensor(X_test_shape, dtype=torch.float32),
    torch.tensor(X_test_texture, dtype=torch.float32),
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

# Mixed precision
scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


def compute_log_loss(y_true, y_pred_probs):
    """Compute multi-class log loss."""
    eps = 1e-15
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)
    y_pred_probs = y_pred_probs / y_pred_probs.sum(axis=1, keepdims=True)
    n = len(y_true)
    loss = 0.0
    for i in range(n):
        loss += -np.log(y_pred_probs[i, y_true[i]])
    return loss / n


# Training loop
best_val_score = float("inf")
best_model_path = "./working/best_model.pt"
os.makedirs("./working", exist_ok=True)
patience = 20
patience_counter = 0
n_epochs = 60
start_time = time.time()

for epoch in range(n_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch_margin, batch_shape, batch_texture, batch_labels in train_loader:
        batch_margin = batch_margin.to(device)
        batch_shape = batch_shape.to(device)
        batch_texture = batch_texture.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(batch_margin, batch_shape, batch_texture)
            loss = criterion(logits, batch_labels)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        train_loss += loss.item() * len(batch_labels)

    train_loss /= len(train_dataset)

    # Validation
    model.eval()
    all_val_probs = []
    all_val_labels = []
    with torch.no_grad():
        for batch_margin, batch_shape, batch_texture, batch_labels in val_loader:
            batch_margin = batch_margin.to(device)
            batch_shape = batch_shape.to(device)
            batch_texture = batch_texture.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(batch_margin, batch_shape, batch_texture)
                probs = F.softmax(logits, dim=1)
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(batch_labels.numpy())

    val_probs = np.vstack(all_val_probs)
    val_labels = np.concatenate(all_val_labels)
    val_score = compute_log_loss(val_labels, val_probs)

    scheduler.step()
    print(
        f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val LogLoss: {val_score:.4f}"
    )

    # Early stopping
    if val_score < best_val_score:
        best_val_score = val_score
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# Final validation evaluation
all_val_probs = []
with torch.no_grad():
    for batch_margin, batch_shape, batch_texture, _ in val_loader:
        batch_margin = batch_margin.to(device)
        batch_shape = batch_shape.to(device)
        batch_texture = batch_texture.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(batch_margin, batch_shape, batch_texture)
            probs = F.softmax(logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())

val_probs_final = np.vstack(all_val_probs)
final_val_score = compute_log_loss(y_val, val_probs_final)

# Test inference
print("Generating test predictions...")
all_test_probs = []
with torch.no_grad():
    for batch_margin, batch_shape, batch_texture in test_loader:
        batch_margin = batch_margin.to(device)
        batch_shape = batch_shape.to(device)
        batch_texture = batch_texture.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(batch_margin, batch_shape, batch_texture)
            probs = F.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.vstack(all_test_probs)

# Create submission file
species_labels = label_encoder.classes_.tolist()
species_to_idx = {species: idx for idx, species in enumerate(species_labels)}
col_indices = [species_to_idx[col] for col in species_cols]

test_probs_reordered = test_probs[:, col_indices]
eps = 1e-15
test_probs_clipped = np.clip(test_probs_reordered, eps, 1 - eps)
test_probs_clipped = test_probs_clipped / test_probs_clipped.sum(axis=1, keepdims=True)

submission_df = pd.DataFrame(test_probs_clipped, columns=species_cols)
submission_df.insert(0, "id", test_df["id"].values)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Total training time: {time.time() - start_time:.1f}s")
print(f"Final Validation Score: {final_val_score}")
