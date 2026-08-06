import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft
import time

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# Identify feature columns (actual names: margin1-64, shape1-64, texture1-64)
margin_cols = [c for c in train_df.columns if c.startswith("margin")]
shape_cols = [c for c in train_df.columns if c.startswith("shape")]
texture_cols = [c for c in train_df.columns if c.startswith("texture")]
feature_cols = margin_cols + shape_cols + texture_cols

print(f"Train: {train_df.shape}, Test: {test_df.shape}")


# ============================================================================
# 2. FEATURE ENGINEERING - Tabular features
# ============================================================================
def engineer_tabular_features(df, feature_cols, margin_cols, shape_cols, texture_cols):
    """Create rich statistical features from tabular leaf characteristics."""
    features = {}
    groups = {"margin": margin_cols, "shape": shape_cols, "texture": texture_cols}

    for group_name, cols in groups.items():
        X = df[cols].values
        features[f"{group_name}_mean"] = X.mean(axis=1)
        features[f"{group_name}_std"] = X.std(axis=1)
        features[f"{group_name}_min"] = X.min(axis=1)
        features[f"{group_name}_max"] = X.max(axis=1)
        features[f"{group_name}_median"] = np.median(X, axis=1)
        features[f"{group_name}_range"] = X.max(axis=1) - X.min(axis=1)
        features[f"{group_name}_skew"] = skew(X, axis=1)
        features[f"{group_name}_kurt"] = kurtosis(X, axis=1)

        def calc_entropy(row):
            row_abs = np.abs(row) + 1e-10
            row_norm = row_abs / row_abs.sum()
            return entropy(row_norm)

        features[f"{group_name}_entropy"] = np.apply_along_axis(calc_entropy, 1, X)

        X_fft = np.abs(fft(X, axis=1))
        features[f"{group_name}_fft_mean"] = X_fft[:, 1:].mean(axis=1)
        features[f"{group_name}_fft_std"] = X_fft[:, 1:].std(axis=1)
        features[f"{group_name}_fft_first"] = X_fft[:, 1]
        features[f"{group_name}_fft_energy"] = np.sum(X_fft[:, 1:] ** 2, axis=1)

        for p in [10, 25, 75, 90]:
            features[f"{group_name}_pct_{p}"] = np.percentile(X, p, axis=1)

    for g1 in groups:
        for g2 in groups:
            if g1 < g2:
                X1 = df[groups[g1]].values
                X2 = df[groups[g2]].values
                corrs = []
                for i in range(X1.shape[0]):
                    c = np.corrcoef(X1[i], X2[i])[0, 1]
                    corrs.append(c if not np.isnan(c) else 0)
                features[f"corr_{g1}_{g2}"] = corrs

    X_all = df[feature_cols].values
    features["all_mean"] = X_all.mean(axis=1)
    features["all_std"] = X_all.std(axis=1)
    features["all_norm"] = np.linalg.norm(X_all, axis=1)

    return pd.DataFrame(features)


print("Engineering tabular features...")
train_tab_features = engineer_tabular_features(
    train_df, feature_cols, margin_cols, shape_cols, texture_cols
)
test_tab_features = engineer_tabular_features(
    test_df, feature_cols, margin_cols, shape_cols, texture_cols
)
print(f"Tabular features: {train_tab_features.shape[1]} dimensions")

# Combine original + engineered features
train_tab = np.hstack([train_df[feature_cols].values, train_tab_features.values])
test_tab = np.hstack([test_df[feature_cols].values, test_tab_features.values])
print(f"Combined tabular: {train_tab.shape}")

# ============================================================================
# 3. CREATE TRAIN/VALIDATION SPLIT (Stratified)
# ============================================================================
species_list = sorted(train_df["species"].unique())
species_to_idx = {sp: i for i, sp in enumerate(species_list)}
train_labels = train_df["species"].map(species_to_idx).values

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(strat_split.split(train_df, train_labels))

print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
assert len(set(train_idx) & set(val_idx)) == 0, "Overlap in splits!"

# ============================================================================
# 4. SCALE FEATURES (fit on train only)
# ============================================================================
scaler = StandardScaler()
train_tab_scaled = scaler.fit_transform(train_tab[train_idx])
val_tab_scaled = scaler.transform(train_tab[val_idx])
test_tab_scaled = scaler.transform(test_tab)


# ============================================================================
# 5. MODEL ARCHITECTURE
# ============================================================================
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.fc1(x)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        self.encoder = nn.Sequential(*layers)
        self.se = SqueezeExcitation(prev_dim)
        self.output_dim = prev_dim

    def forward(self, x):
        out = self.encoder(x)
        out = self.se(out)
        return out


class LeafClassifier(nn.Module):
    def __init__(
        self,
        margin_dim=64,
        shape_dim=64,
        texture_dim=64,
        engineered_dim=0,
        num_classes=99,
        dropout=0.35,
    ):
        super().__init__()
        self.margin_encoder = TabularEncoder(margin_dim)
        self.shape_encoder = TabularEncoder(shape_dim)
        self.texture_encoder = TabularEncoder(texture_dim)
        self.engineered_dim = engineered_dim
        if engineered_dim > 0:
            self.engineered_encoder = nn.Sequential(
                nn.Linear(engineered_dim, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fusion_in = 128 + 128 + 128 + 128
        else:
            fusion_in = 128 + 128 + 128

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(64, num_classes)
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

    def forward(self, margin, shape, texture, engineered=None):
        margin_emb = self.margin_encoder(margin)
        shape_emb = self.shape_encoder(shape)
        texture_emb = self.texture_encoder(texture)
        parts = [margin_emb, shape_emb, texture_emb]
        if self.engineered_dim > 0 and engineered is not None:
            eng_emb = self.engineered_encoder(engineered)
            parts.append(eng_emb)
        fusion_input = torch.cat(parts, dim=1)
        fused = self.fusion(fusion_input)
        logits = self.classifier(fused)
        return logits


# ============================================================================
# 6. TRAINING SETUP
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

margin_dim = 64
shape_dim = 64
texture_dim = 64
engineered_dim = train_tab.shape[1] - 192
num_classes = len(species_list)

model = LeafClassifier(
    margin_dim=margin_dim,
    shape_dim=shape_dim,
    texture_dim=texture_dim,
    engineered_dim=engineered_dim,
    num_classes=num_classes,
    dropout=0.35,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
total_epochs = 60
scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

# ============================================================================
# 7. TRAINING LOOP
# ============================================================================
batch_size = 16
train_labels_tensor = torch.tensor(train_labels[train_idx], dtype=torch.long)
val_labels_tensor = torch.tensor(train_labels[val_idx], dtype=torch.long)

train_dataset = TensorDataset(
    torch.tensor(train_tab_scaled[:, 0:64], dtype=torch.float32),
    torch.tensor(train_tab_scaled[:, 64:128], dtype=torch.float32),
    torch.tensor(train_tab_scaled[:, 128:192], dtype=torch.float32),
    torch.tensor(train_tab_scaled[:, 192:], dtype=torch.float32),
    train_labels_tensor,
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)

val_dataset = TensorDataset(
    torch.tensor(val_tab_scaled[:, 0:64], dtype=torch.float32),
    torch.tensor(val_tab_scaled[:, 64:128], dtype=torch.float32),
    torch.tensor(val_tab_scaled[:, 128:192], dtype=torch.float32),
    torch.tensor(val_tab_scaled[:, 192:], dtype=torch.float32),
    val_labels_tensor,
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

best_val_score = float("inf")
best_epoch = -1
patience = 20

print("Starting training...")
for epoch in range(total_epochs):
    model.train()
    train_loss = 0.0
    train_total = 0

    for margin_b, shape_b, texture_b, eng_b, labels_b in train_loader:
        margin_b = margin_b.to(device)
        shape_b = shape_b.to(device)
        texture_b = texture_b.to(device)
        eng_b = eng_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        logits = model(margin_b, shape_b, texture_b, eng_b)
        loss = criterion(logits, labels_b)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * margin_b.size(0)
        train_total += labels_b.size(0)

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    val_total = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for margin_b, shape_b, texture_b, eng_b, labels_b in val_loader:
            margin_b = margin_b.to(device)
            shape_b = shape_b.to(device)
            texture_b = texture_b.to(device)
            eng_b = eng_b.to(device)
            labels_b = labels_b.to(device)

            logits = model(margin_b, shape_b, texture_b, eng_b)
            loss = criterion(logits, labels_b)

            val_loss += loss.item() * margin_b.size(0)
            probs = F.softmax(logits, dim=1)
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels_b.cpu().numpy())
            val_total += labels_b.size(0)

    val_probs_all = np.concatenate(all_val_probs, axis=0)
    val_labels_all = np.concatenate(all_val_labels, axis=0)

    eps = 1e-15
    val_probs_clipped = np.clip(val_probs_all, eps, 1 - eps)
    val_probs_norm = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

    n = len(val_labels_all)
    ll_sum = 0.0
    for i in range(n):
        ll_sum += -np.log(val_probs_norm[i, val_labels_all[i]])
    val_logloss = ll_sum / n

    if val_logloss < best_val_score:
        best_val_score = val_logloss
        best_epoch = epoch
        torch.save(model.state_dict(), "./working/best_model.pth")

    print(
        f"Epoch {epoch+1}/{total_epochs} | Train Loss: {train_loss/train_total:.4f} | Val LogLoss: {val_logloss:.4f}"
    )

    if epoch - best_epoch >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"Best validation log loss: {best_val_score:.6f} at epoch {best_epoch+1}")

# ============================================================================
# 8. TEST INFERENCE & SUBMISSION
# ============================================================================
model.load_state_dict(torch.load("./working/best_model.pth"))
model.eval()

test_dataset = TensorDataset(
    torch.tensor(test_tab_scaled[:, 0:64], dtype=torch.float32),
    torch.tensor(test_tab_scaled[:, 64:128], dtype=torch.float32),
    torch.tensor(test_tab_scaled[:, 128:192], dtype=torch.float32),
    torch.tensor(test_tab_scaled[:, 192:], dtype=torch.float32),
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

all_test_probs = []
with torch.no_grad():
    for margin_b, shape_b, texture_b, eng_b in test_loader:
        margin_b = margin_b.to(device)
        shape_b = shape_b.to(device)
        texture_b = texture_b.to(device)
        eng_b = eng_b.to(device)

        logits = model(margin_b, shape_b, texture_b, eng_b)
        probs = F.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)

# Create submission with exact column order from sample_submission
submission_cols = sample_sub.columns.tolist()
species_cols = [c for c in submission_cols if c != "id"]
test_ids = test_df["id"].values

submission_df = pd.DataFrame({"id": test_ids})
for i, sp in enumerate(species_cols):
    submission_df[sp] = test_probs[:, i]

submission_df = submission_df[submission_cols]

# Clip and normalize
prob_cols = [c for c in submission_df.columns if c != "id"]
submission_df[prob_cols] = np.clip(submission_df[prob_cols], eps, 1 - eps)
row_sums = submission_df[prob_cols].sum(axis=1)
submission_df[prob_cols] = submission_df[prob_cols].div(row_sums, axis=0)
submission_df[prob_cols] = submission_df[prob_cols].round(6)

submission_df.to_csv("./submission/submission.csv", index=False)
print(
    f"Submission saved to ./submission/submission.csv with shape {submission_df.shape}"
)

print(f"Final Validation Score: {best_val_score}")
