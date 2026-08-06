import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================
# 1. LOAD DATA
# ============================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Identify feature columns (no underscores in names)
margin_cols = [c for c in train_df.columns if c.startswith("margin") and c != "margin"]
shape_cols = [c for c in train_df.columns if c.startswith("shape") and c != "shape"]
texture_cols = [
    c for c in train_df.columns if c.startswith("texture") and c != "texture"
]

print(
    f"Margin cols: {len(margin_cols)}, Shape cols: {len(shape_cols)}, Texture cols: {len(texture_cols)}"
)


# ============================================
# 2. FEATURE ENGINEERING - Statistical Features
# ============================================
def extract_statistical_features(df, cols, prefix):
    """Extract statistical features for a group of columns"""
    X = df[cols].values
    features = {}
    features[f"{prefix}_mean"] = X.mean(axis=1)
    features[f"{prefix}_std"] = X.std(axis=1)
    features[f"{prefix}_min"] = X.min(axis=1)
    features[f"{prefix}_max"] = X.max(axis=1)
    features[f"{prefix}_range"] = X.max(axis=1) - X.min(axis=1)
    features[f"{prefix}_median"] = np.median(X, axis=1)
    features[f"{prefix}_q25"] = np.percentile(X, 25, axis=1)
    features[f"{prefix}_q75"] = np.percentile(X, 75, axis=1)
    features[f"{prefix}_iqr"] = np.percentile(X, 75, axis=1) - np.percentile(
        X, 25, axis=1
    )
    features[f"{prefix}_skew"] = stats.skew(X, axis=1)
    features[f"{prefix}_kurtosis"] = stats.kurtosis(X, axis=1)
    features[f"{prefix}_energy"] = np.sum(X**2, axis=1)
    features[f"{prefix}_l1norm"] = np.sum(np.abs(X), axis=1)
    features[f"{prefix}_l2norm"] = np.sqrt(np.sum(X**2, axis=1))
    centered = X - X.mean(axis=1, keepdims=True)
    signs = np.sign(centered)
    zero_crossings = np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1) / (
        X.shape[1] - 1
    )
    features[f"{prefix}_zero_crossings"] = zero_crossings
    features[f"{prefix}_num_peaks"] = np.sum(
        (X[:, 1:-1] > X[:, :-2]) & (X[:, 1:-1] > X[:, 2:]), axis=1
    ) / (X.shape[1] - 2)
    return pd.DataFrame(features)


print("Extracting statistical features...")
train_stats = pd.DataFrame({"id": train_df["id"]})
test_stats = pd.DataFrame({"id": test_df["id"]})

for prefix, cols in [
    ("margin", margin_cols),
    ("shape", shape_cols),
    ("texture", texture_cols),
]:
    train_stats = pd.concat(
        [train_stats, extract_statistical_features(train_df, cols, prefix)], axis=1
    )
    test_stats = pd.concat(
        [test_stats, extract_statistical_features(test_df, cols, prefix)], axis=1
    )

print(f"Statistical features: {train_stats.shape[1] - 1} features per sample")

# ============================================
# 3. FEATURE ENGINEERING - Cross-group Interactions
# ============================================
print("Creating cross-group features...")


def add_cross_features(train_df, test_df, margin_cols, shape_cols, texture_cols):
    cross_features_train = {}
    cross_features_test = {}
    groups = [("margin", margin_cols), ("shape", shape_cols), ("texture", texture_cols)]
    for i, (prefix1, cols1) in enumerate(groups):
        for j, (prefix2, cols2) in enumerate(groups):
            if i >= j:
                continue
            X1_train = train_df[cols1].values
            X2_train = train_df[cols2].values
            X1_test = test_df[cols1].values
            X2_test = test_df[cols2].values
            cross_features_train[f"{prefix1}_{prefix2}_mean_dist"] = np.sqrt(
                np.sum((X1_train - X2_train) ** 2, axis=1)
            )
            cross_features_test[f"{prefix1}_{prefix2}_mean_dist"] = np.sqrt(
                np.sum((X1_test - X2_test) ** 2, axis=1)
            )
            norm1_train = np.sqrt(np.sum(X1_train**2, axis=1)) + 1e-10
            norm2_train = np.sqrt(np.sum(X2_train**2, axis=1)) + 1e-10
            norm1_test = np.sqrt(np.sum(X1_test**2, axis=1)) + 1e-10
            norm2_test = np.sqrt(np.sum(X2_test**2, axis=1)) + 1e-10
            cross_features_train[f"{prefix1}_{prefix2}_cosine_sim"] = np.sum(
                X1_train * X2_train, axis=1
            ) / (norm1_train * norm2_train)
            cross_features_test[f"{prefix1}_{prefix2}_cosine_sim"] = np.sum(
                X1_test * X2_test, axis=1
            ) / (norm1_test * norm2_test)
    return pd.DataFrame(cross_features_train), pd.DataFrame(cross_features_test)


train_cross, test_cross = add_cross_features(
    train_df, test_df, margin_cols, shape_cols, texture_cols
)
train_cross.insert(0, "id", train_df["id"].values)
test_cross.insert(0, "id", test_df["id"].values)

print(f"Cross-group features: {train_cross.shape[1] - 1} features")

# ============================================
# 4. PCA ON ORIGINAL FEATURES
# ============================================
print("Applying PCA to feature groups...")
train_pca_dict = {"id": train_df["id"].values}
test_pca_dict = {"id": test_df["id"].values}

for prefix, cols in [
    ("margin", margin_cols),
    ("shape", shape_cols),
    ("texture", texture_cols),
]:
    n_components = min(24, len(cols))
    pca = PCA(n_components=n_components, random_state=42)
    pca_train = pca.fit_transform(train_df[cols].values)
    pca_test = pca.transform(test_df[cols].values)
    for i in range(n_components):
        train_pca_dict[f"{prefix}_pca_{i+1}"] = pca_train[:, i]
        test_pca_dict[f"{prefix}_pca_{i+1}"] = pca_test[:, i]

train_pca = pd.DataFrame(train_pca_dict)
test_pca = pd.DataFrame(test_pca_dict)

print(f"PCA features: {train_pca.shape[1] - 1} features")

# ============================================
# 5. COMBINE ALL FEATURES
# ============================================
print("Combining all features...")

train_combined = train_df[["id", "species"]].copy()
test_combined = test_df[["id"]].copy()

train_combined = train_combined.merge(train_stats, on="id", how="left")
test_combined = test_combined.merge(test_stats, on="id", how="left")
train_combined = train_combined.merge(train_cross, on="id", how="left")
test_combined = test_combined.merge(test_cross, on="id", how="left")
train_combined = train_combined.merge(train_pca, on="id", how="left")
test_combined = test_combined.merge(test_pca, on="id", how="left")

orig_feat_cols = margin_cols + shape_cols + texture_cols
train_combined = pd.concat(
    [train_combined, train_df[orig_feat_cols].reset_index(drop=True)], axis=1
)
test_combined = pd.concat(
    [test_combined, test_df[orig_feat_cols].reset_index(drop=True)], axis=1
)

train_combined = train_combined.fillna(0)
test_combined = test_combined.fillna(0)

feature_cols = [c for c in train_combined.columns if c not in ["id", "species"]]
print(f"Total features: {len(feature_cols)}")

# ============================================
# 6. ENCODE LABELS
# ============================================
label_encoder = LabelEncoder()
train_combined["species_encoded"] = label_encoder.fit_transform(
    train_combined["species"]
)

species_order = sample_sub.columns[1:].tolist()
print(f"Number of species in submission: {len(species_order)}")

# ============================================
# 7. SPLIT DATA (Stratified) - FIX INDEX BUG
# ============================================
train_idx, val_idx = train_test_split(
    np.arange(len(train_combined)),
    test_size=0.15,
    random_state=42,
    stratify=train_combined["species_encoded"],
)

assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap detected!"

# Use sub-DataFrames directly to avoid index misalignment
train_set = train_combined.iloc[train_idx].reset_index(drop=True)
val_set = train_combined.iloc[val_idx].reset_index(drop=True)

X_train = train_set[feature_cols].values.astype(np.float32)
y_train = train_set["species_encoded"].values
X_val = val_set[feature_cols].values.astype(np.float32)
y_val = val_set["species_encoded"].values
X_test = test_combined[feature_cols].values.astype(np.float32)

print(
    f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}"
)

# ============================================
# 8. STANDARDIZE FEATURES
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)


# ============================================
# 9. MODEL ARCHITECTURE
# ============================================
class MultiModalFusionModel(nn.Module):
    def __init__(
        self,
        num_classes=99,
        hidden_dim_tab=128,
        embed_dim_tab=64,
        fusion_dim=256,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.encoder_margin = nn.Sequential(
            nn.Linear(64, hidden_dim_tab),
            nn.BatchNorm1d(hidden_dim_tab),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_tab, embed_dim_tab),
        )
        self.encoder_shape = nn.Sequential(
            nn.Linear(64, hidden_dim_tab),
            nn.BatchNorm1d(hidden_dim_tab),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_tab, embed_dim_tab),
        )
        self.encoder_texture = nn.Sequential(
            nn.Linear(64, hidden_dim_tab),
            nn.BatchNorm1d(hidden_dim_tab),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_tab, embed_dim_tab),
        )

        fusion_input_dim = embed_dim_tab * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.8),
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, margin, shape, texture):
        emb_margin = self.encoder_margin(margin)
        emb_shape = self.encoder_shape(shape)
        emb_texture = self.encoder_texture(texture)
        fused_features = torch.cat([emb_margin, emb_shape, emb_texture], dim=1)
        fused = self.fusion(fused_features)
        logits = self.classifier(fused)
        return logits


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.15):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        n_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        loss = torch.sum(-true_dist * log_probs, dim=1)
        return loss.mean()


# ============================================
# 10. TRAINING SETUP
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MultiModalFusionModel(num_classes=99, dropout_rate=0.35).to(device)
criterion = LabelSmoothingCrossEntropy(smoothing=0.15)
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)


def split_features(X_batch):
    return X_batch[:, :64], X_batch[:, 64:128], X_batch[:, 128:192]


X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

batch_size = 32
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

# ============================================
# 11. TRAINING LOOP
# ============================================
num_epochs = 100
patience = 20
best_val_score = float("inf")
best_epoch = 0
no_improve = 0
best_model_state = None

scaler_amp = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
use_amp = torch.cuda.is_available()

print(
    f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}"
)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        margin, shape, texture = split_features(batch_X)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(margin, shape, texture)
                loss = criterion(logits, batch_y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            logits = model(margin, shape, texture)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        train_loss += loss.item() * batch_X.size(0)
        preds = logits.argmax(dim=1)
        train_correct += (preds == batch_y).sum().item()
        train_total += batch_y.size(0)

    scheduler.step()

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            margin, shape, texture = split_features(batch_X)
            logits = model(margin, shape, texture)
            probs = torch.softmax(logits, dim=1)
            probs = torch.clamp(probs, 1e-15, 1 - 1e-15)
            log_probs = torch.log(probs)
            n_classes = logits.size(1)
            one_hot = torch.zeros(batch_y.size(0), n_classes, device=batch_y.device)
            one_hot.scatter_(1, batch_y.unsqueeze(1), 1)
            batch_loss = -(one_hot * log_probs).sum(dim=1).mean()
            val_loss += batch_loss.item() * batch_X.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_y.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_val_loss = val_loss / len(val_loader.dataset)
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total

    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
    )

    if avg_val_loss < best_val_score:
        best_val_score = avg_val_loss
        best_epoch = epoch + 1
        no_improve = 0
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}, best epoch: {best_epoch}")
            break

print(
    f"Training complete. Best validation loss: {best_val_score:.4f} at epoch {best_epoch}"
)

# ============================================
# 12. FINAL VALIDATION SCORE
# ============================================
model.load_state_dict(best_model_state)
model.eval()

val_probs_list = []
with torch.no_grad():
    for batch_X, _ in val_loader:
        batch_X = batch_X.to(device)
        margin, shape, texture = split_features(batch_X)
        logits = model(margin, shape, texture)
        probs = torch.softmax(logits, dim=1)
        val_probs_list.append(probs.cpu().numpy())

val_probs = np.concatenate(val_probs_list, axis=0)
val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

n_classes = val_probs.shape[1]
y_val_onehot = np.zeros((len(y_val), n_classes))
for i, label in enumerate(y_val):
    y_val_onehot[i, label] = 1

log_loss_val = -(y_val_onehot * np.log(val_probs_clipped)).sum(axis=1).mean()
score = float(log_loss_val)

# ============================================
# 13. TEST INFERENCE AND SUBMISSION
# ============================================
print("\nGenerating test predictions...")
test_probs_list = []
with torch.no_grad():
    for (batch_X,) in test_loader:
        batch_X = batch_X.to(device)
        margin, shape, texture = split_features(batch_X)
        logits = model(margin, shape, texture)
        probs = torch.softmax(logits, dim=1)
        test_probs_list.append(probs.cpu().numpy())

test_probs = np.concatenate(test_probs_list, axis=0)
test_probs_clipped = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs_clipped = test_probs_clipped / test_probs_clipped.sum(axis=1, keepdims=True)

# Reorder predictions to match submission column order
species_map = {}
for i, cls in enumerate(label_encoder.classes_):
    if cls in species_order:
        species_map[i] = species_order.index(cls)

test_preds_reordered = np.zeros((len(test_probs_clipped), len(species_order)))
for orig_idx, sub_idx in species_map.items():
    test_preds_reordered[:, sub_idx] = test_probs_clipped[:, orig_idx]

test_preds_reordered = test_preds_reordered / test_preds_reordered.sum(
    axis=1, keepdims=True
)
test_preds_reordered = np.clip(test_preds_reordered, 1e-15, 1 - 1e-15)
test_preds_reordered = test_preds_reordered / test_preds_reordered.sum(
    axis=1, keepdims=True
)

submission_df = pd.DataFrame(test_preds_reordered, columns=species_order)
submission_df.insert(0, "id", test_df["id"].values)
submission_df["id"] = submission_df["id"].astype(int)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(
    f"Columns match sample: {list(submission_df.columns) == list(sample_sub.columns)}"
)

print(f"Final Validation Score: {score}")
