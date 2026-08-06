"""
Merged Leaf Classification Pipeline
- Data Processing & Feature Engineering
- Model Design (Multi-branch MLP + Neural Network)
- Training & Evaluation
- Submission Generation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import time
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. Data Processing & Feature Engineering
# ============================================================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Dynamic column identification (adopting memory: columns have no underscores)
feature_cols = [c for c in train_df.columns if c not in ["id", "species"]]
margin_cols = [c for c in feature_cols if c.startswith("margin")]
shape_cols = [c for c in feature_cols if c.startswith("shape")]
texture_cols = [c for c in feature_cols if c.startswith("texture")]


def engineer_features(df, margin_cols, shape_cols, texture_cols):
    """Create enhanced features from raw leaf attributes"""
    df_feat = df[["id"]].copy()
    feature_cols = margin_cols + shape_cols + texture_cols

    # Raw features
    for col in feature_cols:
        df_feat[col] = df[col].values

    # Statistical features per group
    for group_name, group_cols in [
        ("margin", margin_cols),
        ("shape", shape_cols),
        ("texture", texture_cols),
    ]:
        group_data = df[group_cols].values
        df_feat[f"{group_name}_mean"] = group_data.mean(axis=1)
        df_feat[f"{group_name}_std"] = group_data.std(axis=1)
        df_feat[f"{group_name}_min"] = group_data.min(axis=1)
        df_feat[f"{group_name}_max"] = group_data.max(axis=1)
        df_feat[f"{group_name}_range"] = group_data.max(axis=1) - group_data.min(axis=1)
        df_feat[f"{group_name}_median"] = np.median(group_data, axis=1)
        df_feat[f"{group_name}_q25"] = np.percentile(group_data, 25, axis=1)
        df_feat[f"{group_name}_q75"] = np.percentile(group_data, 75, axis=1)
        df_feat[f"{group_name}_iqr"] = (
            df_feat[f"{group_name}_q75"] - df_feat[f"{group_name}_q25"]
        )
        df_feat[f"{group_name}_energy"] = np.sum(group_data**2, axis=1)
        df_feat[f"{group_name}_entropy"] = -np.sum(
            group_data * np.log(group_data + 1e-10), axis=1
        )

    # Cross-feature interactions
    margin_data = df[margin_cols].values
    shape_data = df[shape_cols].values
    texture_data = df[texture_cols].values

    df_feat["margin_shape_corr"] = [
        np.corrcoef(m, s)[0, 1] if np.std(m) > 0 and np.std(s) > 0 else 0
        for m, s in zip(margin_data, shape_data)
    ]
    df_feat["margin_texture_corr"] = [
        np.corrcoef(m, t)[0, 1] if np.std(m) > 0 and np.std(t) > 0 else 0
        for m, t in zip(margin_data, texture_data)
    ]
    df_feat["shape_texture_corr"] = [
        np.corrcoef(s, t)[0, 1] if np.std(s) > 0 and np.std(t) > 0 else 0
        for s, t in zip(shape_data, texture_data)
    ]

    df_feat["margin_shape_ratio"] = margin_data.mean(axis=1) / (
        shape_data.mean(axis=1) + 1e-10
    )
    df_feat["margin_texture_ratio"] = margin_data.mean(axis=1) / (
        texture_data.mean(axis=1) + 1e-10
    )
    df_feat["shape_texture_ratio"] = shape_data.mean(axis=1) / (
        texture_data.mean(axis=1) + 1e-10
    )

    return df_feat


# Apply feature engineering
train_feat = engineer_features(train_df, margin_cols, shape_cols, texture_cols)
test_feat = engineer_features(test_df, margin_cols, shape_cols, texture_cols)
print(f"Engineered train features: {train_feat.shape}")

# Encode labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df["species"].values)
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# Stratified split (using indices directly to avoid INDEX_BUG)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(train_feat, train_labels)))
assert len(set(train_idx) & set(val_idx)) == 0, "Data leakage detected!"

train_features = train_feat.iloc[train_idx].reset_index(drop=True)
val_features = train_feat.iloc[val_idx].reset_index(drop=True)
train_labels_split = train_labels[train_idx]
val_labels_split = train_labels[val_idx]

# Scale features (fit ONLY on training)
feat_cols_final = [c for c in train_features.columns if c != "id"]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_features[feat_cols_final])
X_val_scaled = scaler.transform(val_features[feat_cols_final])
X_test_scaled = scaler.transform(test_feat[feat_cols_final])

X_train_arr = X_train_scaled.astype(np.float32)
X_val_arr = X_val_scaled.astype(np.float32)
X_test_arr = X_test_scaled.astype(np.float32)
y_train_arr = train_labels_split.astype(np.int64)
y_val_arr = val_labels_split.astype(np.int64)
test_ids = test_feat["id"].values

feature_dim = X_train_arr.shape[1]
print(f"Train: {X_train_arr.shape}, Val: {X_val_arr.shape}, Test: {X_test_arr.shape}")
print(f"Feature dim: {feature_dim}")


# ============================================================
# 2. Model Design
# ============================================================
class LeafClassifier(nn.Module):
    """MLP classifier with strong regularization for leaf features"""

    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout_rate=0.3):
        super(LeafClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(hidden_dim // 4, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)


# ============================================================
# 3. Training & Evaluation
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Data loaders
train_dataset = TensorDataset(torch.tensor(X_train_arr), torch.tensor(y_train_arr))
val_dataset = TensorDataset(torch.tensor(X_val_arr), torch.tensor(y_val_arr))
test_dataset = TensorDataset(torch.tensor(X_test_arr))

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)


def multiclass_log_loss(y_true, y_pred_proba, eps=1e-15):
    """Exact competition metric"""
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    N = len(y_true)
    y_onehot = np.zeros((N, y_pred_proba.shape[1]))
    y_onehot[np.arange(N), y_true] = 1
    return -np.sum(y_onehot * np.log(y_pred_proba)) / N


# Model setup
model = LeafClassifier(input_dim=feature_dim, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = 300
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-5
)

best_val_loss = float("inf")
best_epoch = -1
patience_counter = 0
early_stopping_patience = 40
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)
model_path = "./working/best_model.pth"

print("\nTraining started...")
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss_sum = 0.0
    train_total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss_sum += loss.item() * batch_X.size(0)
        train_total += batch_y.size(0)

    train_loss = train_loss_sum / train_total
    scheduler.step()

    # Validation
    model.eval()
    val_loss_sum = 0.0
    val_total = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            probs = F.softmax(logits, dim=1)
            val_loss_sum += loss.item() * batch_X.size(0)
            val_total += batch_y.size(0)
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(batch_y.cpu().numpy())

    val_loss = val_loss_sum / val_total
    val_probs_concat = np.vstack(all_val_probs)
    val_labels_concat = np.concatenate(all_val_labels)
    val_logloss = multiclass_log_loss(val_labels_concat, val_probs_concat)

    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_logloss:.4f}"
    )

    # Early stopping
    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), model_path)
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(
                f"Best model from epoch {best_epoch} (val logloss: {best_val_loss:.4f})"
            )
            break

# Load best model
print(f"\nLoading best model from epoch {best_epoch}...")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Final validation evaluation
all_val_probs = []
with torch.no_grad():
    for batch_X, _ in val_loader:
        batch_X = batch_X.to(device)
        logits = model(batch_X)
        probs = F.softmax(logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())

val_probs_final = np.vstack(all_val_probs)
score = multiclass_log_loss(val_labels_concat, val_probs_final)
print(f"Final validation multi-class log loss: {score:.6f}")

# Test inference
print("\nGenerating test predictions...")
all_test_probs = []
with torch.no_grad():
    for (batch_X,) in test_loader:
        batch_X = batch_X.to(device)
        logits = model(batch_X)
        probs = F.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.vstack(all_test_probs)
print(f"Test predictions shape: {test_probs.shape}")

# ============================================================
# 4. Generate Submission (adopting memory: use sample_sub for exact column order)
# ============================================================
submission = sample_sub.copy()
for i, test_id in enumerate(test_ids):
    row_idx = submission["id"] == test_id
    if row_idx.any():
        for j, class_name in enumerate(label_encoder.classes_):
            submission.loc[row_idx, class_name] = test_probs[i, j]

# Verify
assert list(submission.columns) == list(sample_sub.columns), "Column order mismatch!"
assert not submission.isnull().any().any(), "NaN values found!"
assert len(submission) == len(sample_sub), "Row count mismatch!"

submission_path = "./submission/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"✓ Submission saved to {submission_path}")
print(f"✓ Submission shape: {submission.shape}")

# Final required output
print(f"Final Validation Score: {score}")
