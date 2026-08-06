import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import pickle

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Get feature column names
margin_cols = [c for c in train_df.columns if c.startswith("margin")]
shape_cols = [c for c in train_df.columns if c.startswith("shape")]
texture_cols = [c for c in train_df.columns if c.startswith("texture")]
all_feature_cols = margin_cols + shape_cols + texture_cols
print(f"Total features: {len(all_feature_cols)}")

# Combine train and test for feature engineering
train_ids = train_df["id"].values
test_ids = test_df["id"].values
train_species = train_df["species"].values
X_full = pd.concat(
    [train_df[all_feature_cols], test_df[all_feature_cols]], axis=0, ignore_index=True
)


# Feature engineering
def add_group_statistics(df, group_cols, prefix):
    group_data = df[group_cols].values
    eps = 1e-10
    df[f"{prefix}_mean"] = group_data.mean(axis=1)
    df[f"{prefix}_std"] = group_data.std(axis=1)
    df[f"{prefix}_min"] = group_data.min(axis=1)
    df[f"{prefix}_max"] = group_data.max(axis=1)
    df[f"{prefix}_range"] = df[f"{prefix}_max"] - df[f"{prefix}_min"]
    df[f"{prefix}_skew"] = stats.skew(group_data + eps, axis=1)
    df[f"{prefix}_kurt"] = stats.kurtosis(group_data + eps, axis=1)
    df[f"{prefix}_median"] = np.median(group_data, axis=1)
    df[f"{prefix}_p25"] = np.percentile(group_data, 25, axis=1)
    df[f"{prefix}_p75"] = np.percentile(group_data, 75, axis=1)
    df[f"{prefix}_energy"] = np.sum(group_data**2, axis=1)
    df[f"{prefix}_abs_sum"] = np.sum(np.abs(group_data), axis=1)
    df[f"{prefix}_n_peaks"] = np.sum(
        (
            group_data
            > (
                df[f"{prefix}_mean"].values[:, None]
                + df[f"{prefix}_std"].values[:, None]
            )
        ),
        axis=1,
    )
    signs = np.sign(group_data - df[f"{prefix}_mean"].values[:, None] + eps)
    df[f"{prefix}_zero_cross"] = np.sum(signs[:, 1:] * signs[:, :-1] < 0, axis=1)
    return df


X_full = add_group_statistics(X_full, margin_cols, "margin")
X_full = add_group_statistics(X_full, shape_cols, "shape")
X_full = add_group_statistics(X_full, texture_cols, "texture")

# Cross-group correlations
X_full["margin_shape_corr"] = [
    (
        np.corrcoef(X_full.loc[i, margin_cols], X_full.loc[i, shape_cols])[0, 1]
        if np.std(X_full.loc[i, margin_cols]) > 0
        and np.std(X_full.loc[i, shape_cols]) > 0
        else 0
    )
    for i in range(len(X_full))
]
X_full["margin_texture_corr"] = [
    (
        np.corrcoef(X_full.loc[i, margin_cols], X_full.loc[i, texture_cols])[0, 1]
        if np.std(X_full.loc[i, margin_cols]) > 0
        and np.std(X_full.loc[i, texture_cols]) > 0
        else 0
    )
    for i in range(len(X_full))
]
X_full["shape_texture_corr"] = [
    (
        np.corrcoef(X_full.loc[i, shape_cols], X_full.loc[i, texture_cols])[0, 1]
        if np.std(X_full.loc[i, shape_cols]) > 0
        and np.std(X_full.loc[i, texture_cols]) > 0
        else 0
    )
    for i in range(len(X_full))
]

# Ratios and differences
X_full["margin_shape_ratio"] = X_full["margin_mean"] / (X_full["shape_mean"] + 1e-10)
X_full["margin_texture_ratio"] = X_full["margin_mean"] / (
    X_full["texture_mean"] + 1e-10
)
X_full["shape_texture_ratio"] = X_full["shape_mean"] / (X_full["texture_mean"] + 1e-10)
X_full["margin_shape_diff"] = X_full["margin_mean"] - X_full["shape_mean"]
X_full["margin_texture_diff"] = X_full["margin_mean"] - X_full["texture_mean"]
X_full["shape_texture_diff"] = X_full["shape_mean"] - X_full["texture_mean"]

# PCA per feature group
n_components_each = 20
for prefix, cols in [
    ("margin", margin_cols),
    ("shape", shape_cols),
    ("texture", texture_cols),
]:
    scaler_tmp = StandardScaler()
    X_group_scaled = scaler_tmp.fit_transform(X_full[cols].values)
    pca = PCA(n_components=n_components_each, random_state=42)
    X_pca = pca.fit_transform(X_group_scaled)
    for i in range(n_components_each):
        X_full[f"{prefix}_pca_{i}"] = X_pca[:, i]

# Additional derived features
for prefix, cols in [
    ("margin", margin_cols),
    ("shape", shape_cols),
    ("texture", texture_cols),
]:
    half = len(cols) // 2
    first_half = X_full[cols[:half]].values
    second_half = X_full[cols[half:]].values
    X_full[f"{prefix}_first_second_ratio"] = np.sum(first_half, axis=1) / (
        np.sum(second_half, axis=1) + 1e-10
    )
    X_full[f"{prefix}_argmax"] = np.argmax(X_full[cols].values, axis=1)
    X_full[f"{prefix}_argmin"] = np.argmin(X_full[cols].values, axis=1)

# Label encode species
le = LabelEncoder()
y_encoded = le.fit_transform(train_species)
class_names = le.classes_
num_classes = len(class_names)

# Split data
n_train = len(train_df)
X_train_full = X_full.iloc[:n_train].copy()
X_test_full = X_full.iloc[n_train:].copy()

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(X_train_full, y_encoded)))
assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap detected!"

X_train = X_train_full.iloc[train_idx].reset_index(drop=True)
X_val = X_train_full.iloc[val_idx].reset_index(drop=True)
y_train = y_encoded[train_idx]
y_val = y_encoded[val_idx]
train_ids_split = train_ids[train_idx]
val_ids_split = train_ids[val_idx]

# Scale engineered features (fit on train only)
engineered_cols = [
    c for c in X_train.columns if not c.startswith(("margin_", "shape_", "texture_"))
]
scale_cols = [
    c for c in engineered_cols if c not in ["margin_mean", "shape_mean", "texture_mean"]
]
scaler = StandardScaler()
scaler.fit(X_train[scale_cols])
X_train[scale_cols] = scaler.transform(X_train[scale_cols])
X_val[scale_cols] = scaler.transform(X_val[scale_cols])
X_test_full[scale_cols] = scaler.transform(X_test_full[scale_cols])

# Save processed data
X_train.to_pickle("./working/X_train.pkl")
X_val.to_pickle("./working/X_val.pkl")
X_test_full.to_pickle("./working/X_test_full.pkl")
np.save("./working/y_train.npy", y_train)
np.save("./working/y_val.npy", y_val)
np.save("./working/train_ids.npy", train_ids_split)
np.save("./working/val_ids.npy", val_ids_split)
np.save("./working/test_ids.npy", test_ids)
np.save("./working/class_names.npy", class_names)


# ============ MODEL DEFINITION ============
class LeafClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64], dropout=0.3):
        super(LeafClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
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

    def forward(self, x):
        return self.classifier(self.features(x))


# ============ TRAINING SETUP ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
input_dim = X_train.shape[1]
model = LeafClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
total_epochs = 100
scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

# Data loaders
X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
X_val_tensor = torch.tensor(X_val.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.astype(np.int64))
y_val_tensor = torch.tensor(y_val.astype(np.int64))
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)

# ============ TRAINING LOOP ============
best_val_loss = float("inf")
patience = 30
patience_counter = 0
best_model_state = None

for epoch in range(total_epochs):
    model.train()
    train_loss = 0.0
    train_total = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
        train_total += batch_y.size(0)
    avg_train_loss = train_loss / train_total

    model.eval()
    val_loss = 0.0
    val_total = 0
    all_val_probs = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            val_total += batch_y.size(0)
            probs = F.softmax(logits, dim=1)
            all_val_probs.append(probs.cpu().numpy())
    avg_val_loss = val_loss / val_total
    val_probs_array = np.concatenate(all_val_probs, axis=0)
    val_log_loss = log_loss(y_val, val_probs_array, labels=np.arange(num_classes))
    print(
        f"Epoch {epoch+1:3d}/{total_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_log_loss:.4f}"
    )
    scheduler.step()

    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

# ============ VALIDATION PREDICTIONS ============
all_val_probs = []
with torch.no_grad():
    for batch_X, _ in val_loader:
        batch_X = batch_X.to(device)
        logits = model(batch_X)
        probs = F.softmax(logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())
val_probs_array = np.concatenate(all_val_probs, axis=0)
final_val_log_loss = log_loss(y_val, val_probs_array, labels=np.arange(num_classes))

# ============ TEST INFERENCE ============
X_test_tensor = torch.tensor(X_test_full.values.astype(np.float32))
test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)

all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        batch_X = batch[0].to(device)
        logits = model(batch_X)
        probs = F.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())
test_probs_array = np.concatenate(all_test_probs, axis=0)

# ============ CREATE SUBMISSION ============
submission = pd.DataFrame(test_probs_array, columns=class_names)
submission.insert(0, "id", test_ids)
submission = submission[sample_sub.columns]
epsilon = 1e-15
submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(epsilon, 1 - epsilon)
row_sums = submission.iloc[:, 1:].sum(axis=1)
submission.iloc[:, 1:] = submission.iloc[:, 1:].div(row_sums, axis=0)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")

# ============ FINAL SCORE ============
score = final_val_log_loss
print(f"Final Validation Score: {score}")
