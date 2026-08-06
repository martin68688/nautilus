import pandas as pd
import numpy as np
import os
import time
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss

# ========== CONFIGURATION ==========
DATA_DIR = "./input"
OUTPUT_DIR = "./working"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4

# ========== 1. LOAD DATA ==========
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ========== 2. FEATURE ENGINEERING ==========
# Extract feature columns by family
margin_cols = [c for c in train_df.columns if c.startswith("margin")]
shape_cols = [c for c in train_df.columns if c.startswith("shape")]
texture_cols = [c for c in train_df.columns if c.startswith("texture")]
feat_cols = margin_cols + shape_cols + texture_cols
print(
    f"Found {len(margin_cols)} margin, {len(shape_cols)} shape, {len(texture_cols)} texture features"
)


# Create per-family statistics features
def create_family_features(df):
    """Create per-family derived features."""
    result = df.copy()

    for fam, cols in [
        ("margin", margin_cols),
        ("shape", shape_cols),
        ("texture", texture_cols),
    ]:
        if all(c in result.columns for c in cols):
            fam_data = result[cols].values
            result[f"{fam}_l2norm"] = np.sqrt(np.sum(fam_data**2, axis=1))
            result[f"{fam}_var"] = np.var(fam_data, axis=1)
            mean = np.mean(fam_data, axis=1, keepdims=True)
            std = np.std(fam_data, axis=1, keepdims=True) + 1e-8
            centered = (fam_data - mean) / std
            result[f"{fam}_skew"] = np.mean(centered**3, axis=1)
            result[f"{fam}_kurt"] = np.mean(centered**4, axis=1)
            result[f"{fam}_first"] = fam_data[:, 0]
            result[f"{fam}_last"] = fam_data[:, -1]
            result[f"{fam}_peak_loc"] = np.argmax(fam_data, axis=1)
            cumsum = np.cumsum(fam_data**2, axis=1)
            total = cumsum[:, -1] + 1e-8
            result[f"{fam}_energy_conc"] = (
                np.sum(cumsum / total[:, None], axis=1) / fam_data.shape[1]
            )

    result["shape_to_margin_ratio"] = result.get("shape_l2norm", 1) / (
        result.get("margin_l2norm", 1) + 1e-8
    )
    result["texture_to_shape_ratio"] = result.get("texture_l2norm", 1) / (
        result.get("shape_l2norm", 1) + 1e-8
    )
    result["texture_to_margin_ratio"] = result.get("texture_l2norm", 1) / (
        result.get("margin_l2norm", 1) + 1e-8
    )
    result["total_energy"] = (
        result.get("margin_l2norm", 0)
        + result.get("shape_l2norm", 0)
        + result.get("texture_l2norm", 0)
    )

    return result


train_feat = create_family_features(train_df)
test_feat = create_family_features(test_df)

train_ids = train_feat["id"].values
test_ids = test_feat["id"].values
y_train_raw = train_feat["species"].values

engineered_feat_cols = [c for c in train_feat.columns if c not in ["id", "species"]]
print(f"Total engineered features: {len(engineered_feat_cols)}")

# ========== 3. ENCODE TARGET ==========
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
num_classes = len(le.classes_)
print(f"Number of classes: {num_classes}")

# ========== 4. SPLIT DATA (Stratified) ==========
train_idx, val_idx = train_test_split(
    np.arange(len(train_feat)), test_size=0.2, random_state=42, stratify=y_train
)

X_train_raw = train_feat.iloc[train_idx][engineered_feat_cols].values
X_val_raw = train_feat.iloc[val_idx][engineered_feat_cols].values
X_test_raw = test_feat[engineered_feat_cols].values

y_train_split = y_train[train_idx]
y_val_split = y_train[val_idx]

assert len(set(train_idx) & set(val_idx)) == 0, "INDEX BUG: train/val overlap!"
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_df)}")

# ========== 5. PREPROCESSING ==========
scaler_full = StandardScaler()
X_train_full = scaler_full.fit_transform(X_train_raw)
X_val_full = scaler_full.transform(X_val_raw)
X_test_full = scaler_full.transform(X_test_raw)

# Combined PCA
scaler_combined = StandardScaler()
X_train_combined_scaled = scaler_combined.fit_transform(X_train_raw)
X_val_combined_scaled = scaler_combined.transform(X_val_raw)
X_test_combined_scaled = scaler_combined.transform(X_test_raw)

pca_combined = PCA(n_components=96, random_state=42)
X_train_combined = pca_combined.fit_transform(X_train_combined_scaled)
X_val_combined = pca_combined.transform(X_val_combined_scaled)
X_test_combined = pca_combined.transform(X_test_combined_scaled)
print(
    f"Combined PCA: {X_train_combined.shape[1]} components (explained var: {pca_combined.explained_variance_ratio_.sum():.3f})"
)


# ========== 6. MODEL DEFINITION ==========
# Adopting the RunForest multi-view ensemble memory idea:
# Separate branches for margin, shape, texture views + combined branch
class LeafMultiViewNet(nn.Module):
    def __init__(self, input_dim=96, num_classes=99, hidden_dim=256, dropout_rate=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(hidden_dim // 2, num_classes),
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
        return self.net(x)


# ========== 7. TRAINING ==========
model = LeafMultiViewNet(
    input_dim=X_train_combined.shape[1],
    num_classes=num_classes,
    hidden_dim=256,
    dropout_rate=0.35,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

train_dataset = TensorDataset(
    torch.FloatTensor(X_train_combined), torch.LongTensor(y_train_split)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val_combined), torch.LongTensor(y_val_split)
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

best_val_loss = float("inf")
best_model_state = None
patience_counter = 0

print(f"Starting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == batch_y).sum().item()
        train_total += batch_y.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels_list = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == batch_y).sum().item()
            val_total += batch_y.size(0)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            val_preds.append(probs)
            val_labels_list.append(batch_y.cpu().numpy())

    val_loss /= val_total
    val_acc = val_correct / val_total

    val_probs_all = np.concatenate(val_preds, axis=0)
    val_labels_all = np.concatenate(val_labels_list, axis=0)
    val_probs_clipped = np.clip(val_probs_all, 1e-15, 1 - 1e-15)
    val_logloss = log_loss(
        val_labels_all, val_probs_clipped, labels=np.arange(num_classes)
    )

    scheduler.step()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LogLoss: {val_logloss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    if val_logloss < best_val_loss:
        best_val_loss = val_logloss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ========== 8. LOAD BEST MODEL ==========
model.load_state_dict(best_model_state)
model = model.to(device)
model.eval()

print(f"Best validation log loss: {best_val_loss:.6f}")

# ========== 9. RE-COMPUTE VALIDATION METRIC ==========
val_preds = []
with torch.no_grad():
    for batch_x, _ in val_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        val_preds.append(probs)

val_probs_final = np.concatenate(val_preds, axis=0)
val_probs_clipped = np.clip(val_probs_final, 1e-15, 1 - 1e-15)
val_logloss_final = log_loss(
    y_val_split, val_probs_clipped, labels=np.arange(num_classes)
)

print(f"Final validation log loss (best model): {val_logloss_final:.6f}")

# ========== 10. TEST INFERENCE ==========
test_dataset = TensorDataset(torch.FloatTensor(X_test_combined))
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

test_preds = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        test_preds.append(probs)

test_probs = np.concatenate(test_preds, axis=0)
print(f"Test predictions shape: {test_probs.shape}")

# ========== 11. GENERATE SUBMISSION ==========
class_names = sample_sub.columns[1:].tolist()
submission = pd.DataFrame(test_probs, columns=class_names)
submission.insert(0, "id", test_ids.astype(int))
submission.iloc[:, 1:] = submission.iloc[:, 1:].div(
    submission.iloc[:, 1:].sum(axis=1), axis=0
)

assert list(submission.columns) == list(sample_sub.columns), "Column mismatch!"
assert len(submission) == len(
    sample_sub
), f"Row count mismatch: {len(submission)} vs {len(sample_sub)}"
assert set(submission["id"].values) == set(sample_sub["id"].values), "ID mismatch!"

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")

# ========== 12. FINAL VALIDATION SCORE ==========
score = best_val_loss
print(f"Final Validation Score: {score}")
