import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.EState import EState
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ===== Step 1: Data Processing and Feature Engineering =====


def compute_rdkit_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
    except:
        return None

    features = []

    # 1. Morgan fingerprint (2048 bits)
    try:
        fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features.extend(list(fp))
    except:
        features.extend([0.0] * 2048)

    # 2. Physicochemical properties
    desc_funcs = [
        Descriptors.MolWt,
        Descriptors.NumRotatableBonds,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        Descriptors.TPSA,
        Descriptors.MolLogP,
        Descriptors.NumAromaticRings,
        Descriptors.NumSaturatedRings,
        Descriptors.NumAliphaticRings,
        Descriptors.FractionCSP3,
        Descriptors.BalabanJ,
        Descriptors.BertzCT,
        Descriptors.Chi0v,
        Descriptors.Chi1v,
        Descriptors.Chi2v,
        Descriptors.Chi3v,
        Descriptors.Chi4v,
        Descriptors.HeavyAtomCount,
        Descriptors.NHOHCount,
        Descriptors.NOCount,
        Descriptors.NumHeteroatoms,
        Descriptors.NumValenceElectrons,
        Descriptors.RingCount,
    ]
    for func in desc_funcs:
        try:
            val = func(mol)
            if np.isscalar(val):
                features.append(float(val))
            else:
                features.append(0.0)
        except:
            features.append(0.0)

    # 3. EState indices
    try:
        state_inds = EState.EStateIndices(mol)
        if len(state_inds) < 40:
            state_inds = list(state_inds) + [0.0] * (40 - len(state_inds))
        else:
            state_inds = state_inds[:40]
        features.extend(state_inds)
    except:
        features.extend([0.0] * 40)

    # 4. Morgan feature counts (3 radii)
    for radius in [1, 2, 3]:
        try:
            fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=1024)
            features.extend(list(fp))
        except:
            features.extend([0.0] * 1024)

    # 5. MACCS keys
    try:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        features.extend(list(maccs))
    except:
        features.extend([0.0] * 166)

    # 6. Atom pair fingerprints
    try:
        ap = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)
        features.extend(list(ap))
    except:
        features.extend([0.0] * 1024)

    # 7. Topological torsions
    try:
        tt = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol, nBits=1024
        )
        features.extend(list(tt))
    except:
        features.extend([0.0] * 1024)

    return features


def preprocess_data():
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
    print(f"Loaded training data: {train_df.shape}")
    print(f"Loaded test data: {test_df.shape}")

    target_cols = [
        "LogD",
        "KSOL",
        "HLM CLint",
        "MLM CLint",
        "Caco-2 Permeability Papp A>B",
        "Caco-2 Permeability Efflux",
        "MPPB",
        "MBPB",
        "MGMB",
    ]
    log_transform_cols = [col for col in target_cols if col != "LogD"]

    # Log transform non-LogD targets
    train_targets = train_df[target_cols].copy()
    for col in log_transform_cols:
        mask = ~train_targets[col].isna()
        if mask.sum() > 0:
            train_targets.loc[mask, col] = np.log10(
                train_targets.loc[mask, col].clip(lower=1e-10)
            )

    # Compute features for training
    train_features_list = []
    valid_indices = []
    print("Computing RDKit features for training data...")
    for idx, row in train_df.iterrows():
        features = compute_rdkit_features(row["SMILES"])
        if features is not None:
            train_features_list.append(features)
            valid_indices.append(idx)
    train_features = np.array(train_features_list, dtype=np.float32)
    print(f"Training features shape: {train_features.shape}")
    train_targets_filtered = train_targets.iloc[valid_indices]
    train_names = train_df["Molecule Name"].iloc[valid_indices]

    # Compute features for test
    test_features_list = []
    test_valid_indices = []
    print("Computing RDKit features for test data...")
    for idx, row in test_df.iterrows():
        features = compute_rdkit_features(row["SMILES"])
        if features is not None:
            test_features_list.append(features)
            test_valid_indices.append(idx)
    test_features = np.array(test_features_list, dtype=np.float32)
    test_names = test_df["Molecule Name"].iloc[test_valid_indices]
    print(f"Test features shape: {test_features.shape}")

    # Train/validation split FIRST
    train_idx, val_idx = train_test_split(
        np.arange(len(train_features)), test_size=0.15, random_state=42
    )

    X_train_raw = train_features[train_idx]
    X_val_raw = train_features[val_idx]
    X_test_raw = test_features

    # Scale features — fit ONLY on training split
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Remove near-zero variance features — fit ONLY on training split
    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)
    print(f"Features after variance threshold: {X_train.shape[1]}")
    y_train = train_targets_filtered.iloc[train_idx].values
    y_val = train_targets_filtered.iloc[val_idx].values
    train_names_split = train_names.iloc[train_idx]
    val_names_split = train_names.iloc[val_idx]
    train_mask = ~np.isnan(y_train)
    val_mask = ~np.isnan(y_val)

    processed_data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "train_names": train_names_split.values,
        "val_names": val_names_split.values,
        "test_names": test_names.values,
        "target_cols": target_cols,
        "log_transform_cols": log_transform_cols,
    }

    print(f"\nFinal dataset sizes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    return processed_data


processed_data = preprocess_data()
X_train = processed_data["X_train"]
X_val = processed_data["X_val"]
X_test = processed_data["X_test"]
y_train = processed_data["y_train"]
y_val = processed_data["y_val"]
train_mask = processed_data["train_mask"]
val_mask = processed_data["val_mask"]
test_names = processed_data["test_names"]
target_cols = processed_data["target_cols"]
log_transform_cols = processed_data["log_transform_cols"]

# ===== Step 2: Model Design =====


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_targets=9, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.residual2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.target_heads = nn.ModuleList()
        for _ in range(num_targets):
            self.target_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_dim // 2, 1),
                )
            )

    def forward(self, x):
        h = self.encoder(x)
        h_res = self.residual1(h)
        h_res = self.bn1(h_res)
        h_res = F.relu(h_res)
        h_res = self.dropout1(h_res)
        h = h + h_res
        h_res = self.residual2(h)
        h_res = self.bn2(h_res)
        h_res = F.relu(h_res)
        h_res = self.dropout2(h_res)
        h = h + h_res
        outputs = [head(h) for head in self.target_heads]
        return torch.cat(outputs, dim=1)


def compute_masked_mse_loss(predictions, targets, mask):
    loss = 0.0
    valid_count = 0
    for i in range(predictions.shape[1]):
        valid_mask = mask[:, i]
        if valid_mask.sum() > 0:
            loss += F.mse_loss(predictions[valid_mask, i], targets[valid_mask, i])
            valid_count += 1
    return loss / max(valid_count, 1)


def compute_ma_rae(predictions, targets, mask):
    n_targets = predictions.shape[1]
    rae_per_target = []
    for i in range(n_targets):
        valid_idx = mask[:, i]
        if valid_idx.sum() > 0:
            y_pred = predictions[valid_idx, i]
            y_true = targets[valid_idx, i]
            y_pred = np.nan_to_num(y_pred, nan=0.0)
            y_true = np.nan_to_num(y_true, nan=0.0)
            abs_error = np.abs(y_pred - y_true)
            mean_true = np.mean(y_true)
            baseline = np.abs(y_true - mean_true)
            baseline = np.clip(baseline, a_min=1e-10, a_max=None)
            rae = np.mean(abs_error / baseline)
            rae_per_target.append(rae)
    return np.mean(rae_per_target) if rae_per_target else 0.0


# ===== Step 3: Training and Evaluation =====

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train)
y_val_tensor = torch.FloatTensor(y_val)
train_mask_tensor = torch.BoolTensor(train_mask)
val_mask_tensor = torch.BoolTensor(val_mask)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_mask_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor, val_mask_tensor)
test_dataset = TensorDataset(X_test_tensor)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
)

input_dim = X_train.shape[1]
model = MultiTaskMLP(input_dim, hidden_dim=512, num_targets=9, dropout=0.3)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

num_epochs = 200
best_val_ma_rae = float("inf")
best_model_state = None
patience = 30
patience_counter = 0

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_batches = 0
    for features, targets, mask in train_loader:
        features = features.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        predictions = model(features)
        loss = compute_masked_mse_loss(predictions, targets, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1

    scheduler.step()

    model.eval()
    val_predictions = []
    val_targets = []
    val_valid_mask = []
    with torch.no_grad():
        for features, targets, mask in val_loader:
            features = features.to(device)
            predictions = model(features)
            val_predictions.append(predictions.cpu().numpy())
            val_targets.append(targets.numpy())
            val_valid_mask.append(mask.numpy())

    val_predictions = np.concatenate(val_predictions, axis=0)
    val_targets_np = np.concatenate(val_targets, axis=0)
    val_mask_np = np.concatenate(val_valid_mask, axis=0)

    # Inverse log transform for MA-RAE in original space
    val_preds_orig = val_predictions.copy()
    val_targets_orig = val_targets_np.copy()
    for i_col, col in enumerate(target_cols):
        if col != "LogD":
            val_preds_orig[:, i_col] = 10 ** val_predictions[:, i_col]
            val_targets_orig[:, i_col] = 10 ** val_targets_np[:, i_col]

    val_ma_rae = compute_ma_rae(val_preds_orig, val_targets_orig, val_mask_np)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/train_batches:.4f}, Val MA-RAE: {val_ma_rae:.4f}"
        )

    if val_ma_rae < best_val_ma_rae:
        best_val_ma_rae = val_ma_rae
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model and final validation
model.load_state_dict(best_model_state)
model.eval()

val_predictions = []
with torch.no_grad():
    for features, _, _ in val_loader:
        features = features.to(device)
        predictions = model(features)
        val_predictions.append(predictions.cpu().numpy())
val_predictions = np.concatenate(val_predictions, axis=0)

val_preds_orig = val_predictions.copy()
val_targets_orig = val_targets_np.copy()
for i_col, col in enumerate(target_cols):
    if col != "LogD":
        val_preds_orig[:, i_col] = 10 ** val_predictions[:, i_col]
        val_targets_orig[:, i_col] = 10 ** val_targets_np[:, i_col]

final_val_ma_rae = compute_ma_rae(val_preds_orig, val_targets_orig, val_mask_np)
print(f"\nBest validation MA-RAE: {best_val_ma_rae:.4f}")
print(f"Final validation MA-RAE: {final_val_ma_rae:.4f}")

# Test inference
test_predictions = []
with torch.no_grad():
    for batch in test_loader:
        features = batch[0].to(device)
        predictions = model(features)
        test_predictions.append(predictions.cpu().numpy())
test_predictions = np.concatenate(test_predictions, axis=0)

# Inverse log transform for submission
for i_col, col in enumerate(target_cols):
    if col != "LogD":
        test_predictions[:, i_col] = 10 ** test_predictions[:, i_col]

submission_df = pd.DataFrame(
    {
        "Molecule Name": test_names,
        "LogD": test_predictions[:, 0],
        "KSOL": test_predictions[:, 1],
        "HLM CLint": test_predictions[:, 2],
        "MLM CLint": test_predictions[:, 3],
        "Caco-2 Permeability Papp A>B": test_predictions[:, 4],
        "Caco-2 Permeability Efflux": test_predictions[:, 5],
        "MPPB": test_predictions[:, 6],
        "MBPB": test_predictions[:, 7],
        "MGMB": test_predictions[:, 8],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

score = final_val_ma_rae
print(f"Final Validation Score: {score}")