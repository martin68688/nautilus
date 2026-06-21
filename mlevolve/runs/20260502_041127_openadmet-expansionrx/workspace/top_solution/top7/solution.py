import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

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

log_transform_cols = [
    "HLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
]


# ============================================================
# 2. FEATURE ENGINEERING WITH RDKit
# ============================================================
import random as _random

def augment_smiles(smi, prob=0.8):
    """With given probability, generate a random non-canonical SMILES."""
    if _random.random() < prob:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return Chem.MolToSmiles(mol, doRandom=True)
        except Exception:
            pass
    return smi

def smi_to_features(smiles_list, do_augment=False):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    descriptors = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        try:
            if do_augment:
                smi = augment_smiles(smi)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotb = Descriptors.NumRotatableBonds(mol)
            tpsa = Descriptors.TPSA(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            saturated_rings = rdMolDescriptors.CalcNumSaturatedRings(mol)
            hetero_atoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
            frac_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
            ring_count = rdMolDescriptors.CalcNumRings(mol)
            aromatic_proportion = sum(
                1 for atom in mol.GetAtoms() if atom.GetIsAromatic()
            ) / max(heavy_atoms, 1)

            descriptors.append(
                [
                    mw,
                    logp,
                    hbd,
                    hba,
                    rotb,
                    tpsa,
                    heavy_atoms,
                    aromatic_rings,
                    saturated_rings,
                    hetero_atoms,
                    frac_sp3,
                    ring_count,
                    aromatic_proportion,
                ]
            )
            valid_indices.append(i)
        except Exception:
            continue

    descriptor_array = np.array(descriptors) if descriptors else np.empty((0, 13))
    valid_smiles = [smiles_list[i] for i in valid_indices] if valid_indices else []
    fingerprints = []
    for smi in valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = mfpgen.GetFingerprint(mol)
            fingerprints.append(np.array(fp))
    fingerprint_array = np.vstack(fingerprints) if fingerprints else np.empty((0, 2048))

    return descriptor_array, fingerprint_array, valid_indices


desc_names = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "TPSA",
    "NumHeavyAtoms",
    "NumAromaticRings",
    "NumSaturatedRings",
    "NumHeteroatoms",
    "FractionCSP3",
    "NumRings",
    "AromaticProportion",
]

print("Processing training molecules...")
train_desc, train_fp, train_valid = smi_to_features(train["SMILES"].values)
print("Processing test molecules...")
test_desc, test_fp, test_valid = smi_to_features(test["SMILES"].values)

train_feat = pd.DataFrame(train_desc, columns=desc_names)
train_fp_df = pd.DataFrame(train_fp, columns=[f"fp_{i}" for i in range(2048)])
train_features = pd.concat([train_feat, train_fp_df], axis=1)

test_feat = pd.DataFrame(test_desc, columns=desc_names)
test_fp_df = pd.DataFrame(test_fp, columns=[f"fp_{i}" for i in range(2048)])
test_features = pd.concat([test_feat, test_fp_df], axis=1)

train_valid_df = train.iloc[train_valid].reset_index(drop=True)
test_valid_df = test.iloc[test_valid].reset_index(drop=True)

# ============================================================
# 3. CREATE VALIDATION SPLIT (BEFORE target preprocessing)
# ============================================================
def get_molwt(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Descriptors.MolWt(mol)
    except Exception:
        pass
    return 300.0


molwt_values = train_valid_df["SMILES"].apply(get_molwt)
molwt_bins = pd.qcut(molwt_values, q=5, labels=False, duplicates="drop")

# Split data first before any target preprocessing
train_idx, val_idx = train_test_split(
    np.arange(len(train_valid_df)),
    test_size=0.2,
    random_state=42,
    stratify=molwt_bins,
)

train_split_df = train_valid_df.iloc[train_idx].reset_index(drop=True)
val_split_df = train_valid_df.iloc[val_idx].reset_index(drop=True)
X_train_split = train_features.iloc[train_idx].reset_index(drop=True)
X_val_split = train_features.iloc[val_idx].reset_index(drop=True)


# ============================================================
# 4. HANDLE MISSING TARGETS (fit on train only)
# ============================================================
target_train = train_split_df[target_cols].copy()
for col in log_transform_cols:
    if col in target_cols:
        target_train[col] = np.log1p(target_train[col].clip(lower=0))

target_val = val_split_df[target_cols].copy()
for col in log_transform_cols:
    if col in target_cols:
        target_val[col] = np.log1p(target_val[col].clip(lower=0))

knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
target_train_imputed = knn_imputer.fit_transform(target_train.values)
target_val_imputed = knn_imputer.transform(target_val.values)
target_imputed_df = pd.DataFrame(target_train_imputed, columns=target_cols)
target_val_imputed_df = pd.DataFrame(target_val_imputed, columns=target_cols)

# Compute target ranges for MA-RAE using ORIGINAL scale non-log targets (train only)
train_orig = train_split_df[target_cols].values
target_ranges = {}
for j, col in enumerate(target_cols):
    valid_mask = ~np.isnan(train_orig[:, j])
    if valid_mask.sum() > 0:
        target_min = np.nanmin(train_orig[:, j])
        target_max = np.nanmax(train_orig[:, j])
    else:
        target_min, target_max = 0.0, 1.0
    target_ranges[col] = target_max - target_min if target_max > target_min else 1.0

# Pre-compute clipping bounds for inverse log-transform using train only
clip_bounds = {}
for col in target_cols:
    if col in log_transform_cols:
        orig_vals = train_split_df[col].dropna().clip(lower=0).values
        clip_bounds[col] = (np.min(orig_vals), np.max(orig_vals))
    else:
        clip_bounds[col] = (0.0, 1.0)


# ============================================================
# 5. SCALE FEATURES (fit on train only)
# ============================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_split.values)
X_val = scaler.transform(X_val_split.values)
X_test = scaler.transform(test_features.values)

y_train = target_train_imputed
y_val = target_val_imputed


# ============================================================
# 5. MODEL DEFINITION
# ============================================================
class MoleculeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_tasks=9, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim // 8, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(64, 1),
                )
                for _ in range(num_tasks)
            ]
        )
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, x):
        features = self.net(x)
        outputs = [head(features) for head in self.task_heads]
        return torch.cat(outputs, dim=1)


input_dim = X_train.shape[1]
model = MoleculeMLP(
    input_dim=input_dim, hidden_dim=1024, num_tasks=len(target_cols), dropout=0.4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ============================================================
# 6. TRAINING SETUP
# ============================================================
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)

batch_size = 64
num_epochs = 500
early_stopping_patience = 30
best_val_score = float("inf")
patience_counter = 0
best_model_state = None

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)


def compute_ma_rae(y_true, y_pred, target_ranges_dict, target_cols_list):
    maes = []
    for j, col in enumerate(target_cols_list):
        valid_mask = ~np.isnan(y_true[:, j])
        if valid_mask.sum() > 0:
            mae = mean_absolute_error(y_true[valid_mask, j], y_pred[valid_mask, j])
            rae = mae / max(target_ranges_dict[col], 1e-10)
            maes.append(rae)
    return np.mean(maes) if maes else 0.0


n_train = len(X_train_tensor)

for epoch in range(num_epochs):
    model.train()
    indices = torch.randperm(n_train)
    epoch_loss = 0.0
    n_batches = 0

    for start_idx in range(0, n_train, batch_size):
        end_idx = min(start_idx + batch_size, n_train)
        batch_indices = indices[start_idx:end_idx]

        # Use pre-computed features - no online SMILES augmentation to avoid scaler mismatch
        x_batch = X_train_tensor[batch_indices]

        y_batch = y_train_tensor[batch_indices]

        optimizer.zero_grad()
        predictions = model(x_batch)

        # --- Per-task gradient scaling based on target sparsity ---
        total_loss = 0.0
        per_task_losses = []
        for t in range(predictions.shape[1]):
            task_pred = predictions[:, t]
            task_target = y_batch[:, t]
            delta = torch.abs(task_pred - task_target)
            huber_loss = torch.where(delta < 1.0, 0.5 * delta**2, delta - 0.5)
            # Count observed (non-NaN) samples in this batch for task t
            # Since we imputed NaN, we use the original target array to compute sparsity
            orig_vals = train_split_df[target_cols[t]].values[batch_indices.cpu().numpy()]
            observed_mask = ~np.isnan(orig_vals)
            n_observed = observed_mask.sum()
            # Compute scaling factor: higher scaling for sparse tasks (fewer observed samples)
            batch_size_actual = len(batch_indices)
            sparsity_ratio = n_observed / max(batch_size_actual, 1)
            # Invert: tasks with fewer observed samples get higher gradient scaling
            gradient_scale = 1.0 / max(sparsity_ratio, 0.1)
            task_loss = huber_loss.mean()
            precision = torch.exp(-model.log_vars[t])
            weighted_loss = precision * task_loss + model.log_vars[t] / 2.0
            # Apply gradient scaling factor to the weighted loss
            weighted_loss = weighted_loss * gradient_scale
            total_loss += weighted_loss
            per_task_losses.append(weighted_loss.item() if hasattr(weighted_loss, 'item') else 0.0)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        epoch_loss += total_loss.item()
        n_batches += 1

    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor).cpu().numpy()

    val_pred_orig = val_predictions.copy()
    val_true_orig = y_val.copy()

    for j, col in enumerate(target_cols):
        if col in log_transform_cols:
            low, high = clip_bounds[col]
            val_pred_orig[:, j] = np.expm1(
                np.clip(val_predictions[:, j], 0.0, np.log1p(high))
            )
            val_true_orig[:, j] = np.expm1(np.clip(y_val[:, j], 0.0, np.log1p(high)))

    val_score = compute_ma_rae(val_true_orig, val_pred_orig, target_ranges, target_cols)
    scheduler.step()

    if epoch % 5 == 0 or epoch < 10:
        print(
            f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/n_batches:.4f} | Val MA-RAE: {val_score:.4f}"
        )

    if val_score < best_val_score:
        best_val_score = val_score
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\nBest validation MA-RAE: {best_val_score:.6f}")

# ============================================================
# 7. FINAL PREDICTIONS
# ============================================================
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).cpu().numpy()

test_pred_orig = test_predictions.copy()
for j, col in enumerate(target_cols):
    if col in log_transform_cols:
        low, high = clip_bounds[col]
        test_pred_orig[:, j] = np.expm1(
            np.clip(test_predictions[:, j], 0.0, np.log1p(high))
        )

# ============================================================
# 8. CREATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {
        "Molecule Name": test_valid_df["Molecule Name"].values,
        target_cols[0]: test_pred_orig[:, 0],
        target_cols[1]: test_pred_orig[:, 1],
        target_cols[2]: test_pred_orig[:, 2],
        target_cols[3]: test_pred_orig[:, 3],
        target_cols[4]: test_pred_orig[:, 4],
        target_cols[5]: test_pred_orig[:, 5],
        target_cols[6]: test_pred_orig[:, 6],
        target_cols[7]: test_pred_orig[:, 7],
        target_cols[8]: test_pred_orig[:, 8],
    }
)

for col in target_cols:
    submission[col] = submission[col].astype(float)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")

# Final validation score
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    final_val_preds = model(X_val_tensor).cpu().numpy()

final_val_preds_orig = final_val_preds.copy()
final_val_true_orig = y_val.copy()
for j, col in enumerate(target_cols):
    if col in log_transform_cols:
        low, high = clip_bounds[col]
        final_val_preds_orig[:, j] = np.expm1(
            np.clip(final_val_preds[:, j], 0.0, np.log1p(high))
        )
        final_val_true_orig[:, j] = np.expm1(np.clip(y_val[:, j], 0.0, np.log1p(high)))

final_val_score = compute_ma_rae(
    final_val_true_orig, final_val_preds_orig, target_ranges, target_cols
)
print(f"Final Validation Score: {final_val_score:.6f}")