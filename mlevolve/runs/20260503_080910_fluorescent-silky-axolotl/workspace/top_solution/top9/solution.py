import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.Avalon import pyAvalonTools
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os
import joblib

warnings.filterwarnings("ignore")

print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

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
train_targets_orig = train_df[target_cols].copy()

# Apply log10 transform to non-LogD targets (with safe clipping to avoid -inf)
for col in log_transform_cols:
    mask = ~train_df[col].isna()
    if mask.sum() > 0:
        vals = train_df.loc[mask, col].values.astype(float)
        # Clip to a small positive value to avoid log(0) or log(negative) producing -inf/NaN
        vals = np.maximum(vals, 1e-6)
        train_df.loc[mask, col] = np.log10(vals)

print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")


# Feature extraction function
def get_rdkit_features(smiles_list):
    """Extract comprehensive RDKit features from SMILES strings."""
    morgan_fps = []
    maccs_fps = []
    avalon_fps = []
    physchem_feats = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            morgan_fps.append(np.zeros(2048, dtype=np.float32))
            maccs_fps.append(np.zeros(167, dtype=np.float32))
            avalon_fps.append(np.zeros(512, dtype=np.float32))
            physchem_feats.append(np.zeros(55, dtype=np.float32))
            continue

        # Morgan fingerprints (radius=2, 2048 bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        morgan_fps.append(arr.copy())

        # MACCS keys (167 bits)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        arr_m = np.zeros((167,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(maccs, arr_m)
        maccs_fps.append(arr_m.copy())

        # Avalon fingerprints (512 bits)
        avalon = pyAvalonTools.GetAvalonFP(mol, nBits=512)
        arr_a = np.zeros((512,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(avalon, arr_a)
        avalon_fps.append(arr_a.copy())

        # Physicochemical descriptors (55 features)
        desc = []
        desc.append(Descriptors.MolWt(mol))
        desc.append(Descriptors.NumHDonors(mol))
        desc.append(Descriptors.NumHAcceptors(mol))
        desc.append(Descriptors.MolLogP(mol))
        desc.append(Descriptors.NumRotatableBonds(mol))
        desc.append(Descriptors.TPSA(mol))
        desc.append(Descriptors.NumAromaticRings(mol))
        desc.append(Descriptors.NumAliphaticRings(mol))
        desc.append(Descriptors.RingCount(mol))
        desc.append(Descriptors.NumSaturatedRings(mol))
        desc.append(Descriptors.NumHeteroatoms(mol))
        desc.append(Descriptors.FractionCSP3(mol))
        desc.append(Descriptors.NumValenceElectrons(mol))
        desc.append(Descriptors.MaxPartialCharge(mol))
        desc.append(Descriptors.MinPartialCharge(mol))
        desc.append(Descriptors.BalabanJ(mol))
        desc.append(Descriptors.BertzCT(mol))
        desc.append(Descriptors.Chi0(mol))
        desc.append(Descriptors.Chi1(mol))
        desc.append(Descriptors.Chi2n(mol))
        desc.append(Descriptors.Chi3n(mol))
        desc.append(Descriptors.Chi4n(mol))
        desc.append(Descriptors.Chi0v(mol))
        desc.append(Descriptors.Chi1v(mol))
        desc.append(Descriptors.Chi2v(mol))
        desc.append(Descriptors.Chi3v(mol))
        desc.append(Descriptors.Chi4v(mol))
        desc.append(Descriptors.EState_VSA1(mol))
        desc.append(Descriptors.EState_VSA2(mol))
        desc.append(Descriptors.EState_VSA3(mol))
        desc.append(Descriptors.EState_VSA4(mol))
        desc.append(Descriptors.EState_VSA5(mol))
        desc.append(Descriptors.EState_VSA6(mol))
        desc.append(Descriptors.EState_VSA7(mol))
        desc.append(Descriptors.EState_VSA8(mol))
        desc.append(Descriptors.EState_VSA9(mol))
        desc.append(Descriptors.EState_VSA10(mol))
        desc.append(Descriptors.EState_VSA11(mol))
        desc.append(Descriptors.VSA_EState1(mol))
        desc.append(Descriptors.VSA_EState2(mol))
        desc.append(Descriptors.VSA_EState3(mol))
        desc.append(Descriptors.VSA_EState4(mol))
        desc.append(Descriptors.VSA_EState5(mol))
        desc.append(Descriptors.VSA_EState6(mol))
        desc.append(Descriptors.VSA_EState7(mol))
        desc.append(Descriptors.VSA_EState8(mol))
        desc.append(Descriptors.VSA_EState9(mol))
        desc.append(Descriptors.VSA_EState10(mol))
        desc.append(Descriptors.MolMR(mol))
        desc.append(Descriptors.HeavyAtomCount(mol))
        desc.append(Descriptors.NumHeterocycles(mol))
        desc.append(Descriptors.NumAromaticHeterocycles(mol))
        desc.append(Descriptors.NumSaturatedHeterocycles(mol))
        desc.append(Descriptors.NumAliphaticHeterocycles(mol))
        desc.append(Descriptors.NumAromaticCarbocycles(mol))
        physchem_feats.append(np.array(desc, dtype=np.float32))

    morgan_arr = np.array(morgan_fps, dtype=np.float32)
    maccs_arr = np.array(maccs_fps, dtype=np.float32)
    avalon_arr = np.array(avalon_fps, dtype=np.float32)
    physchem_arr = np.array(physchem_feats, dtype=np.float32)

    return np.concatenate([morgan_arr, maccs_arr, avalon_arr, physchem_arr], axis=1)


print("Extracting features for training data...")
train_features = get_rdkit_features(train_df["SMILES"].values)
print(f"Train features shape: {train_features.shape}")

print("Extracting features for test data...")
test_features = get_rdkit_features(test_df["SMILES"].values)
print(f"Test features shape: {test_features.shape}")

# Create data availability pattern for stratified split
avail_counts = []
for _, row in train_df[target_cols].iterrows():
    avail = sum(~np.isnan(row.values.astype(float)))
    avail_counts.append(avail)
avail_pattern = pd.qcut(avail_counts, q=4, labels=False, duplicates="drop")
unique_groups = len(set(avail_pattern))
n_splits = min(5, unique_groups)
print(f"Using {n_splits}-fold cross-validation ({unique_groups} unique groups)")

gkf = GroupKFold(n_splits=n_splits)
groups = avail_pattern

for train_idx, val_idx in gkf.split(train_df, groups=groups):
    train_idx = train_idx
    val_idx = val_idx
    break

train_features_split = train_features[train_idx]
val_features_split = train_features[val_idx]

train_targets = train_df[target_cols].values[train_idx].astype(float)
val_targets = train_df[target_cols].values[val_idx].astype(float)

train_mask = ~np.isnan(train_targets)
val_mask = ~np.isnan(val_targets)

train_targets = np.nan_to_num(train_targets, nan=0.0)
val_targets = np.nan_to_num(val_targets, nan=0.0)

print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

# Standardize features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features_split)
val_features_scaled = scaler.transform(val_features_split)
test_features_scaled = scaler.transform(test_features)

# Save original targets for MA-RAE computation
val_df_orig = train_df.iloc[val_idx]
val_targets_original = np.zeros_like(val_targets)
for i, col in enumerate(target_cols):
    vals = val_df_orig[col].values.astype(float)
    val_targets_original[:, i] = np.nan_to_num(vals, nan=0.0)

# Convert to torch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_features_tensor = torch.FloatTensor(train_features_scaled).to(device)
val_features_tensor = torch.FloatTensor(val_features_scaled).to(device)
test_features_tensor = torch.FloatTensor(test_features_scaled).to(device)
train_targets_tensor = torch.FloatTensor(train_targets).to(device)
val_targets_tensor = torch.FloatTensor(val_targets).to(device)
train_mask_tensor = torch.BoolTensor(train_mask).to(device)
val_mask_tensor = torch.BoolTensor(val_mask).to(device)

# Custom Dataset with random noise augmentation (feature noise only, not SMILES augmentation)
class AugmentedMoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, features_tensor, targets_tensor, mask_tensor, noise_std=0.02, augment_prob=0.3):
        self.features_tensor = features_tensor
        self.targets_tensor = targets_tensor
        self.mask_tensor = mask_tensor
        self.noise_std = noise_std
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.features_tensor)

    def __getitem__(self, idx):
        features = self.features_tensor[idx]
        targets = self.targets_tensor[idx]
        mask = self.mask_tensor[idx]

        # Apply small Gaussian noise to features (scaled version) with probability
        if np.random.random() < self.augment_prob:
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise

        return features, targets, mask


# Create augmented training dataset
train_dataset = AugmentedMoleculeDataset(
    train_features_tensor, train_targets_tensor, train_mask_tensor,
    noise_std=0.02, augment_prob=0.3
)

val_dataset = TensorDataset(val_features_tensor, val_targets_tensor, val_mask_tensor)
test_dataset = TensorDataset(test_features_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Model definition with task-specific heads
class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512], dropout=0.3, num_tasks=9):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.shared_encoder = nn.Sequential(*layers)

        # Simpler task-specific heads with shared representation
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1),
            ) for _ in range(num_tasks)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        shared = self.shared_encoder(x)  # (batch, prev_dim)
        outputs = []
        for head in self.task_heads:
            outputs.append(head(shared))
        return torch.cat(outputs, dim=1)


# Simple MSE loss with masking
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        loss = (pred - target) ** 2
        loss = loss * mask.float()
        return loss.sum() / (mask.sum() + 1e-8)


# Initialize model, loss, optimizer
input_dim = train_features_scaled.shape[1]
model = MultiTaskMLP(
    input_dim=input_dim,
    hidden_dims=[1024, 512, 256],
    dropout=0.3,
    num_tasks=9
).to(device)
criterion = MaskedMSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
num_epochs = 200
best_val_metric = float("inf")
patience = 20
patience_counter = 0
grad_clip_norm = 5.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_features, batch_targets, batch_mask in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_targets, batch_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    all_val_preds = []

    with torch.no_grad():
        for batch_features, batch_targets, batch_mask in val_loader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets, batch_mask)
            val_loss += loss.item()
            all_val_preds.append(outputs.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_predictions = np.concatenate(all_val_preds, axis=0)

    # Compute MA-RAE (using original-scale targets for proper comparison)
    # RAE = |pred - true| / |true| (relative error relative to true value magnitude)
    task_raes = []
    for i_col in range(len(target_cols)):
        valid_idx = np.where(val_mask[:, i_col])[0]
        if len(valid_idx) > 0:
            preds_i = val_predictions[valid_idx, i_col]
            # Convert predictions back to original scale for non-LogD targets
            col_name = target_cols[i_col]
            if col_name != "LogD":
                preds_i = 10 ** preds_i
            targets_i = val_targets_original[valid_idx, i_col]
            abs_errors = np.abs(preds_i - targets_i)
            # Normalize by absolute true value (standard RAE denominator)
            denominator = np.abs(targets_i)
            # Use max of |true| and 0.01 to avoid division by very small true values
            denominator = np.maximum(denominator, 0.01)
            rae = abs_errors / denominator
            # Clip RAE to a reasonable range [0, 100] to avoid extreme outliers dominating
            rae = np.clip(rae, 0, 100)
            task_raes.append(np.mean(rae))

    val_ma_rae = np.mean(task_raes) if len(task_raes) > 0 else 1.0

    scheduler.step()

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, MA-RAE: {val_ma_rae:.4f}"
    )

    if val_ma_rae < best_val_metric:
        best_val_metric = val_ma_rae
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Test predictions
model.eval()
all_test_preds = []
with torch.no_grad():
    for batch_features in test_loader:
        outputs = model(batch_features[0].to(device))
        all_test_preds.append(outputs.cpu().numpy())

test_predictions = np.concatenate(all_test_preds, axis=0)

# Inverse transform for submission
for i_col, col in enumerate(target_cols):
    if col != "LogD":
        test_predictions[:, i_col] = 10 ** test_predictions[:, i_col]

# Clip predictions to reasonable ranges
test_predictions[:, 0] = np.clip(test_predictions[:, 0], -5.0, 10.0)
test_predictions[:, 1] = np.clip(test_predictions[:, 1], 0.0, 10000.0)
test_predictions[:, 2] = np.clip(test_predictions[:, 2], 0.0, 100000.0)
test_predictions[:, 3] = np.clip(test_predictions[:, 3], 0.0, 100000.0)
test_predictions[:, 4] = np.clip(test_predictions[:, 4], 0.0, 1000.0)
test_predictions[:, 5] = np.clip(test_predictions[:, 5], 0.0, 1000.0)
test_predictions[:, 6] = np.clip(test_predictions[:, 6], 0.0, 100.0)
test_predictions[:, 7] = np.clip(test_predictions[:, 7], 0.0, 100.0)
test_predictions[:, 8] = np.clip(test_predictions[:, 8], 0.0, 100.0)

# Create submission dataframe
submission_df = pd.DataFrame(
    {
        "Molecule Name": test_df["Molecule Name"].values,
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

print(f"Final Validation Score: {best_val_metric:.6f}")