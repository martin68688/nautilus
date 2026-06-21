import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")

# ============================================================
# 1. MORGAN FINGERPRINT FEATURIZATION (deterministic, no network)
# ============================================================
def get_morgan_fp(smiles, radius=3, bit_length=4096):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(bit_length, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_length)
        return np.array(fp, dtype=np.float32)
    except:
        return np.zeros(bit_length, dtype=np.float32)

from rdkit.Chem import Descriptors, Crippen, MolSurf, GraphDescriptors

def get_rdkit_descs(mol):
    """Compute 20 RDKit physicochemical descriptors."""
    try:
        if mol is None:
            return np.zeros(20, dtype=np.float32)
        descs = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            MolSurf.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.NumSaturatedRings(mol),
            GraphDescriptors.Chi0v(mol),
            GraphDescriptors.HallKierAlpha(mol),
            GraphDescriptors.Kappa1(mol),
            GraphDescriptors.Kappa2(mol),
            GraphDescriptors.Kappa3(mol),
            GraphDescriptors.Phi(mol),
            GraphDescriptors.BalabanJ(mol),
            Descriptors.Ipc(mol),
        ]
        return np.array(descs, dtype=np.float32)
    except:
        return np.zeros(20, dtype=np.float32)


# ============================================================
# 2. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

test_names = test_df["Molecule Name"].values

# ============================================================
# 3. COMPUTE FINGERPRINT FEATURES
# ============================================================
print("Computing Morgan fingerprints and RDKit descriptors for train...")
train_mols = [Chem.MolFromSmiles(s) for s in train_df["SMILES"].values]
train_fp = np.array([get_morgan_fp(s) for s in train_df["SMILES"].values], dtype=np.float32)
train_desc = np.array([get_rdkit_descs(mol) for mol in train_mols], dtype=np.float32)

print("Computing Morgan fingerprints and RDKit descriptors for test...")
test_mols = [Chem.MolFromSmiles(s) for s in test_df["SMILES"].values]
test_fp = np.array([get_morgan_fp(s) for s in test_df["SMILES"].values], dtype=np.float32)
test_desc = np.array([get_rdkit_descs(mol) for mol in test_mols], dtype=np.float32)

# Standardize physicochemical descriptors using training set statistics
desc_mean = np.mean(train_desc, axis=0)
desc_std = np.std(train_desc, axis=0)
desc_std = np.where(desc_std < 1e-8, 1.0, desc_std)
train_desc = (train_desc - desc_mean) / desc_std
test_desc = (test_desc - desc_mean) / desc_std

# Concatenate fingerprints and descriptors
train_fps = np.concatenate([train_fp, train_desc], axis=1)
test_fps = np.concatenate([test_fp, test_desc], axis=1)
input_dim = train_fps.shape[1]
print(f"Feature dimension: {input_dim}")

# ============================================================
# 4. HANDLE TARGET VARIABLES
# ============================================================
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

train_y = train_df[target_cols].copy()
missing_indicators = train_y.isna().astype(float)
missing_indicators.columns = [f"{col}_missing" for col in target_cols]

# ============================================================
# 5. SPLIT DATA
# ============================================================
logd_bins = pd.qcut(
    train_df["LogD"].fillna(train_df["LogD"].median()),
    q=4,
    labels=False,
    duplicates="drop",
)
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=42,
    stratify=logd_bins if len(logd_bins.unique()) > 1 else None,
)

# ============================================================
# 6. PREPARE DATA
# ============================================================
y_train = train_y.iloc[train_idx].values.astype(np.float32)
y_val = train_y.iloc[val_idx].values.astype(np.float32)
missing_train = missing_indicators.iloc[train_idx].values.astype(np.float32)
missing_val = missing_indicators.iloc[val_idx].values.astype(np.float32)

train_fp_data = train_fps[train_idx]
val_fp_data = train_fps[val_idx]
test_fp_data = test_fps

# ============================================================
# 7. COMPUTE PER-TARGET NORMALIZATION STATISTICS (ON TRAIN FOLD ONLY)
# ============================================================
target_means = np.zeros(len(target_cols), dtype=np.float32)
target_stds = np.ones(len(target_cols), dtype=np.float32)
for i in range(len(target_cols)):
    valid_mask = ~np.isnan(y_train[:, i])
    if valid_mask.sum() > 0:
        target_means[i] = np.nanmean(y_train[:, i])
        target_stds[i] = np.nanstd(y_train[:, i])
        if target_stds[i] < 1e-8:
            target_stds[i] = 1.0

y_train_norm = (y_train - target_means) / target_stds
y_val_norm = (y_val - target_means) / target_stds
y_train_norm = np.nan_to_num(y_train_norm, nan=0.0)
y_val_norm = np.nan_to_num(y_val_norm, nan=0.0)

# ============================================================
# 8. MODEL DEFINITION - MLP-based multi-task predictor
# ============================================================
class GaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class MultiTaskPredictor(nn.Module):
    def __init__(self, input_dim=4116, hidden_dim=1024, num_targets=9, dropout_rate=0.5):
        super(MultiTaskPredictor, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            GaussianNoise(std=0.05),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.target_heads = nn.ModuleList()
        for _ in range(num_targets):
            head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.target_heads.append(head)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        shared_out = self.shared(x)
        outputs = []
        for head in self.target_heads:
            out = head(shared_out)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(MaskedSmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, predictions, targets, mask):
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        smooth_l1 = torch.where(
            abs_diff < self.beta,
            0.5 * (diff**2) / self.beta,
            abs_diff - 0.5 * self.beta,
        )
        masked_loss = smooth_l1 * mask
        valid_count = mask.sum()
        if valid_count > 0:
            loss = masked_loss.sum() / valid_count
        else:
            loss = torch.tensor(0.0, device=predictions.device)
        return loss


# ============================================================
# 9. SETUP MODEL, LOSS, OPTIMIZER
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MultiTaskPredictor(
    input_dim=input_dim,
    hidden_dim=1024,
    num_targets=len(target_cols),
    dropout_rate=0.5
).to(device)

criterion = MaskedSmoothL1Loss(beta=1.0)
optimizer = optim.AdamW(
    model.parameters(), lr=5e-4, weight_decay=5e-4, betas=(0.9, 0.999), eps=1e-8
)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# ============================================================
# 10. DATASET & DATA LOADER
# ============================================================
batch_size = 64

class FPDataset(Dataset):
    def __init__(self, fps, y_values, missing_values):
        self.fps = torch.FloatTensor(fps)
        self.y_values = torch.FloatTensor(y_values)
        self.missing_values = torch.FloatTensor(missing_values)

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        return self.fps[idx], self.y_values[idx], self.missing_values[idx]

train_dataset = FPDataset(train_fp_data, y_train_norm, missing_train)
val_dataset = FPDataset(val_fp_data, y_val_norm, missing_val)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

# ============================================================
# 11. TRAINING LOOP
# ============================================================
num_epochs = 150
patience = 15
best_val_score = float("inf")
best_model_state = None
epochs_no_improve = 0

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_batches = 0
    for batch_fp, batch_y, batch_mask in train_loader:
        batch_fp = batch_fp.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)
        optimizer.zero_grad()
        predictions = model(batch_fp)
        loss = criterion(predictions, batch_y, batch_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1
    avg_train_loss = train_loss / max(train_batches, 1)

    model.eval()
    val_predictions_list = []
    val_targets_list = []
    val_masks_list = []
    with torch.no_grad():
        for batch_fp, batch_y, batch_mask in val_loader:
            batch_fp = batch_fp.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            predictions = model(batch_fp)
            val_predictions_list.append(predictions.cpu().numpy())
            val_targets_list.append(batch_y.cpu().numpy())
            val_masks_list.append(batch_mask.cpu().numpy())
    val_predictions = np.concatenate(val_predictions_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)
    val_masks = np.concatenate(val_masks_list, axis=0)
    val_predictions_denorm = val_predictions * target_stds + target_means
    val_targets_denorm = val_targets * target_stds + target_means
    ma_rae_scores = []
    for t in range(len(target_cols)):
        t_mask = val_masks[:, t] > 0.5
        if t_mask.sum() > 0:
            t_pred = val_predictions_denorm[t_mask, t]
            t_true = val_targets_denorm[t_mask, t]
            mae = np.mean(np.abs(t_pred - t_true))
            mean_abs_true = np.mean(np.abs(t_true))
            if mean_abs_true > 1e-8:
                relative_error = mae / mean_abs_true
            else:
                relative_error = mae / (mean_abs_true + 1e-8)
            ma_rae_scores.append(relative_error)
    val_ma_rae = np.mean(ma_rae_scores) if len(ma_rae_scores) > 0 else float("inf")
    scheduler.step()
    if val_ma_rae < best_val_score:
        best_val_score = val_ma_rae
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val MA-RAE: {val_ma_rae:.4f} | Best: {best_val_score:.4f}")
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print(f"Training complete. Best validation MA-RAE: {best_val_score:.4f}")

# ============================================================
# 12. LOAD BEST MODEL AND FINAL VALIDATION
# ============================================================
model.load_state_dict(best_model_state)
model = model.to(device)
model.eval()

val_predictions_list = []
with torch.no_grad():
    for batch_fp, _, _ in val_loader:
        batch_fp = batch_fp.to(device, non_blocking=True)
        predictions = model(batch_fp)
        val_predictions_list.append(predictions.cpu().numpy())
val_predictions = np.concatenate(val_predictions_list, axis=0)
val_predictions_denorm = val_predictions * target_stds + target_means
ma_rae_scores_final = []
for t in range(len(target_cols)):
    t_mask = missing_val[:, t] > 0.5
    if t_mask.sum() > 0:
        t_pred = val_predictions_denorm[t_mask, t]
        t_true = y_val[t_mask, t]
        mae = np.mean(np.abs(t_pred - t_true))
        mean_abs_true = np.mean(np.abs(t_true))
        if mean_abs_true > 1e-8:
            relative_error = mae / mean_abs_true
        else:
            relative_error = mae / (mean_abs_true + 1e-8)
        ma_rae_scores_final.append(relative_error)
final_ma_rae = (
    np.mean(ma_rae_scores_final) if len(ma_rae_scores_final) > 0 else float("inf")
)

# ============================================================
# 13. TEST INFERENCE
# ============================================================
test_dataset = FPDataset(test_fp_data, np.zeros((len(test_fp_data), len(target_cols)), dtype=np.float32),
                         np.zeros((len(test_fp_data), len(target_cols)), dtype=np.float32))
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)
test_predictions_list = []
with torch.no_grad():
    for batch_fp, _, _ in test_loader:
        batch_fp = batch_fp.to(device, non_blocking=True)
        predictions = model(batch_fp)
        test_predictions_list.append(predictions.cpu().numpy())
test_predictions = np.concatenate(test_predictions_list, axis=0)
test_predictions_denorm = test_predictions * target_stds + target_means

for t in range(len(target_cols)):
    valid_mask = ~np.isnan(y_train[:, t])
    if valid_mask.sum() > 0:
        train_min = np.nanmin(y_train[:, t])
        train_max = np.nanmax(y_train[:, t])
        range_margin = 0.2 * (train_max - train_min)
        clip_min = train_min - range_margin
        clip_max = train_max + range_margin
        test_predictions_denorm[:, t] = np.clip(
            test_predictions_denorm[:, t], clip_min, clip_max
        )

# ============================================================
# 14. CREATE SUBMISSION FILE
# ============================================================
submission_df = pd.DataFrame({"Molecule Name": test_names})
for i, col in enumerate(target_cols):
    submission_df[col] = test_predictions_denorm[:, i]
submission_df = submission_df[sample_sub.columns]
os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Final Validation Score: {final_ma_rae}")