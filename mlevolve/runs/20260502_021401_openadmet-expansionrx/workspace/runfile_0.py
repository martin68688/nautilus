import os
os.sched_setaffinity(0, {0, 1, 2, 3, 4, 5, 6})
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# Reproducibility
# ==============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ==============================================================================
# Paths
# ==============================================================================
DATA_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

# ==============================================================================
# 1. Data Processing & Feature Engineering
# ==============================================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

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

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


# Clean SMILES
def clean_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except:
        pass
    return None


train_df["clean_smiles"] = train_df["SMILES"].apply(clean_smiles)
test_df["clean_smiles"] = test_df["SMILES"].apply(clean_smiles)

invalid_train = train_df["clean_smiles"].isna().sum()
if invalid_train > 0:
    print(f"Removing {invalid_train} invalid SMILES from training set.")
    train_df = train_df.dropna(subset=["clean_smiles"]).reset_index(drop=True)

invalid_test = test_df["clean_smiles"].isna().sum()
if invalid_test > 0:
    print(f"Warning: {invalid_test} invalid SMILES in test set.")
    test_df["is_valid"] = test_df["clean_smiles"].notna().astype(int)

print(
    f"After cleaning: Train molecules: {len(train_df)}, Test molecules: {len(test_df)}"
)

# Feature extraction functions
DESCRIPTOR_NAMES = [desc[0] for desc in Descriptors._descList]

# No external transformer model – using only RDKit features


def compute_rdkit_features(smiles_list):
    fps = []
    descriptors = []
    maccs = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            fps.append(np.zeros(2048, dtype=np.float32))
            maccs.append(np.zeros(166, dtype=np.float32))
            descriptors.append(np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(np.array(fp, dtype=np.float32))
            mac = MACCSkeys.GenMACCSKeys(mol)
            maccs.append(np.array(mac, dtype=np.float32))
            desc_vals = []
            for name in DESCRIPTOR_NAMES:
                try:
                    val = getattr(Descriptors, name)(mol)
                    if val is None:
                        val = np.nan
                except:
                    val = np.nan
                desc_vals.append(val)
            descriptors.append(np.array(desc_vals, dtype=np.float32))
    fp_array = np.array(fps)
    mac_array = np.array(maccs)
    desc_array = np.array(descriptors)
    fp_cols = [f"morgan_{i}" for i in range(2048)]
    mac_cols = [f"maccs_{i}" for i in range(166)]
    desc_cols = [f"desc_{name}" for name in DESCRIPTOR_NAMES]
    df = pd.DataFrame(np.column_stack([fp_array, mac_array, desc_array]))
    return df


print("Computing RDKit features for train...")
train_rdkit = compute_rdkit_features(train_df["clean_smiles"].tolist())
print("Computing RDKit features for test...")
test_rdkit = compute_rdkit_features(test_df["clean_smiles"].tolist())

print("Using RDKit features only.")
train_features = train_rdkit.values.astype(np.float32)
test_features = test_rdkit.values.astype(np.float32)
print(f"Total features: {train_features.shape[1]}")

# Filter constant / high NaN features
train_df_features = pd.DataFrame(train_features)
nan_ratio = train_df_features.isnull().mean(axis=0)
constant_cols = train_df_features.nunique() == 1
cols_to_drop = (nan_ratio > 0.5) | constant_cols
print(f"Dropping {cols_to_drop.sum()} features (NaN>50% or constant).")
keep_cols = ~cols_to_drop
train_features = train_features[:, keep_cols]
test_features = test_features[:, keep_cols]
print(f"Features after filtering: {train_features.shape[1]}")

# Impute remaining NaN with median
imputer = SimpleImputer(strategy="median")
imputer.fit(train_features)
train_features = imputer.transform(train_features)
test_features = imputer.transform(test_features)

# Scale
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Train/validation split – use indices to preserve molecule names
y_train_full = train_df[target_cols].values.astype(np.float32)
val_ratio = 0.2
all_indices = np.arange(len(train_features_scaled))
train_idx, val_idx = train_test_split(
    all_indices, test_size=val_ratio, random_state=RANDOM_SEED
)
X_train = train_features_scaled[train_idx]
X_val = train_features_scaled[val_idx]
y_train = y_train_full[train_idx]
y_val = y_train_full[val_idx]
print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
print(f"Train features: {X_train.shape[1]}, Val features: {X_val.shape[1]}")
print(f"Train targets shape: {y_train.shape}, Val targets shape: {y_val.shape}")

val_molecule_names = train_df.iloc[val_idx]["Molecule Name"].values
test_molecule_names = test_df["Molecule Name"].values

# ==============================================================================
# 2. Model Design
# ==============================================================================
input_dim = X_train.shape[1]
num_targets = len(target_cols)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        skip = self.skip(x)
        return out + skip


class MultiTaskNN(nn.Module):
    def __init__(self, input_dim, num_tasks, hidden_dims=[1024, 512, 256], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hdim, dropout))
            prev_dim = hdim
        self.shared = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], num_tasks)
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        shared_rep = self.shared(x)
        return self.head(shared_rep)


model = MultiTaskNN(
    input_dim, num_targets, hidden_dims=[1024, 512, 256], dropout=0.3
)
criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200, eta_min=1e-6
)

# ==============================================================================
# 3. Training & Evaluation
# ==============================================================================
train_mask = ~np.isnan(y_train)
val_mask = ~np.isnan(y_val)
y_train_filled = np.nan_to_num(y_train, nan=0.0)
y_val_filled = np.nan_to_num(y_val, nan=0.0)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_filled, dtype=torch.float32)
mask_train_t = torch.tensor(train_mask, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val_filled, dtype=torch.float32)
mask_val_t = torch.tensor(val_mask, dtype=torch.float32)
X_test_t = torch.tensor(test_features_scaled, dtype=torch.float32)

batch_size = 64
train_dataset = TensorDataset(X_train_t, y_train_t, mask_train_t)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
val_dataset = TensorDataset(X_val_t, y_val_t, mask_val_t)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 200
patience = 20
best_val_metric = np.inf
best_epoch = 0
best_model_state = None
early_stop_counter = 0

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y, batch_mask in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)
        optimizer.zero_grad()
        pred_mean, pred_logvar = model(batch_X)
        loss = criterion(pred_mean, pred_logvar, batch_y, batch_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
    avg_train_loss = total_loss / len(train_loader.dataset)

    model.eval()
    all_preds = []
    all_targets = []
    all_mask = []
    with torch.no_grad():
        for batch_X, batch_y, batch_mask in val_loader:
            batch_X = batch_X.to(device)
            pred_mean, _ = model(batch_X)
            all_preds.append(pred_mean.cpu().numpy())
            all_targets.append(batch_y.numpy())
            all_mask.append(batch_mask.numpy())
    val_preds = np.concatenate(all_preds, axis=0)
    val_targets = np.concatenate(all_targets, axis=0)
    val_mask = np.concatenate(all_mask, axis=0).astype(bool)

    rae_per_task = []
    for t in range(num_targets):
        mask_t = val_mask[:, t]
        if mask_t.sum() == 0:
            continue
        y_true = val_targets[mask_t, t]
        y_pred = val_preds[mask_t, t]
        mae = mean_absolute_error(y_true, y_pred)
        denom = np.mean(np.abs(y_true - np.mean(y_true)))
        if denom == 0:
            rae = 1.0
        else:
            rae = mae / denom
        rae_per_task.append(rae)
    val_ma_rae = np.mean(rae_per_task) if len(rae_per_task) > 0 else 1.0

    scheduler.step()
    print(
        f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val MA-RAE: {val_ma_rae:.4f}"
    )

    if val_ma_rae < best_val_metric:
        best_val_metric = val_ma_rae
        best_epoch = epoch
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(
                f"Early stopping at epoch {epoch}. Best epoch {best_epoch} with MA-RAE {best_val_metric:.4f}"
            )
            break

# Restore best model
model.load_state_dict(best_model_state)
model.eval()

# Final validation metric
all_preds = []
with torch.no_grad():
    for batch_X, _, _ in val_loader:
        batch_X = batch_X.to(device)
        pred_mean, _ = model(batch_X)
        all_preds.append(pred_mean.cpu().numpy())
val_preds = np.concatenate(all_preds, axis=0)
rae_per_task = []
for t in range(num_targets):
    mask_t = val_mask[:, t]
    if mask_t.sum() == 0:
        continue
    y_true = val_targets[mask_t, t]
    y_pred = val_preds[mask_t, t]
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.mean(np.abs(y_true - np.mean(y_true)))
    if denom == 0:
        rae = 1.0
    else:
        rae = mae / denom
    rae_per_task.append(rae)
final_val_ma_rae = np.mean(rae_per_task)

# Test inference
test_preds = []
test_loader = DataLoader(
    TensorDataset(X_test_t), batch_size=batch_size, shuffle=False, num_workers=2
)
with torch.no_grad():
    for (batch_X,) in test_loader:
        batch_X = batch_X.to(device)
        pred = model(batch_X)
        test_preds.append(pred.cpu().numpy())
test_preds = np.concatenate(test_preds, axis=0)

# Submission
submission_df = pd.DataFrame(
    {
        "Molecule Name": test_molecule_names,
        target_cols[0]: test_preds[:, 0],
        target_cols[1]: test_preds[:, 1],
        target_cols[2]: test_preds[:, 2],
        target_cols[3]: test_preds[:, 3],
        target_cols[4]: test_preds[:, 4],
        target_cols[5]: test_preds[:, 5],
        target_cols[6]: test_preds[:, 6],
        target_cols[7]: test_preds[:, 7],
        target_cols[8]: test_preds[:, 8],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission_54ad4c05e6ae4109b49a947e42cc0149.csv", index=False)
print("Submission saved to ./submission/submission_54ad4c05e6ae4109b49a947e42cc0149.csv")

score = final_val_ma_rae
print(f"Final Validation Score: {score}")
