import pandas as pd
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors, Lipinski
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdDistGeom
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import lightgbm as lgb
from sklearn.linear_model import Ridge
import pickle
import json
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
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

# ============================================================
# 2. Advanced Molecular Feature Engineering
# ============================================================


def compute_3d_descriptors(mol):
    """Compute 3D molecular descriptors with embedded conformer generation"""
    descriptors = {}
    try:
        mol_with_H = Chem.AddHs(mol)
        params = rdDistGeom.ETMDG()
        params.randomSeed = 42
        params.numThreads = 1
        status = rdDistGeom.EmbedMolecule(mol_with_H, params)
        if status == 0:
            # Optimize conformer
            ff = AllChem.UFFGetMoleculeForceField(mol_with_H)
            if ff is not None:
                ff.Minimize()
            # 3D descriptors
            conf = mol_with_H.GetConformer()
            # Partial charges
            AllChem.ComputeGasteigerCharges(mol_with_H)
            charges = [
                float(mol_with_H.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
                for i in range(mol_with_H.GetNumAtoms())
            ]
            if charges:
                descriptors["max_partial_charge"] = max(charges)
                descriptors["min_partial_charge"] = min(charges)
                descriptors["abs_charge_sum"] = sum(abs(c) for c in charges)
                descriptors["charge_std"] = np.std(charges)

            # Geometric features
            # Volume approximation
            try:
                vol = AllChem.ComputeMolVolume(mol_with_H)
                descriptors["mol_vol"] = vol
            except:
                descriptors["mol_vol"] = Descriptors.MolWt(mol) / 1.0  # fallback

            # Principal moments of inertia
            try:
                mi = rdMolTransforms.ComputePrincipalMomentsOfInertia(conf)
                descriptors["pmom_inertia_1"] = mi[0]
                descriptors["pmom_inertia_2"] = mi[1]
                descriptors["pmom_inertia_3"] = mi[2]
                # Asphericity
                if mi[0] > 0:
                    descriptors["asphericity"] = mi[2] / mi[0]
                else:
                    descriptors["asphericity"] = 1.0
            except:
                pass

            # Radius of gyration
            try:
                coords = [
                    conf.GetAtomPosition(i) for i in range(mol_with_H.GetNumAtoms())
                ]
                center = np.mean([[c.x, c.y, c.z] for c in coords], axis=0)
                rg = np.sqrt(
                    np.mean(
                        [
                            (c.x - center[0]) ** 2
                            + (c.y - center[1]) ** 2
                            + (c.z - center[2]) ** 2
                            for c in coords
                        ]
                    )
                )
                descriptors["radius_gyration"] = rg
            except:
                pass
    except:
        pass
    return descriptors


def compute_advanced_fingerprints(mol):
    """Compute multiple fingerprint types as feature vectors"""
    fps = {}

    # Morgan fingerprints with varying radii
    for radius in [1, 2, 3]:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048)
        arr = np.array(fp)
        # Compress to meaningful features: substructure counts, bit density
        fps[f"morgan_r{radius}_density"] = np.mean(arr)
        fps[f"morgan_r{radius}_nonzero"] = np.sum(arr)
        # Topological features from fingerprint
        on_bits = np.where(arr == 1)[0]
        if len(on_bits) > 0:
            fps[f"morgan_r{radius}_entropy"] = -np.sum(arr * np.log(arr + 1e-10))
            fps[f"morgan_r{radius}_spread"] = np.std(on_bits)
        else:
            fps[f"morgan_r{radius}_entropy"] = 0
            fps[f"morgan_r{radius}_spread"] = 0

    # MACCS keys (166 bits)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.array(maccs)
    fps["maccs_density"] = np.mean(maccs_arr)
    fps["maccs_nsubstruct"] = np.sum(maccs_arr)

    # EState indices - use rdMolDescriptors properly
    try:
        estate = rdMolDescriptors.GetEStateIndices(mol)
    except AttributeError:
        # Fallback: some RDKit versions use a different method name
        try:
            estate = list(rdMolDescriptors.CalcEState(mol))
        except AttributeError:
            estate = []
    if len(estate) > 0:
        fps["estate_sum"] = np.sum(estate)
        fps["estate_mean"] = np.mean(estate)
        fps["estate_std"] = np.std(estate)
        fps["estate_max"] = np.max(estate)
        fps["estate_min"] = np.min(estate)

    return fps


def compute_pharmacophore_features(mol):
    """Compute pharmacophoric features"""
    feats = {}
    from rdkit.Chem import Descriptors as Desc

    # Ring features
    ring_info = mol.GetRingInfo()
    feats["num_rings"] = ring_info.NumRings()
    # Use Descriptors for aromatic ring count (more reliable)
    feats["num_aromatic_rings"] = Desc.NumAromaticRings(mol)
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    if ring_sizes:
        feats["mean_ring_size"] = np.mean(ring_sizes)
        feats["max_ring_size"] = max(ring_sizes)
        feats["ring_size_std"] = np.std(ring_sizes)
    else:
        feats["mean_ring_size"] = 0
        feats["max_ring_size"] = 0
        feats["ring_size_std"] = 0

    # Functional group counts
    feats["num_H_donors"] = Desc.NumHDonors(mol)
    feats["num_H_acceptors"] = Desc.NumHAcceptors(mol)
    feats["num_rotatable"] = Desc.NumRotatableBonds(mol)
    feats["num_heavy_atoms"] = mol.GetNumHeavyAtoms()

    # Aromatic features
    aromatic_atoms = [a.GetIsAromatic() for a in mol.GetAtoms()]
    feats["aromatic_ratio"] = sum(aromatic_atoms) / max(len(aromatic_atoms), 1)

    # Polar surface area components
    ps = Desc.TPSA(mol)
    feats["tpsa"] = ps

    # Labute ASA (approximate surface area contributions)
    try:
        l_asa = rdMolDescriptors.CalcLabuteASA(mol)
        feats["labute_ASA"] = l_asa
    except:
        feats["labute_ASA"] = Desc.MolWt(mol)  # fallback

    # Crippen contributions (logP and MR)
    try:
        logp_contribs = rdMolDescriptors._CalcCrippenContributions(mol)
        if logp_contribs:
            feats["crippen_logP_sum"] = logp_contribs[0]
            feats["crippen_logP_std"] = np.std(logp_contribs[0])
            feats["crippen_MR_sum"] = logp_contribs[1]
            feats["crippen_MR_std"] = np.std(logp_contribs[1])
    except:
        pass

    return feats


def smiles_to_features(smiles_list):
    """Transform SMILES to comprehensive feature set"""
    features = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            features.append({})
            continue

        feat_dict = {}

        # Basic physicochemical descriptors
        feat_dict["MolWt"] = Descriptors.MolWt(mol)
        feat_dict["LogP"] = Descriptors.MolLogP(mol)
        feat_dict["TPSA"] = Descriptors.TPSA(mol)
        feat_dict["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)
        feat_dict["NumHDonors"] = Descriptors.NumHDonors(mol)
        feat_dict["NumRotatableBonds"] = Descriptors.NumRotatableBonds(mol)
        feat_dict["NumAromaticRings"] = Descriptors.NumAromaticRings(mol)
        feat_dict["NumAliphaticRings"] = Descriptors.NumAliphaticRings(mol)
        feat_dict["NumSaturatedRings"] = Descriptors.NumSaturatedRings(mol)
        feat_dict["NumHeteroatoms"] = Descriptors.NumHeteroatoms(mol)
        feat_dict["FractionCSP3"] = Descriptors.FractionCSP3(mol)
        feat_dict["HallKierAlpha"] = Descriptors.HallKierAlpha(mol)
        feat_dict["Kappa1"] = Descriptors.Kappa1(mol)
        feat_dict["Kappa2"] = Descriptors.Kappa2(mol)
        feat_dict["Kappa3"] = Descriptors.Kappa3(mol)
        feat_dict["Chi0"] = Descriptors.Chi0(mol)
        feat_dict["Chi1"] = Descriptors.Chi1(mol)
        feat_dict["FpDensityMorgan1"] = Descriptors.FpDensityMorgan1(mol)
        feat_dict["FpDensityMorgan2"] = Descriptors.FpDensityMorgan2(mol)
        feat_dict["FpDensityMorgan3"] = Descriptors.FpDensityMorgan3(mol)
        # BCUT2D - fallback to available method or safe default
        try:
            bcut_vals = rdMolDescriptors.CalcBCUT2D(mol)
            feat_dict["BCUT2D_MWHI"] = bcut_vals[0] if hasattr(bcut_vals, "__iter__") and len(bcut_vals) > 0 else 0
            feat_dict["BCUT2D_MWLOW"] = bcut_vals[-1] if hasattr(bcut_vals, "__iter__") and len(bcut_vals) > 0 else 0
        except (AttributeError, Exception):
            feat_dict["BCUT2D_MWHI"] = 0.0
            feat_dict["BCUT2D_MWLOW"] = 0.0

        # Additional RDKit descriptors for richer representation
        feat_dict["ExactMolWt"] = Descriptors.ExactMolWt(mol)
        feat_dict["NumValenceElectrons"] = Descriptors.NumValenceElectrons(mol)
        feat_dict["NumSaturatedRings"] = Descriptors.NumSaturatedRings(mol)
        feat_dict["NumAliphaticHeterocycles"] = Descriptors.NumAliphaticHeterocycles(mol)
        feat_dict["NumAromaticHeterocycles"] = Descriptors.NumAromaticHeterocycles(mol)
        feat_dict["NumAliphaticCarbocycles"] = Descriptors.NumAliphaticCarbocycles(mol)
        feat_dict["NumAromaticCarbocycles"] = Descriptors.NumAromaticCarbocycles(mol)
        feat_dict["PEOE_VSA1"] = Descriptors.PEOE_VSA1(mol)
        feat_dict["PEOE_VSA2"] = Descriptors.PEOE_VSA2(mol)
        feat_dict["PEOE_VSA3"] = Descriptors.PEOE_VSA3(mol)
        feat_dict["PEOE_VSA4"] = Descriptors.PEOE_VSA4(mol)
        feat_dict["PEOE_VSA5"] = Descriptors.PEOE_VSA5(mol)
        feat_dict["PEOE_VSA6"] = Descriptors.PEOE_VSA6(mol)
        feat_dict["SlogP_VSA1"] = Descriptors.SlogP_VSA1(mol)
        feat_dict["SlogP_VSA2"] = Descriptors.SlogP_VSA2(mol)
        feat_dict["SlogP_VSA3"] = Descriptors.SlogP_VSA3(mol)
        feat_dict["SlogP_VSA4"] = Descriptors.SlogP_VSA4(mol)
        feat_dict["SlogP_VSA5"] = Descriptors.SlogP_VSA5(mol)
        feat_dict["SlogP_VSA6"] = Descriptors.SlogP_VSA6(mol)
        feat_dict["SlogP_VSA7"] = Descriptors.SlogP_VSA7(mol)
        feat_dict["SMR_VSA1"] = Descriptors.SMR_VSA1(mol)
        feat_dict["SMR_VSA2"] = Descriptors.SMR_VSA2(mol)
        feat_dict["SMR_VSA3"] = Descriptors.SMR_VSA3(mol)
        feat_dict["SMR_VSA4"] = Descriptors.SMR_VSA4(mol)
        feat_dict["SMR_VSA5"] = Descriptors.SMR_VSA5(mol)
        feat_dict["SMR_VSA6"] = Descriptors.SMR_VSA6(mol)
        feat_dict["SMR_VSA7"] = Descriptors.SMR_VSA7(mol)
        feat_dict["Ipc"] = Descriptors.Ipc(mol)
        feat_dict["BertzCT"] = Descriptors.BertzCT(mol)
        feat_dict["Chi0v"] = Descriptors.Chi0v(mol)
        feat_dict["Chi1v"] = Descriptors.Chi1v(mol)
        feat_dict["Chi0n"] = Descriptors.Chi0n(mol)
        feat_dict["Chi1n"] = Descriptors.Chi1n(mol)
        feat_dict["Chi2n"] = Descriptors.Chi2n(mol)
        feat_dict["Chi3n"] = Descriptors.Chi3n(mol)
        feat_dict["Chi4n"] = Descriptors.Chi4n(mol)

        # Advanced fingerprints
        feat_dict.update(compute_advanced_fingerprints(mol))

        # Pharmacophore features
        feat_dict.update(compute_pharmacophore_features(mol))

        # 3D descriptors (with fallback to 2D approximations)
        try:
            feat_dict.update(compute_3d_descriptors(mol))
        except:
            feat_dict["mol_vol"] = Descriptors.MolWt(mol) / 1.0
            # Use the existing num_heavy_atoms key with underscore
            heavy_atoms = feat_dict.get("num_heavy_atoms", 20)
            feat_dict["radius_gyration"] = 0.5 * (heavy_atoms ** 0.33)

        # SMILES length as simple bias feature
        feat_dict["smiles_len"] = len(sm)
        feat_dict["branch_count"] = sm.count("(") + sm.count(")")

        features.append(feat_dict)

    return pd.DataFrame(features)


# Generate all features
print("Computing molecular features for training data...")
train_features = smiles_to_features(train_df["SMILES"].values)
print(f"Generated {len(train_features.columns)} features")

print("Computing molecular features for test data...")
test_features = smiles_to_features(test_df["SMILES"].values)
print(f"Generated {len(test_features.columns)} features")

# ============================================================
# 3. Data Cleaning and Imputation
# ============================================================

# First split the data, then impute on train only
np.random.seed(42)
# Train-validation split
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx_list = []
val_idx_list = []

for train_idx_fold, val_idx_fold in kf.split(train_features):
    train_idx_list.append(train_idx_fold)
    val_idx_list.append(val_idx_fold)

# Use first fold for validation
val_fold = 0
train_idx = train_idx_list[val_fold]
val_idx = val_idx_list[val_fold]

# Split features into train and val
train_features_split = train_features.iloc[train_idx].reset_index(drop=True)
val_features_split = train_features.iloc[val_idx].reset_index(drop=True)

# Fit imputer on training data only
feature_cols = train_features.columns
feature_imputer = SimpleImputer(strategy="median")
train_features_imputed = feature_imputer.fit_transform(train_features_split)
train_features_imputed = pd.DataFrame(train_features_imputed, columns=feature_cols)

val_features_imputed = feature_imputer.transform(val_features_split)
val_features_imputed = pd.DataFrame(val_features_imputed, columns=feature_cols)

test_features_imputed = feature_imputer.transform(test_features)
test_features_imputed = pd.DataFrame(test_features_imputed, columns=feature_cols)

# Handle infinite values
train_features_imputed = train_features_imputed.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)
val_features_imputed = val_features_imputed.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)
test_features_imputed = test_features_imputed.replace([np.inf, -np.inf], np.nan).fillna(
    0
)

# Now assign to X_train, X_val, X_test as expected by downstream code
X_train = train_features_imputed
X_val = val_features_imputed
X_test = test_features_imputed

# ============================================================
# 4. Target Processing - log transform for non-log targets
# ============================================================

# For each target, store whether we log-transformed
log_targets = {}
for col in target_cols:
    # Check if target needs log transform (based on task description)
    if col in ["Caco-2 Permeability Papp A>B", "Caco-2 Permeability Efflux"]:
        # Already in log domain per task description
        log_targets[col] = False
    else:
        log_targets[col] = True

# ============================================================
# 5. Target Split using same indices
# ============================================================

y_train = train_df[target_cols].iloc[train_idx].reset_index(drop=True)
y_val = train_df[target_cols].iloc[val_idx].reset_index(drop=True)

# ============================================================
# 6. Feature Scaling - Fit on training only
# ============================================================

# Use StandardScaler (robust, avoids PowerTransformer failure with constant features)
scaler = StandardScaler()
# Clip extreme values before transform
train_clipped = np.clip(X_train.values, -1e3, 1e3)
val_clipped = np.clip(X_val.values, -1e3, 1e3)
test_clipped = np.clip(X_test.values, -1e3, 1e3)

X_train_scaled = scaler.fit_transform(train_clipped)
X_val_scaled = scaler.transform(val_clipped)
X_test_scaled = scaler.transform(test_clipped)

# ============================================================
# 7. Target Scaling
# ============================================================

target_scalers = {}
y_train_scaled = pd.DataFrame(index=y_train.index)
y_val_scaled = pd.DataFrame(index=y_val.index)

for col in target_cols:
    ts = PowerTransformer(method="yeo-johnson", standardize=True)
    col_data = y_train[col].values.reshape(-1, 1)
    # Handle NaN - impute with median for scaling only
    col_median = np.nanmedian(col_data)
    col_data_imputed = np.where(np.isnan(col_data), col_median, col_data)

    y_train_scaled[col] = ts.fit_transform(col_data_imputed).ravel()

    # Transform validation
    val_data = y_val[col].values.reshape(-1, 1)
    val_data_imputed = np.where(np.isnan(val_data), col_median, val_data)
    y_val_scaled[col] = ts.transform(val_data_imputed).ravel()

    target_scalers[col] = ts

# ============================================================
# 8. Feature Selection - Select top features per target
# ============================================================

# Multi-task feature selection: find jointly informative features
all_selected_features = set()

for col in target_cols:
    # Use only non-NaN samples for this target
    non_nan_mask = y_train[col].notna().values
    if non_nan_mask.sum() < 50:  # Too few samples
        continue

    X_train_sub = X_train_scaled[non_nan_mask]
    y_train_sub = y_train_scaled[col].values[non_nan_mask]

    if len(y_train_sub) > 100:
        selector = SelectKBest(mutual_info_regression, k=min(80, X_train_sub.shape[1]))
        selector.fit(X_train_sub, y_train_sub)
        selected_indices = np.where(
            selector.scores_ > np.percentile(selector.scores_, 70)
        )[0]
        all_selected_features.update(selected_indices.tolist())

# If too few features selected, add all with high variance
if len(all_selected_features) < 20:
    feature_var = np.var(X_train_scaled, axis=0)
    top_var_features = np.argsort(feature_var)[-100:]
    all_selected_features.update(top_var_features.tolist())

selected_features = sorted(list(all_selected_features))
print(f"Selected {len(selected_features)} features for modeling")

# Apply feature selection
X_train_selected = X_train_scaled[:, selected_features]
X_val_selected = X_val_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

# ============================================================
# 9. Save processed data
# ============================================================

# Save as numpy arrays for efficient loading
np.save("./working/X_train.npy", X_train_selected)
np.save("./working/X_val.npy", X_val_selected)
np.save("./working/X_test.npy", X_test_selected)
np.save("./working/y_train.npy", y_train_scaled.values)
np.save("./working/y_val.npy", y_val_scaled.values)

# Save metadata
metadata = {
    "n_features": X_train_selected.shape[1],
    "n_train_samples": X_train_selected.shape[0],
    "n_val_samples": X_val_selected.shape[0],
    "n_test_samples": X_test_selected.shape[0],
    "feature_columns": [f"f{i}" for i in selected_features],
    "target_columns": target_cols,
    "train_indices": train_idx.tolist(),
    "val_indices": val_idx.tolist(),
    "log_targets": {k: bool(v) for k, v in log_targets.items()},
}

with open("./working/metadata.json", "w") as f:
    json.dump(metadata, f)

# Also save the original targets with indices for modeling
train_df.iloc[train_idx][target_cols].to_pickle("./working/y_train_original.pkl")
train_df.iloc[val_idx][target_cols].to_pickle("./working/y_val_original.pkl")

# Save target scalers
with open("./working/target_scalers.pkl", "wb") as f:
    pickle.dump(target_scalers, f)

print(
    f"Data processing complete. Train: {X_train_selected.shape}, Val: {X_val_selected.shape}, Test: {X_test_selected.shape}"
)

# ============================================================
# 10. Model Design and Training - Multi-Task Neural Network
# ============================================================

# Load preprocessed data
X_train = np.load("./working/X_train.npy")
X_val = np.load("./working/X_val.npy")
X_test = np.load("./working/X_test.npy")
y_train = np.load("./working/y_train.npy")
y_val = np.load("./working/y_val.npy")

# Load metadata
with open("./working/metadata.json", "r") as f:
    metadata = json.load(f)

target_cols = metadata["target_columns"]
n_targets = len(target_cols)
n_features = X_train.shape[1]

# Load original targets for proper metric computation
y_train_original = pd.read_pickle("./working/y_train_original.pkl")
y_val_original = pd.read_pickle("./working/y_val_original.pkl")

# Load target scalers (PowerTransformers) for inverse transform
with open("./working/target_scalers.pkl", "rb") as f:
    target_scalers = pickle.load(f)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 11. Multi-Task MLP Model Definition
# ============================================================

class MultiTaskMLP(nn.Module):
    """Multi-task neural network predicting all 9 targets simultaneously"""
    def __init__(self, input_dim, output_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class MoleculeDataset(Dataset):
    """Dataset with NaN masking for multi-task learning"""
    def __init__(self, X, y, nan_mask=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        if nan_mask is None:
            self.nan_mask = ~torch.isnan(self.y)
        else:
            self.nan_mask = torch.BoolTensor(nan_mask)
        # Impute NaN targets with 0 for forward pass (they will be masked in loss)
        self.y = torch.nan_to_num(self.y, nan=0.0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.nan_mask[idx]


def masked_mse_loss(pred, target, mask):
    """MSE loss ignoring NaN targets via masking"""
    diff = (pred - target) ** 2
    diff = diff * mask.float()
    loss = diff.sum() / (mask.sum() + 1e-8)
    return loss


def compute_macro_rae(y_true_original, y_pred_original, target_cols):
    """Compute macro-averaged relative absolute error"""
    scores = []
    for j, col in enumerate(target_cols):
        y_t = y_true_original[col].values
        y_p = y_pred_original[:, j]
        mask = ~np.isnan(y_t)
        if mask.sum() == 0:
            continue
        y_t_masked = y_t[mask]
        y_p_masked = y_p[mask]
        numerator = np.sum(np.abs(y_t_masked - y_p_masked))
        denominator = np.sum(np.abs(y_t_masked - np.mean(y_t_masked)))
        if denominator > 1e-10:
            scores.append(numerator / denominator)
        else:
            scores.append(1.0)
    return np.mean(scores)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch, mask_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = masked_mse_loss(pred, y_batch, mask_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)


def predict_model(model, X, device, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            pred_batch = model(X_batch).cpu().numpy()
            preds.append(pred_batch)
    return np.vstack(preds)


# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# 12. 5-Fold Cross-Validation Training
# ============================================================
print("\nTraining multi-task MLP with 5-fold CV...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store OOF predictions (scaled) for ensemble averaging
oof_preds_scaled = np.zeros((X_train.shape[0], n_targets))
fold_models = []

for fold, (train_idx_fold, val_fold_idx) in enumerate(kf.split(X_train)):
    print(f"\n  Fold {fold+1}/5...")
    X_tr_fold = X_train[train_idx_fold]
    y_tr_fold = y_train[train_idx_fold]
    X_va_fold = X_train[val_fold_idx]
    y_va_fold = y_train[val_fold_idx]

    # Create datasets and dataloaders
    train_dataset = MoleculeDataset(X_tr_fold, y_tr_fold)
    val_dataset = MoleculeDataset(X_va_fold, y_va_fold)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Initialize model
    model = MultiTaskMLP(input_dim=n_features, output_dim=n_targets).to(device)

    # Loss, optimizer, scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 30
    patience_counter = 0

    for epoch in range(200):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, mask_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                pred = model(X_batch)
                loss = masked_mse_loss(pred, y_batch, mask_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.6f}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: train loss = {train_loss:.6f}, val loss = {val_loss:.6f}")

    # Restore best model and keep a copy for ensemble (on CPU)
    model.load_state_dict(best_model_state)
    fold_models.append(model.cpu())  # CPU copy for later ensemble

    # Move the model back to the device for OOF predictions
    model = model.to(device)

    # OOF predictions - ensure no NaN in features
    X_va_fold_clean = np.nan_to_num(X_va_fold, nan=0.0)
    oof_preds_scaled[val_fold_idx] = predict_model(model, X_va_fold_clean, device)
    print(f"  Fold {fold+1} done.")

# ============================================================
# 13. Train final model on full training data
# ============================================================
print("\nTraining final model on full training data...")

final_dataset = MoleculeDataset(X_train, y_train)
final_loader = DataLoader(final_dataset, batch_size=64, shuffle=True)

final_model = MultiTaskMLP(input_dim=n_features, output_dim=n_targets).to(device)
optimizer = optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

best_train_loss = float('inf')
best_final_state = None

for epoch in range(200):
    train_loss = train_epoch(final_model, final_loader, optimizer, device)
    scheduler.step()

    if train_loss < best_train_loss:
        best_train_loss = train_loss
        best_final_state = final_model.state_dict().copy()

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}: train loss = {train_loss:.6f}")

# Restore best model
final_model.load_state_dict(best_final_state)
final_model = final_model.cpu()

# ============================================================
# 14. Validation predictions from ensemble of 5 folds
# ============================================================
print("\nGenerating validation predictions via fold ensemble...")

val_preds_scaled = np.zeros((X_val.shape[0], n_targets))
X_val_clean = np.nan_to_num(X_val, nan=0.0)
for fold_model in fold_models:
    fold_model.eval()
    # Move fold model to device for prediction, then back to CPU
    fold_model = fold_model.to(device)
    val_preds_scaled += predict_model(fold_model, X_val_clean, device) / len(fold_models)
    fold_model.cpu()  # Move back to CPU to save memory

# Inverse transform predictions to original scale for metric
val_preds_original = np.full_like(val_preds_scaled, np.nan)
for j, col in enumerate(target_cols):
    scaler = target_scalers[col]
    pred_reshaped = val_preds_scaled[:, j].reshape(-1, 1)
    val_preds_original[:, j] = scaler.inverse_transform(pred_reshaped).ravel()

final_val_score = compute_macro_rae(y_val_original, val_preds_original, target_cols)
print(f"Validation MA-RAE (multi-task MLP ensemble): {final_val_score:.6f}")

# ============================================================
# 15. Test predictions
# ============================================================
print("\nGenerating test predictions...")

test_preds_scaled = np.zeros((X_test.shape[0], n_targets))
X_test_clean = np.nan_to_num(X_test, nan=0.0)
for fold_model in fold_models:
    fold_model.eval()
    # Move fold model to device for prediction, then back to CPU
    fold_model = fold_model.to(device)
    test_preds_scaled += predict_model(fold_model, X_test_clean, device) / len(fold_models)
    fold_model.cpu()  # Move back to CPU to save memory

# Inverse transform
test_preds_original = np.full_like(test_preds_scaled, np.nan)
for j, col in enumerate(target_cols):
    scaler = target_scalers[col]
    pred_reshaped = test_preds_scaled[:, j].reshape(-1, 1)
    test_preds_original[:, j] = scaler.inverse_transform(pred_reshaped).ravel()

# ============================================================
# 16. Create submission file
# ============================================================
print("\nCreating submission file...")
submission = pd.DataFrame({"Molecule Name": test_df["Molecule Name"].values})

for j, col in enumerate(target_cols):
    submission[col] = test_preds_original[:, j]

# Fill any remaining NaN with 0
submission = submission.fillna(0.0)

# Ensure proper column order
column_order = [
    "Molecule Name",
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
submission = submission[column_order]

# Save submission
submission.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")

# ============================================================
# 17. Save models for reproducibility
# ============================================================
print("\nSaving models...")
model_save_path = "./working/multi_task_mlp.pth"
torch.save({
    'final_model_state': best_final_state,
    'fold_models_state': [m.state_dict() for m in fold_models],
    'model_config': {'input_dim': n_features, 'output_dim': n_targets}
}, model_save_path)
print(f"Model saved to {model_save_path}")

# ============================================================
# 18. Final validation metric
# ============================================================
print(f"Final Validation Score: {final_val_score:.6f}")