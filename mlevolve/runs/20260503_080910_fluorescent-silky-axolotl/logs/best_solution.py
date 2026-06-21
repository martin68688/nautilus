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

# Apply log1p transform to non-LogD targets (safe for zero/positive values)
for col in log_transform_cols:
    mask = ~train_df[col].isna()
    if mask.sum() > 0:
        vals = train_df.loc[mask, col].values.astype(float)
        vals = np.maximum(vals, 0.0)
        train_df.loc[mask, col] = np.log1p(vals)

print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")


# Feature extraction function
def canonicalize_smiles(smiles_list):
    """Canonicalize SMILES strings using RDKit."""
    canonical = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical.append(Chem.MolToSmiles(mol))
        else:
            canonical.append(smi)
    return canonical


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


print("Canonicalizing SMILES...")
train_canon = canonicalize_smiles(train_df["SMILES"].values)
test_canon = canonicalize_smiles(test_df["SMILES"].values)

print("Extracting features for training data...")
train_features = get_rdkit_features(train_canon)
print(f"Train features shape: {train_features.shape}")

print("Extracting features for test data...")
test_features = get_rdkit_features(test_canon)
print(f"Test features shape: {test_features.shape}")



# Create data availability pattern for stratified split (per-target missingness pattern)
# Binary pattern for each target, then use GroupKFold
missing_pattern = np.zeros((len(train_df), len(target_cols)), dtype=int)
for i_t, col in enumerate(target_cols):
    missing_pattern[:, i_t] = (~train_df[col].isna()).astype(int)
# Create a single group label from the binary pattern
pattern_strs = ["".join(str(x) for x in row) for row in missing_pattern]
from sklearn.preprocessing import LabelEncoder
pattern_le = LabelEncoder()
groups = pattern_le.fit_transform(pattern_strs)
unique_groups = len(set(groups))
n_splits = min(5, max(2, unique_groups))
print(f"Using {n_splits}-fold cross-validation ({unique_groups} unique missingness patterns)")

gkf = GroupKFold(n_splits=n_splits)
# We'll do proper 5-fold CV with ensemble later; for now just first split for outer loop
# Actual 5-fold CV will be done inside the training loop for each target

# Per-target feature selection using mutual information
from sklearn.feature_selection import SelectKBest, mutual_info_regression

N_FEATURES_PER_TARGET = 500
feature_indices_per_target = {}
train_df_features = train_features  # full 2782 features

for i_t, col in enumerate(target_cols):
    avail_mask = ~train_df[col].isna()
    if avail_mask.sum() < 50:
        # Too few samples, use all features
        feature_indices_per_target[col] = np.arange(train_features.shape[1])
        continue
    y_col = train_df.loc[avail_mask, col].values.astype(float)
    X_col = train_df_features[avail_mask.values]
    # Subsample features for speed: compute MI on all features but maybe slow for 2782
    # Use all features for selection
    selector = SelectKBest(mutual_info_regression, k=min(N_FEATURES_PER_TARGET, train_features.shape[1]))
    selector.fit(X_col, y_col)
    selected = np.where(selector.get_support())[0]
    feature_indices_per_target[col] = selected
    print(f"  {col}: selected {len(selected)} features from {X_col.shape[1]}")

# Standardize features (global scaler for all features)
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Prepare target values (after log1p transform), store originals for inverse transform later
target_values = train_df[target_cols].values.astype(float)

# Store train and test features in dict for LightGBM per-target training
train_feat_dict = {col: train_features_scaled[:, feature_indices_per_target[col]] for col in target_cols}
test_feat_dict = {col: test_features_scaled[:, feature_indices_per_target[col]] for col in target_cols}

import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Train LightGBM models per target with 5-fold CV
print("\nTraining LightGBM models...")
lgb_params = {
    'learning_rate': 0.03,
    'num_leaves': 63,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'metric': 'mae',
    'verbosity': -1,
    'n_jobs': -1,
    'random_state': 42
}

# Store models and out-of-fold predictions for meta-model
lgb_models = {}
oof_predictions = np.zeros((len(train_df), len(target_cols)))
test_predictions_ensemble = np.zeros((len(test_df), len(target_cols)))

# Per-target 5-fold CV
outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)

for i_t, col in enumerate(target_cols):
    print(f"\nTraining LightGBM for {col}...")
    avail_mask = ~train_df[col].isna()
    avail_indices = np.where(avail_mask)[0]
    n_avail = len(avail_indices)

    if n_avail < 20:
        # Too few samples, use mean
        mean_val = train_df.loc[avail_mask, col].mean()
        oof_predictions[avail_indices, i_t] = mean_val
        test_predictions_ensemble[:, i_t] = mean_val
        lgb_models[col] = None
        continue

    X_avail = train_feat_dict[col][avail_indices]
    y_avail = target_values[avail_indices, i_t]

    fold_preds_oof = np.zeros(n_avail)
    fold_models = []

    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_inner_idx, val_inner_idx in inner_kf.split(np.arange(n_avail)):
        X_tr = X_avail[train_inner_idx]
        y_tr = y_avail[train_inner_idx]
        X_val = X_avail[val_inner_idx]
        y_val = y_avail[val_inner_idx]

        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        fold_preds_oof[val_inner_idx] = model.predict(X_val)
        fold_models.append(model)

    oof_predictions[avail_indices, i_t] = fold_preds_oof

    # Retrain on full avail data
    final_model = lgb.train(
        lgb_params,
        lgb.Dataset(X_avail, y_avail),
        num_boost_round=model.best_iteration if hasattr(model, 'best_iteration') else 500,
        callbacks=[lgb.log_evaluation(0)]
    )
    lgb_models[col] = final_model

    # Predict test
    test_preds = final_model.predict(test_feat_dict[col])
    test_predictions_ensemble[:, i_t] = test_preds

    print(f"  {col}: {n_avail} samples, best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")

# Train linear regression meta-model on out-of-fold predictions
print("\nTraining linear regression meta-model...")
# For meta-model, we need to handle missing values in oof_predictions
# Use only rows where all targets have predictions (intersection of avail masks)
# Or simpler: fill missing with 0 for meta-model input (since standard scaler helps)
oof_meta_X = np.nan_to_num(oof_predictions, nan=0.0)
# Targets for meta-model: original target values (post log1p)
meta_y = np.nan_to_num(target_values, nan=0.0)

meta_model = LinearRegression()
meta_model.fit(oof_meta_X, meta_y)
meta_coef = meta_model.coef_
print(f" Meta-model coefficients shape: {meta_coef.shape}")

# Apply meta-model to test predictions
test_meta_X = np.nan_to_num(test_predictions_ensemble, nan=0.0)
test_predictions = meta_model.predict(test_meta_X)

# Inverse transform for submission (log1p -> expm1)
for i_col, col in enumerate(target_cols):
    if col != "LogD":
        # Ensure predictions are non-negative before expm1
        test_predictions[:, i_col] = np.maximum(test_predictions[:, i_col], 0.0)
        test_predictions[:, i_col] = np.expm1(test_predictions[:, i_col])

# Clip predictions to physically plausible ranges
test_predictions[:, 0] = np.clip(test_predictions[:, 0], -2.0, 6.0)
test_predictions[:, 1] = np.clip(test_predictions[:, 1], 0.0, 500.0)
test_predictions[:, 2] = np.clip(test_predictions[:, 2], 0.0, 3000.0)
test_predictions[:, 3] = np.clip(test_predictions[:, 3], 0.0, 12000.0)
test_predictions[:, 4] = np.clip(test_predictions[:, 4], 0.0, 60.0)
test_predictions[:, 5] = np.clip(test_predictions[:, 5], 0.0, 120.0)
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

# Compute validation MA-RAE score from out-of-fold predictions
from sklearn.metrics import mean_absolute_error

def compute_ma_rae(y_true, y_pred, eps=1e-6):
    """Compute Macro-Averaged Relative Absolute Error.
    RAE = |pred - true| / (|true| + eps) for each sample, then MAE per task, then macro average.
    """
    n_tasks = y_true.shape[1]
    task_maes = []
    for t in range(n_tasks):
        mask = ~np.isnan(y_true[:, t])
        if mask.sum() == 0:
            continue
        yt = y_true[mask, t]
        yp = y_pred[mask, t]
        abs_error = np.abs(yp - yt)
        rel_denom = np.abs(yt) + eps
        rae_per_sample = abs_error / rel_denom
        task_mae = np.mean(rae_per_sample)
        task_maes.append(task_mae)
    ma_rae = np.mean(task_maes)
    return ma_rae

# Compute validation score using out-of-fold predictions on non-NaN targets
val_ma_rae = compute_ma_rae(target_values, oof_predictions)
best_val_metric = val_ma_rae

print(f"Final Validation Score: {best_val_metric:.6f}")