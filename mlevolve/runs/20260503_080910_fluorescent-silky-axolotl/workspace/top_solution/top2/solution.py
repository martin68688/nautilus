#!/usr/bin/env python3
"""
Merged solution for OpenADMET ExpansionRx Challenge
Uses RDKit fingerprints + MLP with masked loss and proper log-transform handling
"""

import numpy as np
import pandas as pd
import warnings
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
import logging
logging.getLogger("rdkit").setLevel(logging.ERROR)

# =============================================================================
# DATA PROCESSING & FEATURE ENGINEERING
# =============================================================================


def compute_rdkit_features(smiles_list, n_bits=2048):
    """Compute comprehensive RDKit features for a list of SMILES"""
    features = []
    valid_indices = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(None)
            continue

        try:
            # Morgan fingerprints (ECFP-like)
            morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits)
            morgan_arr = np.array(morgan)

            # MACCS keys
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_arr = np.array(maccs)

            # Physicochemical descriptors
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            ring_count = rdMolDescriptors.CalcNumRings(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            sp3_fraction = rdMolDescriptors.CalcFractionCSP3(mol)
            num_heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
            heavy_atom_count = mol.GetNumHeavyAtoms()
            num_radical_electrons = Descriptors.NumRadicalElectrons(mol)
            valence_electrons = Descriptors.NumValenceElectrons(mol)

            # Use try-except for partial charges as they may not exist in all RDKit versions
            try:
                max_partial_charge = Chem.rdMolDescriptors.CalcMaxPartialCharge(mol)
            except (AttributeError, RuntimeError):
                max_partial_charge = 0.0
            try:
                min_partial_charge = Chem.rdMolDescriptors.CalcMinPartialCharge(mol)
            except (AttributeError, RuntimeError):
                min_partial_charge = 0.0

            num_acidic_groups = hbd
            num_basic_groups = hba

            physchem = np.array(
                [
                    mw,
                    logp,
                    tpsa,
                    hbd,
                    hba,
                    rotatable_bonds,
                    ring_count,
                    aromatic_rings,
                    sp3_fraction,
                    num_heteroatoms,
                    heavy_atom_count,
                    num_radical_electrons,
                    valence_electrons,
                    max_partial_charge,
                    min_partial_charge,
                    num_acidic_groups,
                    num_basic_groups,
                ]
            )
            physchem = np.clip(physchem, -50, 2000)
            physchem = physchem / (np.abs(physchem).max() + 1e-8)

            # Bemis-Murcko scaffold fingerprint
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is not None:
                scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(
                    scaffold, 2, nBits=256
                )
                scaffold_arr = np.array(scaffold_fp)
            else:
                scaffold_arr = np.zeros(256)

            # Combine all features
            combined = np.concatenate([morgan_arr, maccs_arr, physchem, scaffold_arr])
            features.append(combined)
            valid_indices.append(idx)

        except Exception as e:
            features.append(None)
            continue

    return features, valid_indices


TARGET_COLUMNS = [
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

LOG_TRANSFORM_COLS = [col for col in TARGET_COLUMNS if col != "LogD"]


def load_and_process_data():
    """Load, split, and featurize the data"""
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")

    train_smiles_original = list(train_df["SMILES"].values)
    train_smiles = train_smiles_original
    train_targets = train_df[TARGET_COLUMNS].copy()

    # Compute features for all training molecules first
    train_features_list, train_valid = compute_rdkit_features(train_smiles)
    train_features = [f for f in train_features_list if f is not None]
    train_features = np.array(train_features)
    # train_valid contains indices of valid molecules in original order

    # Filter targets to match only valid molecules
    train_targets = train_targets.iloc[train_valid].reset_index(drop=True)
    train_smiles = [train_smiles[i] for i in train_valid]

    # Random split on valid molecules using direct array indexing
    np.random.seed(42)
    n_valid = len(train_features)
    if n_valid == 0:
        print("FATAL: No valid molecules after feature computation. Using fallback features.")
        # Fallback: compute simple atom-count features only (no partial charges)
        fallback_features = []
        for smi in train_smiles_original:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fallback_features.append(np.zeros(10))
            else:
                fb = np.array([
                    mol.GetNumAtoms(),
                    mol.GetNumHeavyAtoms(),
                    Descriptors.ExactMolWt(mol),
                    Descriptors.MolLogP(mol),
                    rdMolDescriptors.CalcTPSA(mol),
                    rdMolDescriptors.CalcNumHBD(mol),
                    rdMolDescriptors.CalcNumHBA(mol),
                    rdMolDescriptors.CalcNumRotatableBonds(mol),
                    rdMolDescriptors.CalcNumRings(mol),
                    rdMolDescriptors.CalcNumAromaticRings(mol),
                ], dtype=np.float32)
                fb = np.nan_to_num(fb, nan=0.0, posinf=2000.0, neginf=-2000.0)
                fb = fb / (np.abs(fb).max() + 1e-8)
                fallback_features.append(fb)
        train_features = np.array(fallback_features)
        n_valid = len(train_features)

    indices = np.arange(n_valid)
    np.random.shuffle(indices)
    n_val = max(1, n_valid // 5)
    # Ensure at least 1 training sample remains
    if n_val >= n_valid:
        n_val = n_valid // 2
        if n_val < 1:
            n_val = 1 if n_valid > 0 else 0
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = train_features[train_idx]
    X_val = train_features[val_idx]
    y_train = train_targets.iloc[train_idx].reset_index(drop=True)
    y_val = train_targets.iloc[val_idx].reset_index(drop=True)
    y_train_original = y_train.copy()
    y_val_original = y_val.copy()

    # Compute test features
    test_features_list, _ = compute_rdkit_features(list(test_df["SMILES"].values))
    test_features_arr = [f for f in test_features_list if f is not None]
    if len(test_features_arr) == 0:
        # Fallback features for test
        test_fallback = []
        for smi in list(test_df["SMILES"].values):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                test_fallback.append(np.zeros(10))
            else:
                fb = np.array([
                    mol.GetNumAtoms(),
                    mol.GetNumHeavyAtoms(),
                    Descriptors.ExactMolWt(mol),
                    Descriptors.MolLogP(mol),
                    rdMolDescriptors.CalcTPSA(mol),
                    rdMolDescriptors.CalcNumHBD(mol),
                    rdMolDescriptors.CalcNumHBA(mol),
                    rdMolDescriptors.CalcNumRotatableBonds(mol),
                    rdMolDescriptors.CalcNumRings(mol),
                    rdMolDescriptors.CalcNumAromaticRings(mol),
                ], dtype=np.float32)
                fb = np.nan_to_num(fb, nan=0.0, posinf=2000.0, neginf=-2000.0)
                fb = fb / (np.abs(fb).max() + 1e-8)
                test_fallback.append(fb)
        test_features = np.array(test_fallback)
    else:
        test_features = np.array(test_features_arr)

    # Impute missing target values FIRST before log transform - fit ONLY on training data
    y_train_imputed = y_train.copy()
    y_val_imputed = y_val.copy()

    # Safety check: ensure we have training data to fit imputer
    if len(y_train) > 0:
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(y_train[TARGET_COLUMNS])
        y_train_imputed[TARGET_COLUMNS] = imputer.transform(y_train[TARGET_COLUMNS])
        y_val_imputed[TARGET_COLUMNS] = imputer.transform(y_val[TARGET_COLUMNS])
    else:
        # Use all available data for imputer fitting (pre-split)
        all_targets = train_df[TARGET_COLUMNS].copy()
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(all_targets)
        y_train_imputed[TARGET_COLUMNS] = imputer.transform(y_train[TARGET_COLUMNS])
        y_val_imputed[TARGET_COLUMNS] = imputer.transform(y_val[TARGET_COLUMNS])

    # Apply log10 transform to non-LogD targets (use the imputed values)
    for col in LOG_TRANSFORM_COLS:
        # Compute shift from training data to avoid data leakage
        train_vals = y_train_imputed[col]
        if len(train_vals) > 0:
            min_train = train_vals.min()
            shift = max(0, -min_train + 0.001)
        else:
            shift = 0.0

        # Transform training (imputed values)
        y_train_imputed[col] = np.log10(y_train_imputed[col] + shift)

        # Transform validation with same shift
        y_val_imputed[col] = np.log10(y_val_imputed[col] + shift)

    # Create masks for valid targets (based on original, not imputed values)
    train_mask = ~y_train.isna().values
    val_mask = ~y_val.isna().values

    test_names = test_df["Molecule Name"].values

    return (
        X_train,
        y_train_imputed.values,
        train_mask,
        X_val,
        y_val_imputed.values,
        val_mask,
        test_features,
        test_names,
        y_train_original,
        y_val_original,
    )


# =============================================================================
# DATASET AND MODEL DEFINITIONS
# =============================================================================


class MoleculeDataset(Dataset):
    """Dataset for molecular features"""

    def __init__(self, features, targets, mask):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.mask = torch.FloatTensor(mask)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.mask[idx]


class MultiTaskMLP(nn.Module):
    """Multi-task MLP for molecular property prediction"""

    def __init__(
        self, input_dim, hidden_dim=512, num_layers=3, dropout=0.3, num_targets=9
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        self.shared = nn.Sequential(*layers)

        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(num_targets)
            ]
        )

    def forward(self, x):
        shared = self.shared(x)
        outputs = [head(shared) for head in self.output_heads]
        return torch.cat(outputs, dim=1)


class MaskedMSELoss(nn.Module):
    """Masked MSE loss that ignores NaN targets"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        loss = ((pred - target) ** 2) * mask
        return loss.sum() / (mask.sum() + 1e-8)


def compute_ma_rae(y_pred_orig, y_true_orig, mask):
    """Compute Macro-Averaged Relative Absolute Error in original space"""
    num_targets = y_pred_orig.shape[1]
    rae_scores = []

    for t in range(num_targets):
        valid_mask = mask[:, t].astype(bool)
        if valid_mask.sum() == 0:
            continue
        y_t_pred = y_pred_orig[valid_mask, t]
        y_t_true = y_true_orig[valid_mask, t]
        abs_errors = np.abs(y_t_pred - y_t_true)
        y_t_true_abs = np.abs(y_t_true) + 1.0
        rae = abs_errors / y_t_true_abs
        rae_scores.append(np.mean(rae))

    if len(rae_scores) == 0:
        return 0.0
    return np.mean(rae_scores)


# =============================================================================
# MAIN TRAINING AND EVALUATION PIPELINE
# =============================================================================


def train_and_evaluate():
    """Main training pipeline"""
    # Load and process data
    (
        train_features,
        train_targets,
        train_mask,
        val_features,
        val_targets,
        val_mask,
        test_features,
        test_names,
        y_train_original,
        y_val_original,
    ) = load_and_process_data()

    # Save original validation targets for MA-RAE computation
    val_targets_original = y_val_original.values

    # Create datasets and dataloaders
    if len(train_features) == 0:
        print("FATAL: train_features is empty. Cannot train.")
        # Return dummy score
        print("Final Validation Score: 100.0000")
        return 100.0
    train_dataset = MoleculeDataset(train_features, train_targets, train_mask)
    val_dataset = MoleculeDataset(val_features, val_targets, val_mask)
    test_dataset = MoleculeDataset(
        test_features,
        np.zeros((len(test_features), 9)),
        np.ones((len(test_features), 9)),
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Initialize model
    input_dim = train_features.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskMLP(
        input_dim=input_dim, hidden_dim=512, num_layers=3, dropout=0.3, num_targets=9
    )
    model = model.to(device)

    criterion = MaskedMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Training loop
    best_val_score = float("inf")
    patience = 20
    patience_counter = 0
    num_epochs = 200

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, targets, mask in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds_log = []
        val_targets_log = []
        val_masks = []
        with torch.no_grad():
            for features, targets, mask in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
                outputs = model(features)
                val_preds_log.append(outputs.cpu().numpy())
                val_targets_log.append(targets.cpu().numpy())
                val_masks.append(mask.cpu().numpy())

        val_preds_log = np.concatenate(val_preds_log, axis=0)
        val_targets_log = np.concatenate(val_targets_log, axis=0)
        val_mask = np.concatenate(val_masks, axis=0)

        # Compute MA-RAE in original space
        val_preds_orig = val_preds_log.copy()
        val_targets_orig_array = val_targets_original.copy()

        for i_col, col in enumerate(TARGET_COLUMNS):
            if col != "LogD":
                val_preds_orig[:, i_col] = 10 ** val_preds_log[:, i_col]

        val_ma_rae = compute_ma_rae(val_preds_orig, val_targets_orig_array, val_mask)

        scheduler.step()

        if val_ma_rae < best_val_score:
            best_val_score = val_ma_rae
            patience_counter = 0
            torch.save(model.state_dict(), "./working/best_model.pth")
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss/len(train_loader):.4f} | Val MA-RAE: {val_ma_rae:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load("./working/best_model.pth"))
    model.eval()

    # Generate test predictions
    test_preds_log = []
    with torch.no_grad():
        for features, _, _ in test_loader:
            features = features.to(device)
            outputs = model(features)
            test_preds_log.append(outputs.cpu().numpy())
    test_preds_log = np.concatenate(test_preds_log, axis=0)

    # Inverse transform test predictions to original space
    test_predictions = test_preds_log.copy()
    for i_col, col in enumerate(TARGET_COLUMNS):
        if col != "LogD":
            test_predictions[:, i_col] = 10 ** test_preds_log[:, i_col]

    # Clip predictions to reasonable ranges
    clip_ranges = {
        "LogD": (-3.0, 6.0),
        "KSOL": (0.0, 500.0),
        "HLM CLint": (0.0, 10000.0),
        "MLM CLint": (0.0, 10000.0),
        "Caco-2 Permeability Papp A>B": (0.0, 150.0),
        "Caco-2 Permeability Efflux": (0.0, 150.0),
        "MPPB": (0.0, 100.0),
        "MBPB": (0.0, 100.0),
        "MGMB": (0.0, 100.0),
    }
    for i_col, col in enumerate(TARGET_COLUMNS):
        lo, hi = clip_ranges[col]
        test_predictions[:, i_col] = np.clip(test_predictions[:, i_col], lo, hi)

    # Create submission file
    submission = pd.DataFrame(
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
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {best_val_score:.4f}")
    return best_val_score


if __name__ == "__main__":
    score = train_and_evaluate()