import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Step 1: Data Processing and Feature Engineering
# ============================================================

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")


def smiles_to_mol(smiles):
    """Convert SMILES to RDKit mol object, return None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None


def compute_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """Compute Morgan (circular) fingerprint."""
    if mol is None:
        return np.zeros(n_bits)
    arr = np.zeros((1, n_bits), dtype=np.float32)
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    for i, bit in enumerate(fp):
        if bit:
            arr[0, i] = 1.0
    return arr[0]


def compute_maccs_keys(mol):
    """Compute MACCS keys (166-bit fingerprint)."""
    if mol is None:
        return np.zeros(166, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((1, 166), dtype=np.float32)
    for i, bit in enumerate(fp):
        if bit:
            arr[0, i] = 1.0
    return arr[0]


def compute_rdkit_descriptors(mol):
    """Compute basic RDKit physicochemical descriptors."""
    if mol is None:
        return {
            "MolWt": 0.0,
            "LogP": 0.0,
            "NumHDonors": 0,
            "NumHAcceptors": 0,
            "NumRotatableBonds": 0,
            "RingCount": 0,
            "AromaticProportion": 0.0,
            "FractionCsp3": 0.0,
            "HeavyAtomCount": 0,
            "NumHeteroatoms": 0,
            "TPSA": 0.0,
            "NumAromaticRings": 0,
            "NumAliphaticRings": 0,
            "NumSaturatedRings": 0,
            "NumBridgeheadAtoms": 0,
            "NumAtomStereoCenters": 0,
            "NumUnspecifiedAtomStereoCenters": 0,
            "MaxPartialCharge": 0.0,
            "MinPartialCharge": 0.0,
            "MaxAbsPartialCharge": 0.0,
            "MinAbsPartialCharge": 0.0,
            "NumValenceElectrons": 0,
            "Chi1v": 0.0,
            "Chi2v": 0.0,
            "Chi3v": 0.0,
            "Chi4v": 0.0,
            "HallKierAlpha": 0.0,
            "Kappa1": 0.0,
            "Kappa2": 0.0,
            "Kappa3": 0.0,
            "NOCount": 0,
            "NHOHCount": 0,
        }
    try:
        return {
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "RingCount": Descriptors.RingCount(mol),
            "AromaticProportion": Descriptors.AromaticProportion(mol),
            "FractionCsp3": Descriptors.FractionCSP3(mol),
            "HeavyAtomCount": mol.GetNumHeavyAtoms(),
            "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumAromaticRings": Descriptors.NumAromaticRings(mol),
            "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
            "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),
            "NumBridgeheadAtoms": Descriptors.NumBridgeheadAtoms(mol),
            "NumAtomStereoCenters": Descriptors.NumAtomStereoCenters(mol),
            "NumUnspecifiedAtomStereoCenters": Descriptors.NumUnspecifiedAtomStereoCenters(
                mol
            ),
            "MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
            "MinPartialCharge": Descriptors.MinPartialCharge(mol),
            "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge(mol),
            "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge(mol),
            "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
            "Chi1v": Descriptors.Chi1v(mol),
            "Chi2v": Descriptors.Chi2v(mol),
            "Chi3v": Descriptors.Chi3v(mol),
            "Chi4v": Descriptors.Chi4v(mol),
            "HallKierAlpha": Descriptors.HallKierAlpha(mol),
            "Kappa1": Descriptors.Kappa1(mol),
            "Kappa2": Descriptors.Kappa2(mol),
            "Kappa3": Descriptors.Kappa3(mol),
            "NOCount": Descriptors.NOCount(mol),
            "NHOHCount": Descriptors.NHOHCount(mol),
        }
    except:
        return {
            "MolWt": 0.0,
            "LogP": 0.0,
            "NumHDonors": 0,
            "NumHAcceptors": 0,
            "NumRotatableBonds": 0,
            "RingCount": 0,
            "AromaticProportion": 0.0,
            "FractionCsp3": 0.0,
            "HeavyAtomCount": 0,
            "NumHeteroatoms": 0,
            "TPSA": 0.0,
            "NumAromaticRings": 0,
            "NumAliphaticRings": 0,
            "NumSaturatedRings": 0,
            "NumBridgeheadAtoms": 0,
            "NumAtomStereoCenters": 0,
            "NumUnspecifiedAtomStereoCenters": 0,
            "MaxPartialCharge": 0.0,
            "MinPartialCharge": 0.0,
            "MaxAbsPartialCharge": 0.0,
            "MinAbsPartialCharge": 0.0,
            "NumValenceElectrons": 0,
            "Chi1v": 0.0,
            "Chi2v": 0.0,
            "Chi3v": 0.0,
            "Chi4v": 0.0,
            "HallKierAlpha": 0.0,
            "Kappa1": 0.0,
            "Kappa2": 0.0,
            "Kappa3": 0.0,
            "NOCount": 0,
            "NHOHCount": 0,
        }


def process_molecules(df):
    """Convert SMILES to molecular features for the entire dataframe."""
    smiles_list = df["SMILES"].values
    n = len(smiles_list)

    morgan_feats = np.zeros((n, 2048), dtype=np.float32)
    maccs_feats = np.zeros((n, 166), dtype=np.float32)

    desc_keys = [
        "MolWt",
        "LogP",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "RingCount",
        "AromaticProportion",
        "FractionCsp3",
        "HeavyAtomCount",
        "NumHeteroatoms",
        "TPSA",
        "NumAromaticRings",
        "NumAliphaticRings",
        "NumSaturatedRings",
        "NumBridgeheadAtoms",
        "NumAtomStereoCenters",
        "NumUnspecifiedAtomStereoCenters",
        "MaxPartialCharge",
        "MinPartialCharge",
        "MaxAbsPartialCharge",
        "MinAbsPartialCharge",
        "NumValenceElectrons",
        "Chi1v",
        "Chi2v",
        "Chi3v",
        "Chi4v",
        "HallKierAlpha",
        "Kappa1",
        "Kappa2",
        "Kappa3",
        "NOCount",
        "NHOHCount",
    ]
    desc_feats = np.zeros((n, len(desc_keys)), dtype=np.float32)

    invalid_smiles = 0

    for i, smiles in enumerate(smiles_list):
        mol = smiles_to_mol(smiles)
        if mol is None:
            invalid_smiles += 1
        else:
            fp_morgan = compute_morgan_fingerprint(mol)
            morgan_feats[i] = fp_morgan

            fp_maccs = compute_maccs_keys(mol)
            maccs_feats[i] = fp_maccs

            desc_dict = compute_rdkit_descriptors(mol)
            for j, key in enumerate(desc_keys):
                desc_feats[i, j] = desc_dict[key]

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{n} molecules")

    if invalid_smiles > 0:
        print(f"Warning: {invalid_smiles} invalid SMILES found")

    morgan_df = pd.DataFrame(morgan_feats, columns=[f"morgan_{i}" for i in range(2048)])
    maccs_df = pd.DataFrame(maccs_feats, columns=[f"maccs_{i}" for i in range(166)])
    desc_df = pd.DataFrame(desc_feats, columns=desc_keys)

    return morgan_df, maccs_df, desc_df


# Process training data
print("Processing training molecules...")
train_morgan, train_maccs, train_desc = process_molecules(train_df)

# Process test data
print("Processing test molecules...")
test_morgan, test_maccs, test_desc = process_molecules(test_df)

# Combine all features
train_features = pd.concat([train_morgan, train_maccs, train_desc], axis=1)
test_features = pd.concat([test_morgan, test_maccs, test_desc], axis=1)

# Define target columns
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

# Extract targets from training data
train_targets = train_df[target_cols].copy()

# Log-transform non-log targets
log_transform_cols = [
    "KSOL",
    "HLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
    "MPPB",
    "MBPB",
    "MGMB",
]
for col in log_transform_cols:
    train_targets[col] = np.log1p(train_targets[col].clip(lower=0))

# Create train/validation split (80/20)
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, shuffle=True
)

# Split features and targets BEFORE imputation to prevent data leakage
X_train = train_features.iloc[train_idx].values
y_train_raw = train_targets.iloc[train_idx].values
X_val = train_features.iloc[val_idx].values
y_val_raw = train_targets.iloc[val_idx].values
X_test = test_features.values

# Handle missing values in targets (fit ONLY on training data)
print(f"Target missing values before imputation:\n{pd.DataFrame(y_train_raw, columns=target_cols).isnull().sum()}")
target_imputer = SimpleImputer(strategy="median")
y_train = pd.DataFrame(
    target_imputer.fit_transform(y_train_raw), columns=target_cols
)
y_val = pd.DataFrame(
    target_imputer.transform(y_val_raw), columns=target_cols
)
print(
    f"Target missing values after imputation:\n{y_train.isnull().sum()}"
)

# Scale features
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled = feature_scaler.transform(X_val)
X_test_scaled = feature_scaler.transform(X_test)

# Save scaler and imputer for later use
os.makedirs("./working", exist_ok=True)
import joblib

joblib.dump(feature_scaler, "./working/feature_scaler.pkl")
joblib.dump(target_imputer, "./working/target_imputer.pkl")

# Convert to numpy arrays for consistency
y_train = y_train.values
y_val = y_val.values

# Save processed data
np.save("./working/X_train.npy", X_train_scaled)
np.save("./working/y_train.npy", y_train)
np.save("./working/X_val.npy", X_val_scaled)
np.save("./working/y_val.npy", y_val)
np.save("./working/X_test.npy", X_test_scaled)
np.save("./working/train_idx.npy", train_idx)
np.save("./working/val_idx.npy", val_idx)

print(f"Feature engineering complete!")
print(f"Train shape: {X_train_scaled.shape}")
print(f"Val shape: {X_val_scaled.shape}")
print(f"Test shape: {X_test_scaled.shape}")

# ============================================================
# Step 2: Model Design
# ============================================================


class MultiTaskMLP(nn.Module):
    """
    Multi-task neural network for predicting 9 molecular properties.
    """

    def __init__(
        self, input_dim, hidden_dims=[1024, 512, 256], dropout_rate=0.3, num_tasks=9
    ):
        super(MultiTaskMLP, self).__init__()
        self.num_tasks = num_tasks

        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU())  # Swish activation
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.shared_backbone = nn.Sequential(*layers)

        # Task-specific output heads
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dims[-1], 128),
                    nn.BatchNorm1d(128),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate * 0.5),
                    nn.Linear(128, 1),
                )
                for _ in range(num_tasks)
            ]
        )

        # Learnable log variance for each task
        self.log_var = nn.Parameter(torch.full((num_tasks,), -2.3, dtype=torch.float32))

        # Feature importance gate
        self.feature_gate = nn.Parameter(torch.ones(input_dim, dtype=torch.float32))

    def forward(self, x):
        # Apply feature gating
        x = x * torch.sigmoid(self.feature_gate)

        # Shared representation
        shared_features = self.shared_backbone(x)

        # Task-specific outputs
        outputs = []
        for i in range(self.num_tasks):
            output = self.task_heads[i](shared_features)
            outputs.append(output)

        # Stack outputs: (batch_size, num_tasks)
        predictions = torch.cat(outputs, dim=1)
        return predictions

    def get_uncertainty_loss(self, predictions, targets):
        """Heteroscedastic uncertainty weighted loss."""
        task_losses = []
        for i in range(self.num_tasks):
            pred_i = predictions[:, i]
            target_i = targets[:, i]

            # Only compute loss where targets are valid (non-NaN)
            valid_mask = ~torch.isnan(target_i)
            if valid_mask.sum() > 0:
                diff = pred_i[valid_mask] - target_i[valid_mask]
                precision = torch.exp(-self.log_var[i])
                task_loss = 0.5 * precision * (diff**2).mean() + 0.5 * self.log_var[i]
                task_losses.append(task_loss)

        if len(task_losses) == 0:
            return torch.tensor(0.0, device=predictions.device)

        return torch.stack(task_losses).sum()


# Model configuration
INPUT_DIM = 2246  # 2048 (Morgan) + 166 (MACCS) + 32 (RDKit descriptors)
NUM_TASKS = 9

# Initialize model
model = MultiTaskMLP(
    input_dim=INPUT_DIM,
    hidden_dims=[1024, 512, 256],
    dropout_rate=0.3,
    num_tasks=NUM_TASKS,
)

# Loss function
criterion = model.get_uncertainty_loss

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200, eta_min=1e-6
)

print(f"Model designed for multi-task regression:")
print(f"  - Input features: {INPUT_DIM}")
print(f"  - Number of tasks: {NUM_TASKS}")

# ============================================================
# Step 3: Training and Evaluation
# ============================================================

# Convert data to tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

# Build original NaN masks for training data (before imputation)
train_original_targets = train_df[target_cols].copy()
for col in log_transform_cols:
    train_original_targets[col] = np.log1p(train_original_targets[col].clip(lower=0))

# Create NaN masks for train and val
train_nan_mask = ~train_original_targets.isnull().values
val_nan_mask = train_nan_mask[val_idx]
train_nan_mask = train_nan_mask[train_idx]

# Compute per-task weights inversely proportional to missingness
missing_counts = train_original_targets.isnull().sum().values
task_weights = 1.0 / (missing_counts + 1)
task_weights = task_weights / task_weights.sum() * len(target_cols)
task_weights = torch.tensor(task_weights, dtype=torch.float32, device=device)
print(f"Task weights: {task_weights.cpu().numpy().round(2)}")

# Create dataloaders
batch_size = 256
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)

# Move model to device
model = model.to(device)

# Training hyperparameters
num_epochs = 500
best_val_loss = float("inf")
best_model_state = None
patience = 40
no_improve_count = 0

print("Starting training...")
for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0.0
    num_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    avg_train_loss = total_train_loss / num_batches

    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t.to(device))
        val_loss = criterion(val_preds, y_val_t.to(device)).item()

        # Compute per-task MAE on VALID (non-NaN) targets
        val_preds_np = val_preds.cpu().numpy()
        y_val_np = y_val_t.numpy()

        task_maes = []
        for t in range(len(target_cols)):
            mask_t = val_nan_mask[:, t]
            if mask_t.sum() > 0:
                pred_t = val_preds_np[mask_t, t]
                true_t = y_val_np[mask_t, t]
                mae_t = np.mean(np.abs(pred_t - true_t))
                task_maes.append(mae_t)

        if len(task_maes) > 0:
            val_metric = np.mean(task_maes)
        else:
            val_metric = val_loss

    # LR scheduler step
    scheduler.step()

    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_metric:.4f}"
    )

    # Early stopping
    if val_metric < best_val_loss:
        best_val_loss = val_metric
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(
                f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)"
            )
            break

# Load best model
model.load_state_dict(best_model_state)
model = model.to(device)
model.eval()

print(f"\nTraining complete. Best validation MAE: {best_val_loss:.4f}")

# Generate predictions
print("Generating predictions...")

with torch.no_grad():
    val_preds_final = model(X_val_t.to(device)).cpu().numpy()
    test_preds = model(X_test_t.to(device)).cpu().numpy()

# Post-process predictions back to original scale
log_transform_cols_indices = list(range(1, len(target_cols)))  # all except LogD

# For validation predictions
val_preds_original = np.zeros_like(val_preds_final)
val_preds_original[:, 0] = val_preds_final[:, 0]  # LogD - already in log scale
for idx in log_transform_cols_indices:
    val_preds_original[:, idx] = np.expm1(
        np.clip(val_preds_final[:, idx], a_min=None, a_max=10.0)
    )

# For test predictions
test_preds_original = np.zeros_like(test_preds)
test_preds_original[:, 0] = test_preds[:, 0]  # LogD - already in log scale
for idx in log_transform_cols_indices:
    test_preds_original[:, idx] = np.expm1(
        np.clip(test_preds[:, idx], a_min=None, a_max=10.0)
    )

# Compute final validation MAE (on original scale)
val_df = train_df.iloc[val_idx]
val_original_targets = val_df[target_cols].values.astype(np.float64)

task_maes_final = []
for t in range(len(target_cols)):
    mask_t = ~np.isnan(val_original_targets[:, t])
    if mask_t.sum() > 0:
        pred_t = val_preds_original[mask_t, t]
        true_t = val_original_targets[mask_t, t]
        mae_t = np.mean(np.abs(pred_t - true_t))
        task_maes_final.append(mae_t)
        print(f"  {target_cols[t]}: MAE = {mae_t:.4f} (n={mask_t.sum()})")

final_score = np.mean(task_maes_final)
print(f"\nFinal Validation MAE (macro-averaged across tasks): {final_score:.4f}")

# ============================================================
# Create submission file
# ============================================================
os.makedirs("./submission", exist_ok=True)

# Load test molecule names
test_names = test_df["Molecule Name"].values

# Load sample submission for correct column order
sample_sub = pd.read_csv("./input/sample_submission.csv")
cols = list(sample_sub.columns)

# Create submission dataframe
submission_df = pd.DataFrame(test_preds_original, columns=target_cols)
submission_df.insert(0, "Molecule Name", test_names)
submission_df = submission_df[cols]  # Ensure correct column order

# Clip values to reasonable ranges
submission_df["LogD"] = submission_df["LogD"].clip(-3.0, 6.0)
submission_df["KSOL"] = submission_df["KSOL"].clip(0.0, 500.0)
submission_df["HLM CLint"] = submission_df["HLM CLint"].clip(0.0, 5000.0)
submission_df["MLM CLint"] = submission_df["MLM CLint"].clip(0.0, 15000.0)
submission_df["Caco-2 Permeability Papp A>B"] = submission_df[
    "Caco-2 Permeability Papp A>B"
].clip(0.0, 100.0)
submission_df["Caco-2 Permeability Efflux"] = submission_df[
    "Caco-2 Permeability Efflux"
].clip(0.0, 200.0)
submission_df["MPPB"] = submission_df["MPPB"].clip(0.0, 100.0)
submission_df["MBPB"] = submission_df["MBPB"].clip(0.0, 100.0)
submission_df["MGMB"] = submission_df["MGMB"].clip(0.0, 100.0)

# Save submission
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

# Final validation score printing
print(f"Final Validation Score: {final_score}")