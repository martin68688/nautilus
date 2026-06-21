import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import math
import warnings
import pickle  # For saving/loading graph data

warnings.filterwarnings("ignore")


# ============================================================
# DATA PROCESSING AND FEATURE ENGINEERING
# ============================================================


def compute_morgan_fingerprints(smiles_list, radius=2, nBits=2048):
    """Compute ECFP-like Morgan fingerprints with fixed length using modern RDKit API."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(nBits))
        else:
            fp = mfpgen.GetFingerprint(mol)
            fps.append(np.array(fp))
    return np.array(fps)


def compute_maccs_fingerprints(smiles_list):
    """Compute MACCS keys (166-bit substructure fingerprints) using RDKit."""
    from rdkit.Chem import MACCSkeys
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(166, dtype=np.uint8))
        else:
            fp = MACCSkeys.GenMACCSKeys(mol)
            # GenMACCSKeys returns a 167-bit vector where bit 0 is always 0.
            # We convert to numpy and take bits 1-166 to get the standard 166-bit MACCS keys.
            fps.append(np.array(fp)[1:])
    return np.array(fps)


def compute_graph_features(smiles_list):
    """Convert SMILES to molecular graph representations: node features and adjacency matrices.

    For each molecule, computes:
    - Node features (one-hot encoded): atom type (6 types), degree (6), hybridization (5),
      formal charge (one-hot to 5 bins), chirality (4 types), ring membership (binary),
      aromaticity (binary), hydrogen count (one-hot to 5 bins).
    - Adjacency matrix: binary bond presence, with self-loops removed.

    Returns two lists: node_feat_list (list of torch.FloatTensor) and adj_list (list of torch.LongTensor).
    """
    from rdkit.Chem import rdchem

    # Atom type vocabulary (most common in drug-like molecules)
    atom_types = [6, 7, 8, 9, 16, 17]  # C, N, O, F, S, Cl
    atom_type_to_idx = {atom: i for i, atom in enumerate(atom_types)}

    # Hybridization types
    hyb_types = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
                 rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
                 rdchem.HybridizationType.SP3D2]

    # Chirality types
    chirality_types = [rdchem.ChiralType.CHI_UNSPECIFIED, rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                      rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, rdchem.ChiralType.CHI_OTHER]

    node_feat_list = []
    adj_list = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Return dummy graph with 1 node (zero features) for invalid SMILES
            node_feat_list.append(torch.zeros((1, 30), dtype=torch.float))
            adj_list.append(torch.zeros((1, 1), dtype=torch.long))
            continue

        # Ensure explicit hydrogens for accurate degree/h count computation
        mol = Chem.AddHs(mol)

        feat_dim = 30  # Total feature dimension
        num_nodes = mol.GetNumAtoms()

        node_feats = []
        for atom in mol.GetAtoms():
            feat = []
            # 1. Atom type (6-dim one-hot)
            atom_type = atom.GetAtomicNum()
            type_oh = [0.0] * len(atom_types)
            if atom_type in atom_type_to_idx:
                type_oh[atom_type_to_idx[atom_type]] = 1.0
            else:
                type_oh[0] = 1.0  # Default to carbon if unknown
            feat.extend(type_oh)

            # 2. Degree (0-5, one-hot) - capped at 5
            degree = min(atom.GetDegree(), 5)
            deg_oh = [0.0] * 6
            deg_oh[degree] = 1.0
            feat.extend(deg_oh)

            # 3. Hybridization (5-dim one-hot)
            hyb = atom.GetHybridization()
            hyb_oh = [0.0] * len(hyb_types)
            for j, h in enumerate(hyb_types):
                if hyb == h:
                    hyb_oh[j] = 1.0
                    break
            else:
                hyb_oh[0] = 1.0  # Default to SP
            feat.extend(hyb_oh)

            # 4. Formal charge (5 bins: -2, -1, 0, +1, +2)
            fc = atom.GetFormalCharge()
            fc_bin = min(max(fc + 2, 0), 4)  # Shift to 0-index from -2
            fc_oh = [0.0] * 5
            fc_oh[fc_bin] = 1.0
            feat.extend(fc_oh)

            # 5. Chirality (4-dim one-hot)
            chiral_tag = atom.GetChiralTag()
            chir_oh = [0.0] * len(chirality_types)
            for j, ct in enumerate(chirality_types):
                if chiral_tag == ct:
                    chir_oh[j] = 1.0
                    break
            else:
                chir_oh[0] = 1.0  # Default to unspecified
            feat.extend(chir_oh)

            # 6. Ring membership (1-dim binary)
            feat.append(1.0 if atom.IsInRing() else 0.0)

            # 7. Aromaticity (1-dim binary)
            feat.append(1.0 if atom.GetIsAromatic() else 0.0)

            # 8. Hydrogen count (0-4, one-hot)
            h_count = min(atom.GetTotalNumHs(), 4)
            h_oh = [0.0] * 5
            h_oh[h_count] = 1.0
            feat.extend(h_oh)

            node_feats.append(feat[:feat_dim])  # Ensure exact dimension

        # Pad to feature dimension if needed (should be exactly 30)
        node_feats_tensor = torch.tensor(node_feats, dtype=torch.float)
        if node_feats_tensor.shape[1] != feat_dim:
            pad = torch.zeros(node_feats_tensor.shape[0], feat_dim - node_feats_tensor.shape[1])
            node_feats_tensor = torch.cat([node_feats_tensor, pad], dim=1)

        node_feat_list.append(node_feats_tensor)

        # Build adjacency matrix (binary, no self-loops)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj[i, j] = 1
            adj[j, i] = 1

        adj_list.append(adj)

    return node_feat_list, adj_list


def compute_physicochemical_descriptors(smiles_list):
    """Compute a diverse set of molecular descriptors relevant to ADMET.
    Returns 22 columns: 12 original + 10 additional topological descriptors.
    """
    desc_names = [
        "MolWt",
        "LogP",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "NumAromaticRings",
        "NumAliphaticRings",
        "FractionCSP3",
        "RingCount",
        "HeavyAtomCount",
        "NumHeteroatoms",
        "BertzCT",
        "HallKierAlpha",
        "Kappa1",
        "Kappa2",
        "Kappa3",
        "Chi0v",
        "Chi1v",
        "MolMR",
        "HeavyAtomMolWt",
        "NumSaturatedRings",
    ]
    all_descs = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            all_descs.append([0.0] * len(desc_names))
        else:
            descs = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.RingCount(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.BertzCT(mol),
                Descriptors.HallKierAlpha(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.Kappa3(mol),
                Descriptors.Chi0v(mol),
                Descriptors.Chi1v(mol),
                Descriptors.MolMR(mol),
                Descriptors.HeavyAtomMolWt(mol),
                rdMolDescriptors.CalcNumSaturatedRings(mol),
            ]
            all_descs.append(descs)
    return pd.DataFrame(all_descs, columns=desc_names)


# Load data
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

# Extract SMILES
train_smiles = train_df["SMILES"].values
test_smiles = test_df["SMILES"].values

# Compute features
print("Computing Morgan fingerprints...")
train_fp = compute_morgan_fingerprints(train_smiles)
test_fp = compute_morgan_fingerprints(test_smiles)

print("Computing MACCS keys...")
train_maccs = compute_maccs_fingerprints(train_smiles)
test_maccs = compute_maccs_fingerprints(test_smiles)

print("Computing physicochemical descriptors...")
train_desc = compute_physicochemical_descriptors(train_smiles)
test_desc = compute_physicochemical_descriptors(test_smiles)

# Compute molecular graph features (for GNN branch)
# Compute separately for train and test to avoid any data leakage
print("Computing molecular graph features for training set...")
train_node_feats, train_adjs = compute_graph_features(train_smiles)
print("Computing molecular graph features for test set...")
test_node_feats, test_adjs = compute_graph_features(test_smiles)

# Save graph features for later use
import pickle
os.makedirs("./working", exist_ok=True)
with open("./working/train_graphs.pkl", "wb") as f:
    pickle.dump((train_node_feats, train_adjs), f)
with open("./working/test_graphs.pkl", "wb") as f:
    pickle.dump((test_node_feats, test_adjs), f)
print("Graph features saved.")

# Combine features: Morgan (2048) + MACCS (166) + descriptors (22) = 2236
train_features = np.hstack([train_fp, train_maccs, train_desc.values])
test_features = np.hstack([test_fp, test_maccs, test_desc.values])

fp_cols = [f"fp_{i}" for i in range(train_fp.shape[1])]
maccs_cols = [f"maccs_{i}" for i in range(train_maccs.shape[1])]
desc_cols = list(train_desc.columns)
all_feature_cols = fp_cols + maccs_cols + desc_cols

train_feat_df = pd.DataFrame(train_features, columns=all_feature_cols)
test_feat_df = pd.DataFrame(test_features, columns=all_feature_cols)

train_feat_df["Molecule Name"] = train_df["Molecule Name"].values
test_feat_df["Molecule Name"] = test_df["Molecule Name"].values

# Targets - log-transform where appropriate and standardize
from sklearn.preprocessing import StandardScaler

target_train = train_df[target_cols].copy().astype(np.float64)
target_missing_mask = target_train.isna().astype(int)

# Train/validation split FIRST (before any leakage-prone processing)
X_train, X_val, y_train_raw, y_val_raw, mask_train, mask_val = train_test_split(
    train_feat_df,
    target_train,
    target_missing_mask,
    test_size=0.2,
    random_state=42,
)

# Compute robust shift values from TRAINING SPLIT only for log transform
shift_values = {}
for col in target_cols:
    col_valid = y_train_raw[col].dropna()
    col_min = col_valid.min()
    shift_values[col] = max(0, -col_min + 1.0)  # ensure positive shift

# Log-transform targets using ONLY training split statistics
y_train_log = y_train_raw.copy()
for col in target_cols:
    shift = shift_values[col]
    y_train_log[col] = np.log1p(y_train_raw[col] + shift)

y_val_log = y_val_raw.copy()
for col in target_cols:
    shift = shift_values[col]
    y_val_log[col] = np.log1p(y_val_raw[col] + shift)

# Impute NaN with median of log-transformed values (from TRAIN only)
# Also clip extreme values to handle outliers
target_medians_log = y_train_log.median()
y_train_filled = y_train_log.fillna(target_medians_log)
y_val_filled = y_val_log.fillna(target_medians_log)

# Clip extreme values for stability
for col in target_cols:
    # Clip at 5 standard deviations from mean
    col_mean = y_train_filled[col].mean()
    col_std = y_train_filled[col].std()
    upper = col_mean + 5 * col_std
    lower = col_mean - 5 * col_std
    y_train_filled[col] = y_train_filled[col].clip(lower, upper)
    y_val_filled[col] = y_val_filled[col].clip(lower, upper)

X_train_feats = X_train.drop(columns=["Molecule Name"])
X_val_feats = X_val.drop(columns=["Molecule Name"])
test_feats = test_feat_df.drop(columns=["Molecule Name"])

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_feats)
X_val_scaled = scaler.transform(X_val_feats)
test_scaled = scaler.transform(test_feats)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_feats.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val_feats.columns)
test_scaled = pd.DataFrame(test_scaled, columns=test_feats.columns)

X_train_scaled["Molecule Name"] = X_train["Molecule Name"].values
X_val_scaled["Molecule Name"] = X_val["Molecule Name"].values
test_scaled["Molecule Name"] = test_feat_df["Molecule Name"].values

os.makedirs("./working", exist_ok=True)
X_train_scaled.to_parquet("./working/X_train.parquet", index=False)
X_val_scaled.to_parquet("./working/X_val.parquet", index=False)
test_scaled.to_parquet("./working/X_test.parquet", index=False)
y_train_filled.to_parquet("./working/y_train.parquet", index=False)
y_val_filled.to_parquet("./working/y_val.parquet", index=False)
mask_train.to_parquet("./working/mask_train.parquet", index=False)
mask_val.to_parquet("./working/mask_val.parquet", index=False)
target_medians_log.to_frame("medians").to_parquet("./working/target_medians.parquet")

# Also save shift values for inverse transform later (from TRAIN only)
# shift_values already computed above, just save it
pd.Series(shift_values).to_frame("shift").to_parquet("./working/target_shifts.parquet")

print(
    f"Training samples: {len(X_train_scaled)}, Validation samples: {len(X_val_scaled)}, Test samples: {len(test_scaled)}"
)
print(f"Number of features: {len(all_feature_cols)}")
print("Data processing and feature engineering complete.")


# ============================================================
# MODEL DESIGN
# ============================================================


class GINLayer(nn.Module):
    """Graph Isomorphism Network (GIN) layer with learnable epsilon."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.epsilon = nn.Parameter(torch.tensor(0.0))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, node_feats, adj):
        # adj is (N, N) binary - create degree-normalized adjacency
        # Add self-loop based on learned epsilon
        I = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        adj_self = adj + (1.0 + self.epsilon) * I

        # Message passing: sum over neighbors
        # adj_self @ node_feats  (N, N) x (N, Feat) = (N, Feat)
        msg = torch.mm(adj_self.float(), node_feats)

        # Apply MLP
        out = self.mlp(msg)
        return out


class GINE(nn.Module):
    """Graph Isomorphism Network encoder with global mean pooling."""
    def __init__(self, node_feat_dim=30, hidden_dim=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GINLayer(node_feat_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GINLayer(hidden_dim, hidden_dim))
        self.graph_out_dim = hidden_dim  # 64-dim graph embedding

    def forward(self, node_feats, adj):
        # node_feats: list of tensors per graph (batch of variable-size graphs)
        # adj: list of adjacency matrices
        batch_graph_embeddings = []
        for nf, a in zip(node_feats, adj):
            nf = nf.to(next(self.parameters()).device)
            a = a.to(next(self.parameters()).device)
            h = nf
            for conv in self.convs:
                h = conv(h, a)
            # Global mean pooling
            graph_embedding = h.mean(dim=0)  # (hidden_dim,)
            batch_graph_embeddings.append(graph_embedding)
        # Stack into batch tensor
        return torch.stack(batch_graph_embeddings, dim=0)  # (B, hidden_dim)


class MultiTaskADMETPredictor(nn.Module):
    def __init__(
        self,
        input_dim=2236,
        hidden_dim=1024,
        num_targets=9,
        dropout_rate=0.5,
        gnn_hidden_dim=64,
        gnn_num_layers=3,
        node_feat_dim=30,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.gnn_hidden_dim = gnn_hidden_dim

        # Fingerprint / descriptor MLP branch
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Graph Neural Network branch (GIN)
        self.gnn_branch = GINE(
            node_feat_dim=node_feat_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
        )

        # Fusion: combined dim = 1024 (MLP) + 64 (GNN) = 1088
        combined_dim = 1024 + gnn_hidden_dim

        self.target_predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(combined_dim, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, 1),
                )
                for _ in range(num_targets)
            ]
        )

    def forward(self, features, node_feats=None, adjs=None):
        shared = self.shared_encoder(features)

        if node_feats is not None and adjs is not None:
            # GNN branch forward
            graph_embedding = self.gnn_branch(node_feats, adjs)  # (B, 64)
            # Concatenate
            combined = torch.cat([shared, graph_embedding], dim=1)  # (B, 1088)
        else:
            combined = shared  # Fallback to fingerprint-only

        predictions = []
        for t_idx in range(self.num_targets):
            pred = self.target_predictors[t_idx](combined)
            predictions.append(pred)
        return torch.cat(predictions, dim=-1)


class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights=None, alpha=0.5):
        super().__init__()
        self.task_weights = task_weights
        self.alpha = alpha

    def forward(self, predictions, targets, mask):
        abs_error = torch.abs(predictions - targets)
        mae_loss = (abs_error * mask).sum() / (mask.sum() + 1e-8)

        sq_error = (predictions - targets) ** 2
        mse_loss = (sq_error * mask).sum() / (mask.sum() + 1e-8)

        loss = self.alpha * mae_loss + (1 - self.alpha) * mse_loss

        if self.task_weights is not None:
            task_weights_norm = self.task_weights / self.task_weights.mean()
            weighted_loss = (
                abs_error * mask * task_weights_norm.unsqueeze(0)
            ).sum() / (mask.sum() + 1e-8)
            loss = loss + weighted_loss * 0.5

        return loss


# ============================================================
# TRAINING AND EVALUATION
# ============================================================


class ADMETDataset(Dataset):
    def __init__(self, features_df, node_feats_list=None, adj_list=None, targets_df=None, mask_df=None):
        self.features = features_df.drop(columns=["Molecule Name"]).values.astype(
            np.float32
        )
        self.molecule_names = features_df["Molecule Name"].values
        self.node_feats_list = node_feats_list  # list of tensors per molecule
        self.adj_list = adj_list                # list of tensors per molecule
        self.targets = (
            targets_df.values.astype(np.float32) if targets_df is not None else None
        )
        self.mask = mask_df.values.astype(np.float32) if mask_df is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.features[idx])
        result = {"features": feat, "idx": idx}
        if self.node_feats_list is not None:
            result["node_feats"] = self.node_feats_list[idx]
        if self.adj_list is not None:
            result["adj"] = self.adj_list[idx]
        if self.targets is not None:
            result["targets"] = torch.from_numpy(self.targets[idx])
        if self.mask is not None:
            result["mask"] = torch.from_numpy(self.mask[idx])
        return result


class TestDataset(Dataset):
    def __init__(self, features_df, node_feats_list=None, adj_list=None):
        self.features = features_df.drop(columns=["Molecule Name"]).values.astype(
            np.float32
        )
        self.molecule_names = features_df["Molecule Name"].values
        self.node_feats_list = node_feats_list
        self.adj_list = adj_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.features[idx])
        result = {"features": feat, "idx": idx}
        if self.node_feats_list is not None:
            result["node_feats"] = self.node_feats_list[idx]
        if self.adj_list is not None:
            result["adj"] = self.adj_list[idx]
        return result


# tokenize_smiles is not needed - using only fingerprint-based features


def compute_ma_rae(predictions, targets, mask, epsilon=1e-6):
    num_targets = predictions.shape[1]
    rae_per_target = []
    for t in range(num_targets):
        valid_idx = mask[:, t] > 0.5
        if valid_idx.sum() < 2:
            rae_per_target.append(1.0)
            continue
        y_true = targets[valid_idx, t]
        y_pred = predictions[valid_idx, t]
        mae = torch.mean(torch.abs(y_pred - y_true))
        y_mean = torch.mean(y_true)
        mad = torch.mean(torch.abs(y_true - y_mean))
        # If MAD is very small relative to target magnitude, target has near-constant value
        # In that case, use MAE scaled by mean absolute target value instead
        if mad < epsilon:
            ref = torch.mean(torch.abs(y_true))
            if ref < epsilon:
                rae_per_target.append(0.0)
            else:
                rae_per_target.append((mae / (ref + epsilon)).item())
        else:
            rae = mae / (mad + epsilon)
            rae_per_target.append(rae.item())
    ma_rae = np.mean(rae_per_target)
    return ma_rae


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    config,
):
    num_epochs = config["num_epochs"]
    patience = config["patience"]
    gradient_clip_val = config["gradient_clip_val"]

    best_val_score = float("inf")
    patience_counter = 0
    best_model_state = None

    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            # Get graph data (may not be present if not using GNN)
            node_feats = batch.get("node_feats", None)
            adjs = batch.get("adj", None)

            optimizer.zero_grad()
            if node_feats is not None and adjs is not None:
                predictions = model(features, node_feats, adjs)
            else:
                predictions = model(features)
            loss = criterion(predictions, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches

        # Validation
        model.eval()
        all_val_preds, all_val_targets, all_val_masks = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                mask = batch["mask"].to(device)

                node_feats = batch.get("node_feats", None)
                adjs = batch.get("adj", None)

                if node_feats is not None and adjs is not None:
                    predictions = model(features, node_feats, adjs)
                else:
                    predictions = model(features)

                all_val_preds.append(predictions.cpu())
                all_val_targets.append(targets.cpu())
                all_val_masks.append(mask.cpu())

        val_preds = torch.cat(all_val_preds, dim=0)
        val_targets = torch.cat(all_val_targets, dim=0)
        val_masks = torch.cat(all_val_masks, dim=0)

        val_score = compute_ma_rae(val_preds, val_targets, val_masks)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val MA-RAE: {val_score:.4f} | LR: {current_lr:.2e}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    model.load_state_dict(best_model_state)
    model = model.to(device)
    return model, best_val_score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train = pd.read_parquet("./working/X_train.parquet")
    X_val = pd.read_parquet("./working/X_val.parquet")
    X_test = pd.read_parquet("./working/X_test.parquet")
    y_train = pd.read_parquet("./working/y_train.parquet")
    y_val = pd.read_parquet("./working/y_val.parquet")
    target_medians_log = pd.read_parquet("./working/target_medians.parquet")["medians"].values
    target_shifts = pd.read_parquet("./working/target_shifts.parquet")["shift"].values
    mask_train = pd.read_parquet("./working/mask_train.parquet")
    mask_val = pd.read_parquet("./working/mask_val.parquet")

    config = {
        "num_epochs": 100,
        "patience": 30,
        "gradient_clip_val": 1.0,
        "batch_size": 16,
        "num_workers": 2,
        "fp_dim": 2048,
        "maccs_dim": 166,
        "desc_dim": 22,
        "hidden_dim": 2048,
        "num_targets": 9,
        "dropout_rate": 0.5,
        "lr": 2e-4,
        "weight_decay": 0.05,
        "min_lr": 1e-6,
    }

    # Load graph features
    with open("./working/train_graphs.pkl", "rb") as f:
        train_node_feats, train_adjs = pickle.load(f)
    with open("./working/test_graphs.pkl", "rb") as f:
        test_node_feats, test_adjs = pickle.load(f)

    # Split graph features to match train/val split
    # Use same train_test_split as before (X_train, X_val indices)
    X_train_idx = X_train.index.values
    X_val_idx = X_val.index.values

    train_node_feats_split = [train_node_feats[i] for i in X_train_idx]
    train_adjs_split = [train_adjs[i] for i in X_train_idx]
    val_node_feats = [train_node_feats[i] for i in X_val_idx]
    val_adjs = [train_adjs[i] for i in X_val_idx]

    # Graph features for test are already separate
    test_node_feats_for_infer = test_node_feats
    test_adjs_for_infer = test_adjs

    train_dataset = ADMETDataset(X_train, train_node_feats_split, train_adjs_split, y_train, mask_train)
    val_dataset = ADMETDataset(X_val, val_node_feats, val_adjs, y_val, mask_val)

    class GraphCollator:
        """Custom collator for variable-size graph data."""
        def __call__(self, batch):
            features = torch.stack([item["features"] for item in batch], dim=0)
            node_feats_list = [item["node_feats"] for item in batch]
            adj_list = [item["adj"] for item in batch]

            result = {
                "features": features,
                "node_feats": node_feats_list,
                "adj": adj_list,
                "idx": [item["idx"] for item in batch],
            }

            if "targets" in batch[0]:
                result["targets"] = torch.stack([item["targets"] for item in batch], dim=0)
            if "mask" in batch[0]:
                result["mask"] = torch.stack([item["mask"] for item in batch], dim=0)

            return result

    collator = GraphCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # Must be 0 for custom collate with variable-size graph data
        pin_memory=False,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator,
    )

    model = MultiTaskADMETPredictor(
        input_dim=config["fp_dim"] + config["maccs_dim"] + config["desc_dim"],
        hidden_dim=config["hidden_dim"],
        num_targets=config["num_targets"],
        dropout_rate=config["dropout_rate"],
        gnn_hidden_dim=64,
        gnn_num_layers=3,
        node_feat_dim=30,
    ).to(device)

    task_counts = mask_train.sum(axis=0).values
    total_samples = len(mask_train)
    task_weights = total_samples / (task_counts + 1)
    task_weights = task_weights / task_weights.mean()
    task_weights_tensor = torch.tensor(task_weights, dtype=torch.float32).to(device)

    criterion = MultiTaskLoss(task_weights=task_weights_tensor, alpha=0.5)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["num_epochs"],
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
    )

    model, best_val_score = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        config,
    )

    print(f"Best validation MA-RAE: {best_val_score:.4f}")

    torch.save(model.state_dict(), "./working/best_model.pth")
    print("Best model saved to ./working/best_model.pth")

    # Compute per-target valid ranges for clipping (from TRAINING SPLIT only)
    # Use the original (non-log-transformed) training data from the training split indices
    train_original = pd.read_csv("./input/train.csv")
    train_split_indices = X_train.index.values  # Indices from train_test_split
    train_split_original = train_original.iloc[train_split_indices]
    target_ranges = {}
    for col in target_cols:
        col_valid = train_split_original[col].dropna()
        target_ranges[col] = {
            "min": col_valid.min(),
            "max": col_valid.max(),
            "range": col_valid.max() - col_valid.min(),
        }

    # Test inference
    print("Performing test inference...")

    test_dataset = TestDataset(X_test, test_node_feats_for_infer, test_adjs_for_infer)

    class TestCollator:
        """Custom collator for test data with variable-size graphs."""
        def __call__(self, batch):
            features = torch.stack([item["features"] for item in batch], dim=0)
            node_feats_list = [item["node_feats"] for item in batch]
            adj_list = [item["adj"] for item in batch]

            result = {
                "features": features,
                "node_feats": node_feats_list,
                "adj": adj_list,
                "idx": [item["idx"] for item in batch],
            }
            return result

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=TestCollator(),
    )

    model.eval()
    all_test_preds = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            node_feats = batch.get("node_feats", None)
            adjs = batch.get("adj", None)

            if node_feats is not None and adjs is not None:
                predictions = model(features, node_feats, adjs)
            else:
                predictions = model(features)
            all_test_preds.append(predictions.cpu().numpy())

    test_predictions_log = np.concatenate(all_test_preds, axis=0)

    # Inverse log-transform: expm1 - shift
    test_predictions = np.zeros_like(test_predictions_log)
    for i, col in enumerate(target_cols):
        shift_val = target_shifts[i]
        # Clip the log predictions to avoid extreme values during expm1
        clipped_log = np.clip(test_predictions_log[:, i], -20, 20)
        test_predictions[:, i] = np.expm1(clipped_log) - shift_val
        # Clamp to the known valid range from training data
        col_range = target_ranges[col]
        col_min_ref = col_range["min"]
        col_max_ref = col_range["max"]
        col_span = col_range["range"]
        # Use a reasonable range: allow slight extrapolation (10% beyond observed range)
        lower_bound = col_min_ref - col_span * 0.1
        upper_bound = col_max_ref + col_span * 0.1
        test_predictions[:, i] = np.clip(test_predictions[:, i], lower_bound, upper_bound)
        # For non-negative targets, enforce positivity
        if col_min_ref >= 0:
            test_predictions[:, i] = np.maximum(0, test_predictions[:, i])

    # Create submission
    submission_df = pd.DataFrame({"Molecule Name": X_test["Molecule Name"].values})
    for i, col in enumerate(target_cols):
        submission_df[col] = test_predictions[:, i]
    submission_df = submission_df[["Molecule Name"] + target_cols]

    os.makedirs("./submission", exist_ok=True)
    submission_df.to_csv("./submission/submission.csv", index=False)
    print(f"Submission saved to ./submission/submission.csv")
    print(f"Submission shape: {submission_df.shape}")

    print(f"Final Validation Score: {best_val_score}")


if __name__ == "__main__":
    main()