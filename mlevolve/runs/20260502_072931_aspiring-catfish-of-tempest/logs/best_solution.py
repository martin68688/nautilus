import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import pickle
import json
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors

warnings.filterwarnings("ignore")

# Set device and seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ========== DATA LOADING ==========
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Define target columns (matching sample submission order)
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

# Define which targets need log transformation (non-log endpoints)
log_targets = ["HLM CLint", "MLM CLint", "KSOL", "Caco-2 Permeability Efflux"]

# ========== FEATURE ENGINEERING ==========
def compute_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_maccs_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((1,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_rdkit_fingerprint(smiles, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
    arr = np.zeros((1,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_physicochemical_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "MolWt": 0,
            "LogP": 0,
            "TPSA": 0,
            "NumHDonors": 0,
            "NumHAcceptors": 0,
            "NumRotatableBonds": 0,
            "NumAromaticRings": 0,
            "NumAliphaticRings": 0,
            "NumSaturatedRings": 0,
            "NumHeteroatoms": 0,
            "FractionCSP3": 0,
            "HeavyAtomCount": 0,
            "NHOHCount": 0,
            "NOCount": 0,
            "MolMR": 0,
            "RingCount": 0,
        }
    desc = {}
    desc["MolWt"] = Descriptors.MolWt(mol)
    desc["LogP"] = Descriptors.MolLogP(mol)
    desc["TPSA"] = Descriptors.TPSA(mol)
    desc["NumHDonors"] = Descriptors.NumHDonors(mol)
    desc["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)
    desc["NumRotatableBonds"] = Descriptors.NumRotatableBonds(mol)
    desc["NumAromaticRings"] = Descriptors.NumAromaticRings(mol)
    desc["NumAliphaticRings"] = Descriptors.NumAliphaticRings(mol)
    desc["NumSaturatedRings"] = Descriptors.NumSaturatedRings(mol)
    desc["NumHeteroatoms"] = Descriptors.NumHeteroatoms(mol)
    desc["FractionCSP3"] = Descriptors.FractionCSP3(mol)
    desc["HeavyAtomCount"] = Descriptors.HeavyAtomCount(mol)
    desc["NHOHCount"] = Descriptors.NHOHCount(mol)
    desc["NOCount"] = Descriptors.NOCount(mol)
    desc["MolMR"] = Descriptors.MolMR(mol)
    desc["RingCount"] = Descriptors.RingCount(mol)
    ring_info = mol.GetRingInfo()
    ring_sizes = {}
    for ring in ring_info.AtomRings():
        size = len(ring)
        key = f"RingSize{size}"
        ring_sizes[key] = ring_sizes.get(key, 0) + 1
    for size in [3, 4, 5, 6, 7, 8]:
        desc[f"RingSize{size}"] = ring_sizes.get(f"RingSize{size}", 0)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    desc["AromaticAtomCount"] = aromatic_atoms
    return desc


def compute_all_features(smiles_list):
    n = len(smiles_list)
    morgan_fps = []
    maccs_fps = []
    rdkit_fps = []
    descriptors_list = []
    for i, smiles in enumerate(smiles_list):
        if i % 500 == 0:
            print(f"Processing molecule {i}/{n}")
        morgan_fps.append(compute_morgan_fingerprint(smiles))
        maccs_fps.append(compute_maccs_fingerprint(smiles))
        rdkit_fps.append(compute_rdkit_fingerprint(smiles))
        descriptors_list.append(compute_physicochemical_descriptors(smiles))
    morgan_arr = np.stack(morgan_fps)
    maccs_arr = np.stack(maccs_fps)
    rdkit_arr = np.stack(rdkit_fps)
    desc_df = pd.DataFrame(descriptors_list)
    return morgan_arr, maccs_arr, rdkit_arr, desc_df


print("Computing features for training set...")
train_morgan, train_maccs, train_rdkit, train_desc = compute_all_features(
    train_df["SMILES"].values
)
print("Computing features for test set...")
test_morgan, test_maccs, test_rdkit, test_desc = compute_all_features(
    test_df["SMILES"].values
)

# Combine all features
train_features = np.hstack([train_morgan, train_maccs, train_rdkit])
test_features = np.hstack([test_morgan, test_maccs, test_rdkit])
train_features = np.hstack([train_features, train_desc.values])
test_features = np.hstack([test_features, test_desc.values])

fp_feature_names = (
    [f"morgan_{i}" for i in range(2048)]
    + [f"maccs_{i}" for i in range(167)]
    + [f"rdkit_{i}" for i in range(1024)]
    + list(train_desc.columns)
)
print(f"Total feature count: {len(fp_feature_names)}")

# Create train/validation split BEFORE scaling to prevent data leakage
train_df["n_non_missing"] = train_df[target_cols].notna().sum(axis=1)
train_df["missing_category"] = pd.cut(
    train_df["n_non_missing"], bins=[0, 3, 5, 7, 9], labels=[0, 1, 2, 3]
)
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=42,
    stratify=train_df["missing_category"],
)

train_feature_df = pd.DataFrame(train_features, columns=fp_feature_names)
test_feature_df = pd.DataFrame(test_features, columns=fp_feature_names)
train_feature_df = train_feature_df.fillna(0)
test_feature_df = test_feature_df.fillna(0)

n_descriptors = len(train_desc.columns)
if n_descriptors > 0:
    scaler = RobustScaler()
    # Fit scaler ONLY on training data
    train_feature_df.iloc[train_idx, -n_descriptors:] = scaler.fit_transform(
        train_feature_df.iloc[train_idx, -n_descriptors:]
    )
    # Transform validation and test using training statistics
    train_feature_df.iloc[val_idx, -n_descriptors:] = scaler.transform(
        train_feature_df.iloc[val_idx, -n_descriptors:]
    )
    test_feature_df.iloc[:, -n_descriptors:] = scaler.transform(
        test_feature_df.iloc[:, -n_descriptors:]
    )

X_train = train_feature_df.iloc[train_idx].values
X_val = train_feature_df.iloc[val_idx].values
X_test = test_feature_df.values

y_train = train_df[target_cols].iloc[train_idx].values.copy()
y_val = train_df[target_cols].iloc[val_idx].values.copy()

# Apply log transformations AFTER split to prevent data leakage
for col_idx, col in enumerate(target_cols):
    if col in log_targets:
        # Train
        mask_train = ~np.isnan(y_train[:, col_idx]) & (y_train[:, col_idx] > 0)
        y_train[mask_train, col_idx] = np.log1p(y_train[mask_train, col_idx])
        # Validation
        mask_val = ~np.isnan(y_val[:, col_idx]) & (y_val[:, col_idx] > 0)
        y_val[mask_val, col_idx] = np.log1p(y_val[mask_val, col_idx])

# Handle missing values in targets
target_imputer = SimpleImputer(strategy="median")
y_train_imputed = target_imputer.fit_transform(y_train)
y_val_imputed = target_imputer.transform(y_val)

log_transformed_mask = np.array([col in log_targets for col in target_cols])

print(
    f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}"
)
print(f"Feature dimension: {X_train.shape[1]}")

train_molecule_names = train_df["Molecule Name"].iloc[train_idx].values
val_molecule_names = train_df["Molecule Name"].iloc[val_idx].values
test_molecule_names = test_df["Molecule Name"].values

# Save processed data
os.makedirs("./working", exist_ok=True)
np.save("./working/X_train.npy", X_train)
np.save("./working/X_val.npy", X_val)
np.save("./working/X_test.npy", X_test)
np.save("./working/y_train_original.npy", y_train)
np.save("./working/y_val_original.npy", y_val)
np.save("./working/train_idx.npy", train_idx)
np.save("./working/val_idx.npy", val_idx)
np.save("./working/target_cols.npy", np.array(target_cols))
np.save("./working/log_transformed_mask.npy", log_transformed_mask)
np.save("./working/train_molecule_names.npy", train_molecule_names)
np.save("./working/val_molecule_names.npy", val_molecule_names)
np.save("./working/test_molecule_names.npy", test_molecule_names)

with open("./working/target_imputer.pkl", "wb") as f:
    pickle.dump(target_imputer, f)
with open("./working/desc_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# ========== GRAPH FEATURE PREPARATION ==========
def get_node_features_and_adjacency(smiles_list, max_atoms=50):
    atom_types = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Cl",
        "Br",
        "I",
        "P",
        "Si",
        "B",
        "Se",
        "Te",
        "As",
    ]
    batch_node_features = []
    batch_adj = []
    valid_mask = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            node_feats = torch.zeros((max_atoms, 74))
            adj = torch.eye(max_atoms)
            batch_node_features.append(node_feats)
            batch_adj.append(adj)
            valid_mask.append(False)
            continue
        atoms = list(mol.GetAtoms())
        n_atoms = min(len(atoms), max_atoms)
        node_feats = torch.zeros((n_atoms, 74))
        for i, atom in enumerate(atoms[:n_atoms]):
            atom_symbol = atom.GetSymbol()
            if atom_symbol in atom_types:
                node_feats[i, atom_types.index(atom_symbol)] = 1.0
            else:
                node_feats[i, len(atom_types)] = 1.0
            degree = min(atom.GetDegree(), 5)
            node_feats[i, 15 + degree] = 1.0
            formal_charge = atom.GetFormalCharge()
            node_feats[i, 21] = min(max(formal_charge + 2, 0), 4) / 4.0
            hyb = atom.GetHybridization()
            if hyb == Chem.rdchem.HybridizationType.SP:
                node_feats[i, 22] = 1.0
            elif hyb == Chem.rdchem.HybridizationType.SP2:
                node_feats[i, 23] = 1.0
            elif hyb == Chem.rdchem.HybridizationType.SP3:
                node_feats[i, 24] = 1.0
            elif hyb == Chem.rdchem.HybridizationType.SP3D:
                node_feats[i, 25] = 1.0
            else:
                node_feats[i, 26] = 1.0
            node_feats[i, 27] = 1.0 if atom.GetIsAromatic() else 0.0
            num_h = atom.GetTotalNumHs()
            node_feats[i, 28] = min(num_h, 4) / 4.0
            node_feats[i, 29] = atom.GetMass() / 200.0
            electronegativity_map = {
                "C": 2.55,
                "N": 3.04,
                "O": 3.44,
                "S": 2.58,
                "F": 3.98,
                "Cl": 3.16,
                "Br": 2.96,
                "I": 2.66,
                "P": 2.19,
                "Si": 1.90,
                "B": 2.04,
                "Se": 2.55,
                "Te": 2.10,
                "As": 2.18,
            }
            en = electronegativity_map.get(atom_symbol, 2.5)
            node_feats[i, 31] = (en - 1.5) / 3.0
            node_feats[i, 32] = 1.0 if atom.IsInRing() else 0.0
            node_feats[i, 33] = (
                1.0
                if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
                else 0.0
            )
        adj = torch.zeros((n_atoms, n_atoms))
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i < n_atoms and j < n_atoms:
                bond_type = bond.GetBondTypeAsDouble()
                adj[i, j] = bond_type
                adj[j, i] = bond_type
        adj = adj + torch.eye(n_atoms)
        if n_atoms < max_atoms:
            pad_size = max_atoms - n_atoms
            node_feats = torch.cat([node_feats, torch.zeros((pad_size, 74))], dim=0)
            adj_pad_h = torch.cat([adj, torch.zeros((n_atoms, pad_size))], dim=1)
            adj_pad_v = torch.cat(
                [adj_pad_h, torch.zeros((pad_size, max_atoms))], dim=0
            )
            adj_pad_v[range(n_atoms, max_atoms), range(n_atoms, max_atoms)] = 1.0
            adj = adj_pad_v
        batch_node_features.append(node_feats)
        batch_adj.append(adj)
        valid_mask.append(True)
    return (
        torch.stack(batch_node_features),
        torch.stack(batch_adj),
        torch.tensor(valid_mask),
    )


# Get SMILES for train/val/test splits
train_smiles = train_df["SMILES"].iloc[train_idx].values
val_smiles = train_df["SMILES"].iloc[val_idx].values
test_smiles = test_df["SMILES"].values

print("Computing graph features for training set...")
train_node_feats, train_adj, _ = get_node_features_and_adjacency(train_smiles)
print("Computing graph features for validation set...")
val_node_feats, val_adj, _ = get_node_features_and_adjacency(val_smiles)
print("Computing graph features for test set...")
test_node_feats, test_adj, _ = get_node_features_and_adjacency(test_smiles)

# Extract fingerprint components (first 2048+167+1024 features)
train_fp = torch.FloatTensor(X_train[:, :3239])
val_fp = torch.FloatTensor(X_val[:, :3239])
test_fp = torch.FloatTensor(X_test[:, :3239])


# ========== MODEL DEFINITION ==========
class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.15):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=adj_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class GraphTransformerEncoder(nn.Module):
    def __init__(self, node_features=74, hidden_dim=128, num_layers=3, num_heads=4):
        super().__init__()
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.fingerprint_proj = nn.Linear(3239, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.15)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fingerprint_fusion_proj = nn.Linear(3239, hidden_dim)

    def forward(self, node_features, adj_matrix, fingerprints):
        batch_size = node_features.size(0)
        x = self.node_embedding(node_features)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        adj_mask = None
        for layer in self.layers:
            x = layer(x, adj_mask)
        cls_out = x[:, 0, :]
        fp = F.relu(self.fingerprint_proj(fingerprints)).unsqueeze(1)
        cls_out_expanded = cls_out.unsqueeze(1)
        cross_out, _ = self.cross_attn(cls_out_expanded, fp, fp)
        cross_out = self.cross_norm(cls_out_expanded + cross_out).squeeze(1)
        fp_fused = F.relu(self.fingerprint_fusion_proj(fingerprints))
        out = F.relu(self.fusion(torch.cat([cross_out, fp_fused], dim=1)))
        return out


class ExpertModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.gate(x)


class MoE_MultiTaskModel(nn.Module):
    def __init__(self, num_tasks=9, num_experts=4, hidden_dim=128, expert_hidden=64):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.encoder = GraphTransformerEncoder(
            node_features=74, hidden_dim=hidden_dim, num_layers=3, num_heads=4
        )
        self.experts = nn.ModuleList(
            [
                ExpertModule(hidden_dim, expert_hidden, hidden_dim // 2)
                for _ in range(num_experts)
            ]
        )
        self.gate = GatingNetwork(hidden_dim, num_experts)
        self.task_heads = nn.ModuleList(
            [nn.Linear(hidden_dim // 2, 1) for _ in range(num_tasks)]
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, node_features, adj_matrix, fingerprints):
        z = self.encoder(node_features, adj_matrix, fingerprints)
        z = self.dropout(z)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(z))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_weights = self.gate(z)
        gate_weights = gate_weights.unsqueeze(-1)
        outputs = []
        for task_idx in range(self.num_tasks):
            weighted_expert = torch.sum(expert_outputs * gate_weights, dim=1)
            task_out = self.task_heads[task_idx](weighted_expert)
            outputs.append(task_out)
        return torch.cat(outputs, dim=1)


# Initialize model
missing_rates = [
    287 / 5326,
    198 / 5326,
    1567 / 5326,
    804 / 5326,
    3169 / 5326,
    3165 / 5326,
    4024 / 5326,
    4351 / 5326,
    5104 / 5326,
]
num_tasks = 9
num_experts = 4
hidden_dim = 128
expert_hidden = 64

model = MoE_MultiTaskModel(
    num_tasks=num_tasks,
    num_experts=num_experts,
    hidden_dim=hidden_dim,
    expert_hidden=expert_hidden,
).to(device)


# Loss function with uncertainty weighting
class TaskWeightedMAELoss(nn.Module):
    def __init__(self, num_tasks, missing_rates):
        super().__init__()
        self.num_tasks = num_tasks
        task_weights = 1.0 - torch.tensor(missing_rates, dtype=torch.float32)
        task_weights = task_weights / task_weights.sum() * num_tasks
        self.register_buffer("task_weights", task_weights)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, pred, target, mask):
        task_losses = []
        for i in range(self.num_tasks):
            task_mask = mask[:, i]
            if task_mask.sum() > 0:
                mae = torch.abs(pred[:, i] - target[:, i]) * task_mask.float()
                task_loss = mae.sum() / task_mask.sum()
            else:
                task_loss = torch.tensor(0.0, device=pred.device)
            task_losses.append(task_loss)
        task_losses = torch.stack(task_losses)
        precision = torch.exp(-self.log_vars)
        weighted_losses = precision * task_losses + self.log_vars * 0.5
        weighted_losses = weighted_losses * self.task_weights.to(pred.device)
        return weighted_losses.sum()


criterion = TaskWeightedMAELoss(num_tasks=num_tasks, missing_rates=missing_rates).to(
    device
)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

# ========== TRAINING SETUP ==========
batch_size = 32
num_epochs = 200
patience = 20
best_val_score = float("inf")
best_model_state = None
no_improve_epochs = 0

# Create masks for missing values
train_mask = ~np.isnan(y_train)
val_mask = ~np.isnan(y_val)

# Create data loaders
train_dataset = TensorDataset(
    train_node_feats,
    train_adj,
    train_fp,
    torch.FloatTensor(y_train_imputed),
    torch.BoolTensor(train_mask),
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)

val_dataset = TensorDataset(
    val_node_feats,
    val_adj,
    val_fp,
    torch.FloatTensor(y_val_imputed),
    torch.BoolTensor(val_mask),
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

# ========== TRAINING LOOP ==========
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for node_feats, adj, fp, targets, mask in train_loader:
        node_feats, adj, fp, targets, mask = (
            node_feats.to(device),
            adj.to(device),
            fp.to(device),
            targets.to(device),
            mask.to(device),
        )
        optimizer.zero_grad()
        predictions = model(node_feats, adj, fp)
        loss = criterion(predictions, targets, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    scheduler.step()

    # Validation
    model.eval()
    val_preds_list, val_targets_list, val_masks_list = [], [], []
    with torch.no_grad():
        for node_feats, adj, fp, targets, mask in val_loader:
            node_feats, adj, fp, targets, mask = (
                node_feats.to(device),
                adj.to(device),
                fp.to(device),
                targets.to(device),
                mask.to(device),
            )
            predictions = model(node_feats, adj, fp)
            val_preds_list.append(predictions.cpu().numpy())
            val_targets_list.append(targets.cpu().numpy())
            val_masks_list.append(mask.cpu().numpy())
    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)
    val_masks = np.concatenate(val_masks_list, axis=0)

    # Compute MA-RAE
    mae_per_target = np.zeros(num_tasks)
    range_per_target = np.zeros(num_tasks)
    for i in range(num_tasks):
        task_mask = val_masks[:, i]
        if task_mask.sum() > 0:
            mae_per_target[i] = np.mean(
                np.abs(val_preds[task_mask, i] - val_targets[task_mask, i])
            )
            range_per_target[i] = np.max(val_targets[task_mask, i]) - np.min(
                val_targets[task_mask, i]
            )
    rae_per_target = np.where(
        range_per_target > 0, mae_per_target / range_per_target, np.nan
    )
    val_score = np.nanmean(rae_per_target)

    avg_loss = total_loss / num_batches
    print(
        f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val MA-RAE: {val_score:.4f}"
    )

    if val_score < best_val_score:
        best_val_score = val_score
        torch.save(model.state_dict(), "./working/best_model.pt")
        no_improve_epochs = 0
        print(f"  -> New best model saved (MA-RAE: {val_score:.4f})")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model
model.load_state_dict(torch.load("./working/best_model.pt"))
model.to(device)
model.eval()

# ========== FINAL VALIDATION SCORE ==========
print("\nComputing final validation score...")
val_preds_list = []
with torch.no_grad():
    for (
        node_feats,
        adj,
        fp,
        _,
        _,
    ) in val_loader:
        node_feats, adj, fp = node_feats.to(device), adj.to(device), fp.to(device)
        predictions = model(node_feats, adj, fp)
        val_preds_list.append(predictions.cpu().numpy())
val_preds_final = np.concatenate(val_preds_list, axis=0)

mae_per_target = np.zeros(num_tasks)
range_per_target = np.zeros(num_tasks)
for i in range(num_tasks):
    task_mask = val_masks[:, i]
    if task_mask.sum() > 0:
        mae_per_target[i] = np.mean(
            np.abs(val_preds_final[task_mask, i] - val_targets[task_mask, i])
        )
        range_per_target[i] = np.max(val_targets[task_mask, i]) - np.min(
            val_targets[task_mask, i]
        )
rae_per_target = np.where(
    range_per_target > 0, mae_per_target / range_per_target, np.nan
)
final_val_score = np.nanmean(rae_per_target)

print(f"Per-target MAE: {mae_per_target}")
print(f"Per-target RAE: {rae_per_target}")

# ========== TEST INFERENCE ==========
print("\nGenerating test predictions...")
test_dataset = TensorDataset(test_node_feats, test_adj, test_fp)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

test_predictions = []
with torch.no_grad():
    for node_feats, adj, fp in test_loader:
        node_feats, adj, fp = node_feats.to(device), adj.to(device), fp.to(device)
        predictions = model(node_feats, adj, fp)
        test_predictions.append(predictions.cpu().numpy())
test_preds = np.concatenate(test_predictions, axis=0)

# ========== POST-PROCESSING ==========
# Inverse log transformation for targets that were log-transformed
for i in range(num_tasks):
    if log_transformed_mask[i]:
        test_preds[:, i] = np.expm1(test_preds[:, i])
        test_preds[:, i] = np.maximum(test_preds[:, i], 0)

# Clip predictions to reasonable ranges based on training data (use original unlogged data for log-transformed targets)
for i in range(num_tasks):
    task_data = train_df[target_cols[i]].iloc[train_idx].values
    valid_data = task_data[~np.isnan(task_data)]
    if len(valid_data) > 0:
        # For log-transformed targets, clip the inverse-transformed predictions using original scale
        if log_transformed_mask[i]:
            lower = np.percentile(valid_data, 1)
            upper = np.percentile(valid_data, 99)
        else:
            lower = np.percentile(valid_data, 1)
            upper = np.percentile(valid_data, 99)
        test_preds[:, i] = np.clip(test_preds[:, i], lower, upper)

# ========== CREATE SUBMISSION ==========
submission_df = pd.DataFrame(
    {
        "Molecule Name": test_molecule_names,
        "LogD": test_preds[:, 0],
        "KSOL": test_preds[:, 1],
        "HLM CLint": test_preds[:, 2],
        "MLM CLint": test_preds[:, 3],
        "Caco-2 Permeability Papp A>B": test_preds[:, 4],
        "Caco-2 Permeability Efflux": test_preds[:, 5],
        "MPPB": test_preds[:, 6],
        "MBPB": test_preds[:, 7],
        "MGMB": test_preds[:, 8],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Test predictions shape: {test_preds.shape}")
print(f"Final Validation Score: {final_val_score:.4f}")