import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.EState import EState
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ===== Step 1: Data Processing and Feature Engineering =====


def smiles_to_graph(smiles):
    """Convert SMILES string to a dictionary of graph data for PyTorch Geometric."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
    except:
        return None

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        num_hs = atom.GetTotalNumHs()
        is_aromatic = int(atom.GetIsAromatic())
        # One-hot encode atomic number up to 100 and normalize
        feat = [
            atomic_num / 100.0,
            degree / 8.0,
            formal_charge / 5.0,
            num_hs / 8.0,
            is_aromatic,
        ]
        atom_features.append(feat)

    # Edge indices and features
    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Undirected
        bond_type = bond.GetBondTypeAsDouble()
        is_ring = int(bond.IsInRing())
        feat = [bond_type / 4.0, is_ring]
        edge_features.append(feat)
        edge_features.append(feat)

    if len(edge_index) == 0:
        # Handle single-atom molecules
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_features = torch.zeros((0, 2), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float)

    x = torch.tensor(atom_features, dtype=torch.float)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_features,
        "num_nodes": x.shape[0],
    }


def augment_smiles(smiles, num_augmentations=5):
    """Generate random SMILES strings from a molecule for augmentation."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
    except:
        return []

    augmented = []
    for _ in range(num_augmentations):
        try:
            aug_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            augmented.append(aug_smiles)
        except:
            pass

    if not augmented:
        augmented = [smiles]

    return augmented


def build_molecule_graph_dict(name, smiles, is_training=False, aug_factor=5):
    """Build a dictionary mapping molecule name to list of graph dicts.
    For training, include augmented variants; for validation/test, only canonical form.
    """
    graph_dict = {}
    # Canonical graph (always included)
    canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    canonical_graph = smiles_to_graph(canonical_smiles)
    if canonical_graph is None:
        return None

    graphs = [canonical_graph]

    if is_training:
        augmented_smiles_list = augment_smiles(smiles, num_augmentations=aug_factor)
        for aug_smiles in augmented_smiles_list:
            aug_graph = smiles_to_graph(aug_smiles)
            if aug_graph is not None:
                graphs.append(aug_graph)

    graph_dict[name] = graphs
    return graph_dict


def preprocess_data():
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
    print(f"Loaded training data: {train_df.shape}")
    print(f"Loaded test data: {test_df.shape}")

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

    # Log transform non-LogD targets will be done after data split
    train_targets = train_df[target_cols].copy()

    # Build graph dictionaries
    print("Building molecular graphs for training data...")
    train_graphs = {}
    for idx, row in train_df.iterrows():
        graph_dict = build_molecule_graph_dict(row["Molecule Name"], row["SMILES"], is_training=True, aug_factor=5)
        if graph_dict is not None:
            train_graphs.update(graph_dict)

    print("Building molecular graphs for test data...")
    test_graphs_list = []
    test_names = []
    for idx, row in test_df.iterrows():
        graph_dict = build_molecule_graph_dict(row["Molecule Name"], row["SMILES"], is_training=False)
        if graph_dict is not None:
            test_graphs_list.append(graph_dict[row["Molecule Name"]][0])
            test_names.append(row["Molecule Name"])
    print(f"Test graphs built: {len(test_graphs_list)}")

    # Train/validation split by molecule names
    train_molecule_names = list(train_graphs.keys())
    train_idx, val_idx = train_test_split(
        np.arange(len(train_molecule_names)), test_size=0.15, random_state=42
    )

    train_names_split = [train_molecule_names[i] for i in train_idx]
    val_names_split = [train_molecule_names[i] for i in val_idx]

    train_graph_dict = {name: train_graphs[name] for name in train_names_split}
    val_graph_dict = {name: train_graphs[name] for name in val_names_split}

    # Apply log transform only after splitting to avoid data leakage
    y_train_raw = train_targets.loc[train_df["Molecule Name"].isin(train_names_split)].values
    y_val_raw = train_targets.loc[train_df["Molecule Name"].isin(val_names_split)].values

    y_train = y_train_raw.copy()
    y_val = y_val_raw.copy()
    for col_idx, col in enumerate(target_cols):
        if col != "LogD":
            mask_train = ~np.isnan(y_train[:, col_idx])
            mask_val = ~np.isnan(y_val[:, col_idx])
            if mask_train.sum() > 0:
                y_train[mask_train, col_idx] = np.log10(y_train[mask_train, col_idx].clip(min=1e-10))
            if mask_val.sum() > 0:
                y_val[mask_val, col_idx] = np.log10(y_val[mask_val, col_idx].clip(min=1e-10))

    train_mask = ~np.isnan(y_train)
    val_mask = ~np.isnan(y_val)

    processed_data = {
        "train_graph_dict": train_graph_dict,
        "val_graph_dict": val_graph_dict,
        "test_graphs": test_graphs_list,
        "y_train": y_train,
        "y_val": y_val,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "train_names": np.array(train_names_split),
        "val_names": np.array(val_names_split),
        "test_names": np.array(test_names),
        "target_cols": target_cols,
        "log_transform_cols": log_transform_cols,
    }

    print(f"\nFinal dataset sizes:")
    print(f"  Train molecules: {len(train_names_split)}")
    print(f"  Validation molecules: {len(val_names_split)}")
    print(f"  Test molecules: {len(test_names)}")
    return processed_data


processed_data = preprocess_data()
train_graph_dict = processed_data["train_graph_dict"]
val_graph_dict = processed_data["val_graph_dict"]
test_graphs = processed_data["test_graphs"]
y_train = processed_data["y_train"]
y_val = processed_data["y_val"]
train_mask = processed_data["train_mask"]
val_mask = processed_data["val_mask"]
train_names = processed_data["train_names"]
val_names = processed_data["val_names"]
test_names = processed_data["test_names"]
target_cols = processed_data["target_cols"]
log_transform_cols = processed_data["log_transform_cols"]

# Ensure validation graphs only contain canonical (non-augmented) graphs
val_graph_dict_clean = {}
for name in val_names:
    val_graph_dict_clean[name] = [val_graph_dict[name][0]]
val_graph_dict = val_graph_dict_clean

# ===== Step 2: Model Design =====


from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge
from torch_geometric.data import Data, Batch

class GINModel(nn.Module):
    def __init__(self, atom_feat_dim=5, edge_feat_dim=2, hidden_dim=128, num_targets=9, dropout=0.2):
        super().__init__()
        # GIN layers with learnable MLP
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(atom_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            train_eps=True,
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            train_eps=True,
        )
        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            train_eps=True,
        )
        self.conv4 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            train_eps=True,
        )

        # Jumping Knowledge aggregation (concatenate all layer outputs)
        self.jk = JumpingKnowledge(mode='cat')

        # After JK: 4 layers * hidden_dim = 512
        self.dropout = nn.Dropout(dropout)

        # Multi-task prediction heads
        self.target_heads = nn.ModuleList()
        for _ in range(num_targets):
            self.target_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 4, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_dim * 2, 1),
                )
            )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # GIN layers
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h4 = self.conv4(h3, edge_index)

        # Jumping Knowledge: collect all layer outputs
        node_features = [h1, h2, h3, h4]
        h_jk = self.jk(node_features)

        # Global mean pooling
        graph_embedding = global_mean_pool(h_jk, batch)
        graph_embedding = self.dropout(graph_embedding)

        # Multi-task heads
        outputs = [head(graph_embedding) for head in self.target_heads]
        return torch.cat(outputs, dim=1)


def compute_masked_mse_loss(predictions, targets, mask):
    # predictions: (batch_size, num_targets), targets: (batch_size, num_targets), mask: (batch_size, num_targets)
    batch_size = predictions.shape[0]
    num_targets = predictions.shape[1]
    # Ensure targets and mask are 2D
    if targets.dim() == 1:
        targets = targets.view(batch_size, -1)
    if mask.dim() == 1:
        mask = mask.view(batch_size, -1)

    # Safety: ensure mask and predictions shapes match
    if mask.shape != predictions.shape:
        # If mask got flattened to (batch_size,) or (batch_size*num_targets,), reshape
        if mask.dim() == 1:
            if mask.numel() == batch_size:
                # Each sample has same mask for all targets
                mask = mask.unsqueeze(1).expand(-1, num_targets)
            elif mask.numel() == batch_size * num_targets:
                mask = mask.view(batch_size, num_targets)
            else:
                mask = torch.ones(batch_size, num_targets, device=mask.device).bool()

    total_valid = mask.sum().float()
    loss = 0.0
    valid_count = 0
    for i in range(num_targets):
        valid_mask = mask[:, i]
        if valid_mask.sum() > 0:
            weight = total_valid / (valid_mask.sum().float() * num_targets)
            loss += weight * F.mse_loss(predictions[valid_mask, i], targets[valid_mask, i])
            valid_count += 1
    return loss / max(valid_count, 1)


def compute_ma_rae(predictions, targets, mask):
    n_targets = predictions.shape[1]
    rae_per_target = []
    for i in range(n_targets):
        valid_idx = mask[:, i]
        if valid_idx.sum() > 0:
            y_pred = predictions[valid_idx, i]
            y_true = targets[valid_idx, i]
            y_pred = np.nan_to_num(y_pred, nan=0.0)
            y_true = np.nan_to_num(y_true, nan=0.0)
            abs_error = np.abs(y_pred - y_true)
            mean_true = np.mean(y_true)
            baseline = np.abs(y_true - mean_true)
            baseline = np.clip(baseline, a_min=1e-10, a_max=None)
            rae = np.mean(abs_error / baseline)
            rae_per_target.append(rae)
    return np.mean(rae_per_target) if rae_per_target else 0.0


# ===== Step 3: Training and Evaluation =====

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Build PyTorch Geometric Data objects from molecule graph dictionaries
def build_data_list_from_graph_dict(graph_dict, y_dict=None, mask_dict=None):
    data_list = []
    for i, (mol_name, graphs) in enumerate(graph_dict.items()):
        # Always use canonical graph (first one) for validation/test
        chosen_graph = graphs[0]
        data = Data(
            x=chosen_graph["x"],
            edge_index=chosen_graph["edge_index"],
            edge_attr=chosen_graph["edge_attr"],
        )
        if y_dict is not None:
            data.y = torch.FloatTensor(y_dict[mol_name])
            data.mask = torch.BoolTensor(mask_dict[mol_name])
        data_list.append(data)
    return data_list

# Build target and mask dictionaries for training/validation
train_y_dict = {}
train_mask_dict = {}
for i, name in enumerate(train_names):
    train_y_dict[name] = y_train[i]
    train_mask_dict[name] = train_mask[i]

val_y_dict = {}
val_mask_dict = {}
for i, name in enumerate(val_names):
    val_y_dict[name] = y_val[i]
    val_mask_dict[name] = val_mask[i]

# Custom collate function for graph batches
def collate_graphs(batch_data):
    # Handle mask and y manually BEFORE Batch.from_data_list to prevent flattening
    masks = []
    ys = []
    for data in batch_data:
        if hasattr(data, 'mask') and data.mask is not None:
            masks.append(data.mask)
        if hasattr(data, 'y') and data.y is not None:
            ys.append(data.y)

    batch = Batch.from_data_list(batch_data)

    if masks:
        # Stack masks manually to maintain 2D shape: (batch_size, num_targets)
        batch.mask = torch.stack(masks, dim=0)
    if ys:
        batch.y = torch.stack(ys, dim=0)

    return batch

# Build data loaders with graph data
train_dataset = [(name, graphs) for name, graphs in train_graph_dict.items()]
val_dataset = [(name, graphs) for name, graphs in val_graph_dict.items()]

# Custom dataset class for graph data
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_dict, y_dict=None, mask_dict=None, is_training=True):
        self.graph_dict = graph_dict
        self.y_dict = y_dict
        self.mask_dict = mask_dict
        self.is_training = is_training
        self.mol_names = list(graph_dict.keys())

    def __len__(self):
        return len(self.mol_names)

    def __getitem__(self, idx):
        mol_name = self.mol_names[idx]
        graphs = self.graph_dict[mol_name]
        # Randomly select one augmented variant during training
        if self.is_training:
            chosen_graph = np.random.choice(graphs)
        else:
            chosen_graph = graphs[0]
        data = Data(
            x=chosen_graph["x"],
            edge_index=chosen_graph["edge_index"],
            edge_attr=chosen_graph["edge_attr"],
        )
        if self.y_dict is not None:
            data.y = torch.FloatTensor(self.y_dict[mol_name])  # (num_targets,)
            data.mask = torch.BoolTensor(self.mask_dict[mol_name])  # (num_targets,)
        return data

train_dataset = GraphDataset(train_graph_dict, train_y_dict, train_mask_dict, is_training=True)
val_dataset = GraphDataset(val_graph_dict, val_y_dict, val_mask_dict, is_training=False)
test_dataset = test_graphs  # List of graph dicts

# DataLoader with custom collate
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_graphs
)
val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_graphs
)

# Test data loader
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, graph_dicts):
        self.graph_dicts = graph_dicts

    def __len__(self):
        return len(self.graph_dicts)

    def __getitem__(self, idx):
        g = self.graph_dicts[idx]
        return Data(x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"])

test_dataset2 = TestDataset(test_graphs)
test_loader = DataLoader(test_dataset2, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_graphs)

model = GINModel(atom_feat_dim=5, edge_feat_dim=2, hidden_dim=128, num_targets=9, dropout=0.2)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Warmup scheduler: linear warmup to 1e-3 for 10 epochs, then cosine annealing
warmup_epochs = 10
total_epochs = 190
def lambda_lr(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
# Set initial learning rate to 1e-3 (scheduler multiplier will handle warmup)
for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-3

num_epochs = 190
best_val_ma_rae = float("inf")
best_model_state = None
patience = 50
patience_counter = 0

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_batches = 0
    for batch in train_loader:
        batch = batch.to(device)
        targets = batch.y
        mask = batch.mask
        optimizer.zero_grad()
        predictions = model(batch)
        loss = compute_masked_mse_loss(predictions, targets, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1

    scheduler.step()

    model.eval()
    val_predictions = []
    val_targets = []
    val_valid_mask = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            predictions = model(batch)
            val_predictions.append(predictions.cpu().numpy())
            # Ensure y and mask are 2D before converting to numpy
            y_batch = batch.y
            mask_batch = batch.mask
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)
            if mask_batch.dim() == 1:
                mask_batch = mask_batch.unsqueeze(1)
            val_targets.append(y_batch.cpu().numpy())
            val_valid_mask.append(mask_batch.cpu().numpy())

    val_predictions = np.concatenate(val_predictions, axis=0)
    val_targets_np = np.concatenate(val_targets, axis=0)
    val_mask_np = np.concatenate(val_valid_mask, axis=0)

    # Inverse log transform for MA-RAE in original space
    val_preds_orig = val_predictions.copy()
    val_targets_orig = val_targets_np.copy()
    for i_col, col in enumerate(target_cols):
        if col != "LogD":
            val_preds_orig[:, i_col] = 10 ** val_predictions[:, i_col]
            val_targets_orig[:, i_col] = 10 ** val_targets_np[:, i_col]

    val_ma_rae = compute_ma_rae(val_preds_orig, val_targets_orig, val_mask_np)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/train_batches:.4f}, Val MA-RAE: {val_ma_rae:.4f}"
        )

    if val_ma_rae < best_val_ma_rae:
        best_val_ma_rae = val_ma_rae
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model and final validation
if best_model_state is not None:
    model.load_state_dict(best_model_state)
model.eval()

val_predictions = []
val_targets = []
val_valid_mask = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        predictions = model(batch)
        val_predictions.append(predictions.cpu().numpy())
        y_batch = batch.y
        mask_batch = batch.mask
        if y_batch.dim() == 1:
            y_batch = y_batch.unsqueeze(1)
        if mask_batch.dim() == 1:
            mask_batch = mask_batch.unsqueeze(1)
        val_targets.append(y_batch.cpu().numpy())
        val_valid_mask.append(mask_batch.cpu().numpy())
val_predictions = np.concatenate(val_predictions, axis=0)
val_targets_np = np.concatenate(val_targets, axis=0)
val_mask_np = np.concatenate(val_valid_mask, axis=0)

val_preds_orig = val_predictions.copy()
val_targets_orig = val_targets_np.copy()
for i_col, col in enumerate(target_cols):
    if col != "LogD":
        val_preds_orig[:, i_col] = 10 ** val_predictions[:, i_col]
        val_targets_orig[:, i_col] = 10 ** val_targets_np[:, i_col]

final_val_ma_rae = compute_ma_rae(val_preds_orig, val_targets_orig, val_mask_np)
print(f"\nBest validation MA-RAE: {best_val_ma_rae:.4f}")
print(f"Final validation MA-RAE: {final_val_ma_rae:.4f}")

# Test inference
test_predictions = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        predictions = model(batch)
        test_predictions.append(predictions.cpu().numpy())
test_predictions = np.concatenate(test_predictions, axis=0)

# Inverse log transform for submission
for i_col, col in enumerate(target_cols):
    if col != "LogD":
        test_predictions[:, i_col] = 10 ** test_predictions[:, i_col]

submission_df = pd.DataFrame(
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
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

score = final_val_ma_rae
print(f"Final Validation Score: {score}")