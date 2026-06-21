import os
os.sched_setaffinity(0, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.model_selection import StratifiedShuffleSplit
import os
import sys
import warnings
import json

warnings.filterwarnings("ignore")

# Disable DGL's problematic imports
os.environ["DGLBACKEND"] = "pytorch"
os.environ["DGL_DISABLE_GRAPHBOOT"] = "1"
try:
    import dgl
    from dgl.nn import GraphConv
except (ImportError, ModuleNotFoundError):
    dgl = None
    GraphConv = None

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
MODEL_CONFIG = {
    "node_feat_dim": 50,
    "hidden_dim": 256,
    "num_tasks": 9,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "total_steps": 5000,
    "warmup_steps": 500,
    "gate_reg_weight": 0.01,
}

TARGET_COLS = [
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


# ---------------------------------------------------------------
# Molecular fingerprint + MLP baseline (no DGL dependency)
# ---------------------------------------------------------------
def compute_graph_features(smiles_list, node_feat_dim=74):
    """Compute graph features (node features, edge indices, edge features) for a list of SMILES.

    Node features (74-dim):
      0-9: atomic number one-hot (H to Ne, 10 atoms)
      10: degree (scalar)
      11: formal charge (scalar)
      12-18: hybridization one-hot (SP, SP2, SP3, SP3D, SP3D2, OTHER)
      19: chirality (0/1)
      20-23: number of hydrogens (0 to 3+)
      24: implicit valence (scalar)
      25: aromatic flag (0/1)
      26: ring membership (0/1)
      27-73: more features (filled with atomic mass, etc.)

    Edge features (6-dim):
      0-3: bond type one-hot (SINGLE, DOUBLE, TRIPLE, AROMATIC)
      4: conjugated (0/1)
      5: in ring (0/1)
    """
    node_feat_list = []
    edge_idx_list = []
    edge_attr_list = []
    valid_mask = []

    # Atom feature indices (total 74)
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Return zero features for invalid molecule
            node_feat_list.append(torch.zeros((1, node_feat_dim), dtype=torch.float))
            edge_idx_list.append(torch.zeros((2, 0), dtype=torch.long))
            edge_attr_list.append(torch.zeros((0, 6), dtype=torch.float))
            valid_mask.append(False)
            continue

        # Atom features
        num_atoms = mol.GetNumAtoms()
        atom_feats = []
        for atom in mol.GetAtoms():
            feats = []
            # Atomic number one-hot (H to Ne: 1-10)
            atomic_num = atom.GetAtomicNum()
            atomic_one_hot = [0.0] * 10
            if 1 <= atomic_num <= 10:
                atomic_one_hot[atomic_num - 1] = 1.0
            else:
                atomic_one_hot[9] = 1.0  # fallback for larger atoms
            feats.extend(atomic_one_hot)

            # Degree (normalized)
            feats.append(float(atom.GetDegree()) / 6.0)

            # Formal charge
            feats.append(float(atom.GetFormalCharge()) / 2.0)

            # Hybridization
            hybrid = atom.GetHybridization()
            hybrid_map = {
                Chem.rdchem.HybridizationType.SP: 0,
                Chem.rdchem.HybridizationType.SP2: 1,
                Chem.rdchem.HybridizationType.SP3: 2,
                Chem.rdchem.HybridizationType.SP3D: 3,
                Chem.rdchem.HybridizationType.SP3D2: 4,
            }
            hybrid_one_hot = [0.0] * 6
            idx = hybrid_map.get(hybrid, 5)
            hybrid_one_hot[idx] = 1.0
            feats.extend(hybrid_one_hot)

            # Chirality
            feats.append(
                1.0
                if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
                else 0.0
            )

            # Number of hydrogens (0 to 3+, binned)
            total_h = atom.GetTotalNumHs()
            h_one_hot = [0.0] * 4
            if total_h <= 3:
                h_one_hot[total_h] = 1.0
            else:
                h_one_hot[3] = 1.0
            feats.extend(h_one_hot)

            # Implicit valence (normalized)
            feats.append(float(atom.GetImplicitValence()) / 4.0)

            # Aromatic flag
            feats.append(1.0 if atom.GetIsAromatic() else 0.0)

            # Ring membership
            feats.append(1.0 if atom.IsInRing() else 0.0)

            # Fill remaining dimensions with atomic mass (normalized) and other info
            remaining_dim = node_feat_dim - len(feats)
            # Use atomic mass, van der Waals radius proxy, etc.
            extra_feats = []
            extra_feats.append(float(atom.GetMass()) / 200.0)  # mass normalized
            extra_feats.append(
                float(atom.GetDegree()) / 8.0
            )  # degree again but with different scale
            extra_feats.append(float(atom.GetExplicitValence() / 6.0))
            # Fill rest with small constants or zeros
            extra_feats.extend([0.0] * (remaining_dim - len(extra_feats)))
            feats.extend(extra_feats)

            atom_feats.append(feats)

        # Ensure we have exactly node_feat_dim features per atom
        atom_feats = np.array(atom_feats, dtype=np.float32)
        if atom_feats.shape[1] != node_feat_dim:
            if atom_feats.shape[1] < node_feat_dim:
                pad = np.zeros(
                    (atom_feats.shape[0], node_feat_dim - atom_feats.shape[1]),
                    dtype=np.float32,
                )
                atom_feats = np.concatenate([atom_feats, pad], axis=1)
            else:
                atom_feats = atom_feats[:, :node_feat_dim]

        # Edge features and indices
        edge_list = []
        edge_feats = []

        # Add self-loop for each node (for message passing)
        for i in range(num_atoms):
            edge_list.append([i, i])
            # Self-loop edge features: single bond, non-conjugated, non-ring
            edge_feats.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Bond type one-hot
            bond_type = bond.GetBondType()
            type_one_hot = [0.0] * 4
            bond_map = {
                Chem.rdchem.BondType.SINGLE: 0,
                Chem.rdchem.BondType.DOUBLE: 1,
                Chem.rdchem.BondType.TRIPLE: 2,
                Chem.rdchem.BondType.AROMATIC: 3,
            }
            idx = bond_map.get(bond_type, 0)
            type_one_hot[idx] = 1.0
            feats_bond = type_one_hot.copy()

            # Is conjugated
            feats_bond.append(1.0 if bond.GetIsConjugated() else 0.0)

            # Is in ring
            feats_bond.append(1.0 if bond.IsInRing() else 0.0)

            # Add both directions
            edge_list.append([i, j])
            edge_list.append([j, i])
            edge_feats.append(feats_bond)
            edge_feats.append(feats_bond)

        if len(edge_list) == 0:
            edge_tensor = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.zeros((0, 6), dtype=torch.float)
        else:
            edge_tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_feats, dtype=torch.float)

        node_feat_list.append(torch.tensor(atom_feats, dtype=torch.float))
        edge_idx_list.append(edge_tensor)
        edge_attr_list.append(edge_attr_tensor)
        valid_mask.append(True)

    return node_feat_list, edge_idx_list, edge_attr_list, np.array(valid_mask)


class MolecularDataset(torch.utils.data.Dataset):
    def __init__(self, node_feat_list, edge_idx_list, edge_attr_list, y_matrix=None):
        self.node_feats = node_feat_list
        self.edge_idxs = edge_idx_list
        self.edge_attrs = edge_attr_list
        if y_matrix is not None:
            self.targets = torch.tensor(
                np.nan_to_num(y_matrix, nan=0.0), dtype=torch.float
            )
            self.masks = torch.tensor(~np.isnan(y_matrix), dtype=torch.float)
        else:
            self.targets = torch.zeros((len(node_feat_list), 0), dtype=torch.float)
            self.masks = torch.zeros((len(node_feat_list), 0), dtype=torch.float)

    def __len__(self):
        return len(self.node_feats)

    def __getitem__(self, idx):
        return (
            self.node_feats[idx],
            self.edge_idxs[idx],
            self.edge_attrs[idx],
            self.targets[idx],
            self.masks[idx],
        )


# ---------------------------------------------------------------
# Lightweight MLP Model (attention-based)
# ---------------------------------------------------------------
class GINEConvLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) convolution layer."""

    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr):
        """
        x: (N, node_feat_dim)
        edge_index: (2, E)
        edge_attr: (E, edge_feat_dim)
        """
        N = x.size(0)
        row, col = edge_index

        # Compute edge contributions: message = edge_mlp(edge_attr) * node_mlp(x[col])
        edge_feat = self.edge_mlp(edge_attr)  # (E, hidden_dim)
        target_feat = self.node_mlp(x[col])  # (E, hidden_dim)
        messages = edge_feat * target_feat  # (E, hidden_dim)

        # Aggregate messages by target node (row index = target)
        out = torch.zeros(N, messages.size(1), device=x.device)
        out.index_add_(0, row, messages)

        # Combine with self-loop (epsilon * original node features)
        out = (1 + self.eps) * self.node_mlp(x) + out
        return out


class MultiTaskGNN(nn.Module):
    """Graph Neural Network with GINEConv layers for multi-task prediction."""

    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim=6,
        hidden_dim=128,
        num_tasks=9,
        dropout=0.2,
        num_layers=3,
    ):
        super().__init__()
        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINEConvLayer(hidden_dim, edge_feat_dim, hidden_dim))

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Task-specific heads
        self.heads = nn.ModuleList()
        for _ in range(num_tasks):
            head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.heads.append(head)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, node_feats, edge_indices, edge_attrs, batch):
        """
        node_feats: list of tensors (num_nodes_i, node_feat_dim)
        edge_indices: list of tensors (2, num_edges_i)
        edge_attrs: list of tensors (num_edges_i, edge_feat_dim)
        batch: tensor of shape (num_total_nodes,) assigning each node to a graph
        """
        # Concatenate all graph data into a single large batch
        x = torch.cat(node_feats, dim=0)  # (total_nodes, node_feat_dim)

        # Offset edge indices for concatenated graphs
        offset = 0
        edge_idx_list = []
        edge_attr_list = []
        for i, (ei, ea) in enumerate(zip(edge_indices, edge_attrs)):
            edge_idx_list.append(ei + offset)
            edge_attr_list.append(ea)
            offset += node_feats[i].size(0)

        edge_index = (
            torch.cat(edge_idx_list, dim=1)
            if edge_idx_list
            else torch.zeros((2, 0), dtype=torch.long, device=x.device)
        )
        edge_attr = (
            torch.cat(edge_attr_list, dim=0)
            if edge_attr_list
            else torch.zeros((0, 6), device=x.device)
        )

        # Initial embedding
        x = self.node_embed(x)  # (total_nodes, hidden_dim)

        # Message passing layers
        for conv in self.convs:
            x_new = conv(x, edge_index, edge_attr)
            x_new = self.norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # residual

        # Global mean pooling per graph
        num_graphs = batch.max().item() + 1
        pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
        pooled.index_add_(0, batch, x)
        counts = torch.bincount(batch, minlength=num_graphs).float().unsqueeze(1)
        pooled = pooled / (counts + 1e-8)

        # Task-specific predictions
        predictions = []
        for head in self.heads:
            pred = head(pooled)
            predictions.append(pred)
        return torch.cat(predictions, dim=1)


# ---------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------
class MaskedMAELoss(nn.Module):
    def __init__(self, gate_reg_weight=0.01):
        super().__init__()
        self.gate_reg_weight = gate_reg_weight

    def forward(self, pred, target, mask):
        abs_error = torch.abs(pred - target)
        masked_error = abs_error * mask
        task_loss = masked_error.sum(dim=0) / (mask.sum(dim=0) + 1e-8)
        task_weights = 1.0 / (mask.mean(dim=0) + 0.1)
        task_weights = task_weights / task_weights.mean()
        weighted_loss = (task_loss * task_weights).mean()
        return weighted_loss


# ---------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------
def train_model_gnn(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=200,
    device="cuda",
    patience=20,
):
    best_val_score = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            node_feats, edge_idxs, edge_attrs, targets, masks = batch_data

            # Build batch tensor
            batch_list = []
            for i, nf in enumerate(node_feats):
                batch_list.append(torch.full((nf.size(0),), i, dtype=torch.long))
            batch = torch.cat(batch_list, dim=0).to(device)

            # Move data to device
            node_feats = [nf.to(device) for nf in node_feats]
            edge_idxs = [ei.to(device) for ei in edge_idxs]
            edge_attrs = [ea.to(device) for ea in edge_attrs]
            targets = targets.to(device)
            masks = masks.to(device)

            predictions = model(node_feats, edge_idxs, edge_attrs, batch)
            loss = criterion(predictions, targets, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / max(num_batches, 1)
        val_score = evaluate_model_gnn(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{num_epochs} | Loss: {avg_train_loss:.4f} | Val MA-RAE: {val_score:.4f} | LR: {lr:.6f}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    model.load_state_dict(best_model_state)
    model = model.to(device)
    return model, best_val_score


def evaluate_model_gnn(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch_data in data_loader:
            node_feats, edge_idxs, edge_attrs, targets, masks = batch_data

            batch_list = []
            for i, nf in enumerate(node_feats):
                batch_list.append(torch.full((nf.size(0),), i, dtype=torch.long))
            batch = torch.cat(batch_list, dim=0).to(device)

            node_feats = [nf.to(device) for nf in node_feats]
            edge_idxs = [ei.to(device) for ei in edge_idxs]
            edge_attrs = [ea.to(device) for ea in edge_attrs]
            targets = targets.to(device)
            masks = masks.to(device)

            predictions = model(node_feats, edge_idxs, edge_attrs, batch)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    num_tasks = targets.shape[1]
    task_rae = []

    for t in range(num_tasks):
        mask_t = masks[:, t] > 0.5
        if mask_t.sum() > 0:
            pred_t = preds[mask_t, t]
            target_t = targets[mask_t, t]
            mae = np.mean(np.abs(pred_t - target_t))
            mean_abs_dev = np.mean(np.abs(target_t - np.mean(target_t)))
            if mean_abs_dev > 1e-10:
                rae = mae / mean_abs_dev
            else:
                rae = 1.0
            task_rae.append(rae)

    ma_rae = np.mean(task_rae) if task_rae else 1.0
    return ma_rae


def collate_fn_graph(batch):
    """Collate function for graph data with variable sizes."""
    node_feats = [item[0] for item in batch]
    edge_idxs = [item[1] for item in batch]
    edge_attrs = [item[2] for item in batch]
    targets = torch.stack([item[3] for item in batch])
    masks = torch.stack([item[4] for item in batch])
    return node_feats, edge_idxs, edge_attrs, targets, masks


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")

    valid_train_mask = train_df["SMILES"].notna().values
    train_smiles = train_df["SMILES"].values[valid_train_mask]
    train_names = train_df["Molecule Name"].values[valid_train_mask]

    test_valid_mask = test_df["SMILES"].notna().values
    test_smiles = test_df["SMILES"].values[test_valid_mask]
    test_names = test_df["Molecule Name"].values[test_valid_mask]

    # Build target matrix WITHOUT log transformation first
    y_raw = train_df[TARGET_COLS].values[valid_train_mask].copy()

    # Train/validation split FIRST (before log transformation)
    missing_counts = np.isnan(y_raw).sum(axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in sss.split(np.zeros(len(missing_counts)), missing_counts):
        train_smiles_split = [train_smiles[i] for i in train_idx]
        val_smiles_split = [train_smiles[i] for i in val_idx]
        y_train_split = y_raw[train_idx].copy()
        y_val_split = y_raw[val_idx].copy()
        train_names_split = train_names[train_idx]
        val_names_split = train_names[val_idx]
        break

    # Log transformation applied SEPARATELY to train and val
    log_transform_cols = [1, 2, 3, 4, 5, 6, 7, 8]
    for col_idx in log_transform_cols:
        col_data_train = y_train_split[:, col_idx]
        valid_mask_col_train = ~np.isnan(col_data_train)
        y_train_split[valid_mask_col_train, col_idx] = np.log1p(
            col_data_train[valid_mask_col_train]
        )

        col_data_val = y_val_split[:, col_idx]
        valid_mask_col_val = ~np.isnan(col_data_val)
        y_val_split[valid_mask_col_val, col_idx] = np.log1p(
            col_data_val[valid_mask_col_val]
        )

    print(f"Train samples: {len(train_smiles_split)}")
    print(f"Validation samples: {len(val_smiles_split)}")
    print(f"Test samples: {len(test_smiles)}")

    # Compute graph features
    print("Computing graph features (nodes, edges) for GNN...")
    train_nf, train_ei, train_ea, train_valid = compute_graph_features(
        train_smiles_split
    )
    val_nf, val_ei, val_ea, val_valid = compute_graph_features(val_smiles_split)
    test_nf, test_ei, test_ea, test_valid = compute_graph_features(test_smiles)

    # Filter out invalid molecules from training (keep them in val/test but with caution)
    # For simplicity, we keep all and rely on valid_mask
    print(f"  Training: {sum(train_valid)} valid molecules")
    print(f"  Validation: {sum(val_valid)} valid molecules")
    print(f"  Test: {sum(test_valid)} valid molecules")

    NODE_FEAT_DIM = train_nf[0].size(1) if train_valid.any() else 74
    print(f"Node feature dimension: {NODE_FEAT_DIM}")

    # Build model with GNN
    model = MultiTaskGNN(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=6,
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        num_tasks=MODEL_CONFIG["num_tasks"],
        dropout=MODEL_CONFIG["dropout"],
        num_layers=3,
    )
    model = model.to(device)

    criterion = MaskedMAELoss(gate_reg_weight=0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_CONFIG["lr"],
        weight_decay=MODEL_CONFIG["weight_decay"],
    )

    # Create datasets and dataloaders
    train_dataset = MolecularDataset(
        train_nf, train_ei, train_ea, y_matrix=y_train_split
    )
    val_dataset = MolecularDataset(val_nf, val_ei, val_ea, y_matrix=y_val_split)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,  # Smaller batch size due to variable-sized graphs
        shuffle=True,
        collate_fn=collate_fn_graph,
        num_workers=2,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn_graph,
        num_workers=2,
    )

    # Estimate total steps using a cosine schedule with fixed total
    steps_per_epoch = len(train_loader)
    total_scheduler_steps = 200 * steps_per_epoch  # num_epochs * steps_per_epoch
    warmup_scheduler_steps = int(0.1 * total_scheduler_steps)  # 10% warmup

    def lr_lambda(
        current_step, warmup=warmup_scheduler_steps, total=total_scheduler_steps
    ):
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        progress = float(current_step - warmup) / float(max(1, total - warmup))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(
            step, warmup_scheduler_steps, total_scheduler_steps
        ),
    )

    # Train
    print("Starting GNN training...")
    model, best_val_score = train_model_gnn(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=200,
        device=device,
        patience=20,
    )
    print(f"Best validation MA-RAE: {best_val_score:.4f}")

    # Test inference
    print("Performing test inference...")
    test_dataset = MolecularDataset(test_nf, test_ei, test_ea)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn_graph,
        num_workers=2,
    )

    model.eval()
    all_test_preds = []
    with torch.no_grad():
        for batch_data in test_loader:
            node_feats, edge_idxs, edge_attrs, _, _ = batch_data

            batch_list = []
            for i, nf in enumerate(node_feats):
                batch_list.append(torch.full((nf.size(0),), i, dtype=torch.long))
            batch = torch.cat(batch_list, dim=0).to(device)

            node_feats = [nf.to(device) for nf in node_feats]
            edge_idxs = [ei.to(device) for ei in edge_idxs]
            edge_attrs = [ea.to(device) for ea in edge_attrs]

            predictions = model(node_feats, edge_idxs, edge_attrs, batch)
            all_test_preds.append(predictions.cpu().numpy())

    test_preds = np.concatenate(all_test_preds, axis=0)

    # Inverse log transform
    for col_idx in log_transform_cols:
        test_preds[:, col_idx] = np.expm1(test_preds[:, col_idx])

    test_preds = np.maximum(test_preds, 0.0)

    # Create submission
    submission = pd.DataFrame(
        {
            "Molecule Name": test_names,
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
    submission.to_csv("./submission/submission_a896df1e7039447883f4b063eeeef5a5.csv", index=False)
    print(f"Submission saved to ./submission/submission_a896df1e7039447883f4b063eeeef5a5.csv")
    print(f"Submission shape: {submission.shape}")

    score = best_val_score
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
