import os
os.sched_setaffinity(0, {0, 1, 2, 3, 4, 5, 6})
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import gc
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TARGET_NAMES = [
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

LOG_TARGET_INDICES = [1, 2, 3, 4, 5, 6, 7, 8]  # indices of log-transformed targets


# ========== MOLECULAR GRAPH DATASET ==========
class MolecularGraphDataset(Dataset):
    def __init__(self, smiles_list, targets=None):
        self.smiles_list = smiles_list
        self.targets = targets

    def _smiles_to_graph(self, smiles):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum() / 118.0,
                atom.GetDegree() / 8.0,
                atom.GetFormalCharge() / 3.0 + 0.5,
                atom.GetNumRadicalElectrons() / 4.0,
                atom.GetIsAromatic() * 1.0,
                atom.GetHybridization() % 6 / 5.0,
                atom.GetImplicitValence() / 6.0,
                atom.GetNumImplicitHs() / 4.0,
                atom.IsInRing() * 1.0,
                atom.GetMass() / 200.0,
            ]
            element = atom.GetAtomicNum()
            element_onehot = [0.0] * 11
            element_map = {
                6: 0,
                7: 1,
                8: 2,
                16: 3,
                17: 4,
                9: 5,
                15: 6,
                35: 7,
                53: 8,
                5: 9,
            }
            idx = element_map.get(element, 10)
            element_onehot[idx] = 1.0
            features.extend(element_onehot)

            chiral_tag = atom.GetChiralTag()
            features.extend(
                [
                    float(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW),
                    float(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW),
                    float(chiral_tag == Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
                ]
            )
            atom_features.append(features)

        x = torch.tensor(np.array(atom_features, dtype=np.float32))

        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])

            bond_type = bond.GetBondTypeAsDouble()
            is_aromatic = bond.GetIsAromatic() * 1.0
            is_conjugated = bond.GetIsConjugated() * 1.0
            is_in_ring = bond.IsInRing() * 1.0

            bond_type_oh = [0.0] * 5
            type_map = {1.0: 0, 2.0: 1, 3.0: 2, 1.5: 3}
            idx = type_map.get(bond_type, 4)
            bond_type_oh[idx] = 1.0

            edge_feat = [
                bond_type / 3.0,
                is_aromatic,
                is_conjugated,
                is_in_ring,
                bond.GetStereo() / 3.0,
            ]
            edge_feat.extend(bond_type_oh)
            edge_features.extend([edge_feat, edge_feat])

        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_attr = torch.zeros((1, 11), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features, dtype=np.float32))

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        graph = self._smiles_to_graph(smiles)
        if self.targets is not None:
            target = self.targets[idx]
            mask = ~np.isnan(target)
            target_clean = np.nan_to_num(target, nan=0.0)
            return (
                graph,
                torch.tensor(target_clean, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
            )
        return graph


def collate_graphs(batch):
    if len(batch[0]) == 3:
        graphs, targets, masks = zip(*batch)
        valid_graphs = [g for g in graphs if g is not None]
        if len(valid_graphs) == 0:
            return None, None, None
        batch_data = Batch.from_data_list(valid_graphs)
        valid_targets = torch.stack(
            [t for g, t in zip(graphs, targets) if g is not None]
        )
        valid_masks = torch.stack([m for g, m in zip(graphs, masks) if g is not None])
        return batch_data, valid_targets, valid_masks
    else:
        graphs = batch
        valid_graphs = [g for g in graphs if g is not None]
        if len(valid_graphs) == 0:
            return None
        return Batch.from_data_list(valid_graphs)


# ========== MODEL DEFINITION ==========
class GINEConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 10, out_channels * 2),
            nn.BatchNorm1d(out_channels * 2),
            nn.ReLU(),
            nn.Linear(out_channels * 2, out_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_features = edge_attr
        message_input = torch.cat([x[row], x[col], edge_features], dim=1)
        messages = self.mlp(message_input)
        out = torch.zeros_like(x)
        out.index_add_(0, col, messages)
        return F.relu(out)


class MolecularMPNN(nn.Module):
    def __init__(
        self,
        node_features=24,
        edge_features=10,
        hidden_dim=256,
        num_layers=5,
        num_targets=9,
        dropout=0.2,
    ):
        super().__init__()
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = GINEConvLayer(hidden_dim, hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.pool_weights = nn.Parameter(torch.ones(3))

        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.target_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_dim // 4, 1),
                )
                for _ in range(num_targets)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x = self.node_embedding(x)

        xs = [x]
        for i in range(self.num_layers):
            x_new = self.convs[i](x, edge_index, edge_attr)
            x_new = self.norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new
            xs.append(x)

        x_stack = torch.stack(xs, dim=1).mean(dim=1)

        attn_scores = self.attention(x_stack)
        batch = (
            data.batch
            if hasattr(data, "batch")
            else torch.zeros(x_stack.size(0), dtype=torch.long, device=x.device)
        )

        # Compute per-graph softmax attention weights
        attn_exp = torch.exp(attn_scores - attn_scores.max(dim=0, keepdim=True)[0])
        # Sum attn_exp per graph, then scatter back to node level for normalization
        num_graphs = batch.max().item() + 1
        attn_per_graph = torch.zeros(num_graphs, device=x.device).scatter_add_(
            0, batch, attn_exp.squeeze(-1)
        )
        attn_weights = attn_exp / (attn_per_graph[batch].unsqueeze(-1) + 1e-8)

        weighted_x = x_stack * attn_weights
        graph_attn = global_add_pool(weighted_x, batch)
        graph_mean = global_mean_pool(x_stack, batch)
        graph_max = global_max_pool(x_stack, batch)

        pool_weights = F.softmax(self.pool_weights, dim=0)
        graph_repr = (
            pool_weights[0] * graph_attn
            + pool_weights[1] * graph_mean
            + pool_weights[2] * graph_max
        )

        shared = self.shared_head(graph_repr)
        outputs = [head(shared) for head in self.target_heads]
        return torch.cat(outputs, dim=1)


class MAELoss(nn.Module):
    def __init__(self, target_weights=None, eps=1e-8):
        super().__init__()
        if target_weights is not None:
            self.register_buffer(
                "target_weights", torch.tensor(target_weights, dtype=torch.float32)
            )
        else:
            self.target_weights = None
        self.eps = eps

    def forward(self, pred, target, mask):
        abs_diff = torch.abs(pred - target)
        masked_mae = (abs_diff * mask).sum(dim=0) / (mask.sum(dim=0) + self.eps)
        if self.target_weights is not None:
            weighted_mae = masked_mae * self.target_weights.to(pred.device)
            return weighted_mae.mean()
        return masked_mae.mean()


# ========== DATA LOADING ==========
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

target_cols = TARGET_NAMES

# Log transform for non-log endpoints
for col in target_cols[1:]:
    if col in train_df.columns:
        train_df[col] = np.log1p(train_df[col].clip(lower=0))

train_targets = train_df[target_cols].values.astype(np.float32)
train_smiles = train_df["SMILES"].values
test_smiles = test_df["SMILES"].values
test_names = test_df["Molecule Name"].values

# Pre-compute target statistics for relative error normalization
target_mads = []
for t in range(train_targets.shape[1]):
    valid_targets = train_targets[~np.isnan(train_targets[:, t]), t]
    if len(valid_targets) > 0:
        median_val = np.median(valid_targets)
        mad = np.median(np.abs(valid_targets - median_val))
        target_mads.append(max(mad, 0.001))
    else:
        target_mads.append(1.0)

target_mads = np.array(target_mads)
target_weights = 1.0 / (target_mads + 1e-8)
target_weights = target_weights / target_weights.mean()

print(f"Target MADs: {target_mads}")
print(f"Target weights: {target_weights}")


def compute_ma_rae(predictions, targets, masks, target_mads):
    num_targets = predictions.shape[1]
    relative_errors = []
    for t in range(num_targets):
        valid_idx = masks[:, t] > 0.5
        if valid_idx.sum() > 0:
            pred_t = predictions[valid_idx, t]
            target_t = targets[valid_idx, t]
            abs_error = torch.abs(pred_t - target_t).mean().item()
            mad = target_mads[t]
            relative_error = abs_error / max(mad, 1e-8)
            relative_errors.append(relative_error)
    return np.mean(relative_errors) if relative_errors else 0.0


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch_data, targets, masks in dataloader:
        if batch_data is None:
            continue
        batch_data = batch_data.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        predictions = model(batch_data)
        loss = criterion(predictions, targets, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def validate(model, dataloader, target_mads):
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for batch_data, targets, masks in dataloader:
            if batch_data is None:
                continue
            batch_data = batch_data.to(device)
            predictions = model(batch_data)
            all_preds.append(predictions.cpu())
            all_targets.append(targets)
            all_masks.append(masks)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    ma_rae = compute_ma_rae(all_preds, all_targets, all_masks, target_mads)

    per_target_mae = []
    for t in range(all_preds.shape[1]):
        valid_idx = all_masks[:, t] > 0.5
        if valid_idx.sum() > 0:
            mae = (
                torch.abs(all_preds[valid_idx, t] - all_targets[valid_idx, t])
                .mean()
                .item()
            )
            per_target_mae.append(mae)
        else:
            per_target_mae.append(float("nan"))

    return ma_rae, per_target_mae


# ========== TRAINING ==========
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

fold_models = []
fold_scores = []

print(f"\nStarting {n_folds}-fold cross-validation...")

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_targets)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/{n_folds}")
    print(f"{'='*50}")

    smiles_train_fold = train_smiles[train_idx]
    y_train_fold = train_targets[train_idx]
    smiles_valid_fold = train_smiles[valid_idx]
    y_valid_fold = train_targets[valid_idx]

    train_dataset = MolecularGraphDataset(smiles_train_fold, y_train_fold)
    valid_dataset = MolecularGraphDataset(smiles_valid_fold, y_valid_fold)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=2,
        pin_memory=True,
    )

    model = MolecularMPNN(
        node_features=24,
        edge_features=10,
        hidden_dim=256,
        num_layers=5,
        num_targets=9,
        dropout=0.2,
    ).to(device)

    criterion = MAELoss(target_weights=target_weights)

    base_params, head_params = [], []
    for name, param in model.named_parameters():
        if "target_heads" in name or "pool_weights" in name:
            head_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "weight_decay": 1e-5},
            {"params": head_params, "weight_decay": 1e-4},
        ],
        lr=3e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6
    )

    best_ma_rae = float("inf")
    patience = 20
    patience_counter = 0
    best_model_state = None
    n_epochs = 200

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)

        if (epoch + 1) % 5 == 0:
            val_ma_rae, _ = validate(model, valid_loader, target_mads)
            scheduler.step()

            if val_ma_rae < best_ma_rae:
                best_ma_rae = val_ma_rae
                patience_counter = 0
                best_model_state = model.state_dict()
                print(
                    f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val MA-RAE: {val_ma_rae:.4f} | BEST"
                )
            else:
                patience_counter += 1
                print(
                    f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val MA-RAE: {val_ma_rae:.4f}"
                )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        elif (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f}")

    model.load_state_dict(best_model_state)
    fold_models.append(model)
    fold_scores.append(best_ma_rae)
    print(f"Fold {fold + 1} best MA-RAE: {best_ma_rae:.4f}")

    del train_dataset, valid_dataset, train_loader, valid_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

final_val_score = np.mean(fold_scores)
print(f"\n{'='*50}")
print(f"Cross-validation results:")
for i, score in enumerate(fold_scores):
    print(f"  Fold {i+1}: MA-RAE = {score:.4f}")
print(f"  Mean MA-RAE = {final_val_score:.4f}")

# ========== TEST INFERENCE ==========
print(f"\n{'='*50}")
print("Generating test predictions...")
print(f"{'='*50}")

test_dataset = MolecularGraphDataset(test_smiles)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_graphs,
    num_workers=2,
    pin_memory=True,
)

all_test_preds = []
for fold_idx, model in enumerate(fold_models):
    model.eval()
    fold_preds = []
    with torch.no_grad():
        for batch_data in test_loader:
            if batch_data is None:
                continue
            batch_data = batch_data.to(device)
            predictions = model(batch_data)
            fold_preds.append(predictions.cpu().numpy())
    if fold_preds:
        fold_preds = np.concatenate(fold_preds, axis=0)
        all_test_preds.append(fold_preds)

test_predictions = np.mean(all_test_preds, axis=0)

# Inverse log transform
for t_idx in LOG_TARGET_INDICES:
    if t_idx < test_predictions.shape[1]:
        test_predictions[:, t_idx] = np.expm1(test_predictions[:, t_idx])
        test_predictions[:, t_idx] = np.maximum(0, test_predictions[:, t_idx])

# Create submission
submission = pd.DataFrame()
submission["Molecule Name"] = test_names
for i, col_name in enumerate(target_cols):
    submission[col_name] = test_predictions[:, i]

submission = submission[sample_sub.columns]
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_f77cabbd8d364115bb899ae151f1a54f.csv", index=False)
print(f"Submission saved to ./submission/submission_f77cabbd8d364115bb899ae151f1a54f.csv")
print(f"Final Validation Score: {final_val_score}")
