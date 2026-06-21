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
os.environ['DGLBACKEND'] = 'pytorch'
os.environ['DGL_DISABLE_GRAPHBOOT'] = '1'
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
def compute_fingerprints(smiles_list, n_bits=2048, radius=3):
    """Compute Morgan fingerprints for a list of SMILES."""
    fps = []
    valid_mask = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
            valid_mask.append(False)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp, dtype=np.float32))
            valid_mask.append(True)
    return np.array(fps), np.array(valid_mask)


def compute_rdkit_descriptors(smiles_list):
    """Compute a comprehensive set of RDKit descriptors."""
    desc_names = [d[0] for d in Descriptors._descList]
    all_descs = []
    valid_mask = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            all_descs.append(np.zeros(len(desc_names), dtype=np.float32))
            valid_mask.append(False)
        else:
            try:
                descs = [Descriptors.__dict__[name](mol) for name in desc_names]
                descs = np.array([0.0 if d is None else float(d) for d in descs], dtype=np.float32)
                descs = np.nan_to_num(descs, nan=0.0, posinf=1e6, neginf=-1e6)
                all_descs.append(descs)
                valid_mask.append(True)
            except Exception:
                all_descs.append(np.zeros(len(desc_names), dtype=np.float32))
                valid_mask.append(False)
    return np.array(all_descs), np.array(valid_mask)


class MolecularDataset(torch.utils.data.Dataset):
    def __init__(self, fingerprints, descriptors=None, y_matrix=None):
        self.fps = torch.tensor(fingerprints, dtype=torch.float)
        if descriptors is not None:
            self.descs = torch.tensor(descriptors, dtype=torch.float)
        else:
            self.descs = None
        if y_matrix is not None:
            self.targets = torch.tensor(np.nan_to_num(y_matrix, nan=0.0), dtype=torch.float)
            self.masks = torch.tensor(~np.isnan(y_matrix), dtype=torch.float)
        else:
            self.targets = torch.zeros((fingerprints.shape[0], 0), dtype=torch.float)
            self.masks = torch.zeros((fingerprints.shape[0], 0), dtype=torch.float)

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        if self.descs is not None:
            return self.fps[idx], self.descs[idx], self.targets[idx], self.masks[idx]
        return self.fps[idx], self.targets[idx], self.masks[idx]


# ---------------------------------------------------------------
# Lightweight MLP Model (attention-based)
# ---------------------------------------------------------------
class FeatureAttention(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.query = nn.Linear(feat_dim, hidden_dim)
        self.key = nn.Linear(feat_dim, hidden_dim)
        self.value = nn.Linear(feat_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        # x: (batch, seq_len, feat_dim) or (batch, feat_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, feat_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        out = torch.matmul(attn, V)
        return out.squeeze(1)


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim=2048+208, hidden_dim=512, num_tasks=9, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads
        self.heads = nn.ModuleList()
        for _ in range(num_tasks):
            head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.heads.append(head)

        self.task_embeddings = nn.Embedding(num_tasks, 32)
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

    def forward(self, x):
        x = self.input_norm(x)
        shared_out = self.shared(x)
        predictions = []
        for t, head in enumerate(self.heads):
            pred = head(shared_out)
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
def train_model_mlp(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=200,
    device="cuda",
    patience=30,
):
    best_val_score = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for fps, targets, masks in train_loader:
            fps = fps.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            predictions = model(fps)
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
        val_score = evaluate_model_mlp(model, val_loader, device)

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


def evaluate_model_mlp(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for fps, targets, masks in data_loader:
            fps = fps.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            predictions = model(fps)
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
        y_train_split[valid_mask_col_train, col_idx] = np.log1p(col_data_train[valid_mask_col_train])

        col_data_val = y_val_split[:, col_idx]
        valid_mask_col_val = ~np.isnan(col_data_val)
        y_val_split[valid_mask_col_val, col_idx] = np.log1p(col_data_val[valid_mask_col_val])

    print(f"Train samples: {len(train_smiles_split)}")
    print(f"Validation samples: {len(val_smiles_split)}")
    print(f"Test samples: {len(test_smiles)}")

    # Compute features
    print("Computing Morgan fingerprints (radius=3, 2048 bits)...")
    train_fps, _ = compute_fingerprints(train_smiles_split)
    val_fps, _ = compute_fingerprints(val_smiles_split)
    test_fps, _ = compute_fingerprints(test_smiles)

    print("Computing RDKit descriptors...")
    train_descs, _ = compute_rdkit_descriptors(train_smiles_split)
    val_descs, _ = compute_rdkit_descriptors(val_smiles_split)
    test_descs, _ = compute_rdkit_descriptors(test_smiles)

    # Combine features
    train_feat = np.concatenate([train_fps, train_descs], axis=1)
    val_feat = np.concatenate([val_fps, val_descs], axis=1)
    test_feat = np.concatenate([test_fps, test_descs], axis=1)

    # Feature normalization
    feat_mean = np.mean(train_feat, axis=0)
    feat_std = np.std(train_feat, axis=0) + 1e-8
    train_feat = (train_feat - feat_mean) / feat_std
    val_feat = (val_feat - feat_mean) / feat_std
    test_feat = (test_feat - feat_mean) / feat_std

    input_dim = train_feat.shape[1]
    print(f"Feature dimension: {input_dim}")

    # Build model
    model = MultiTaskMLP(
        input_dim=input_dim,
        hidden_dim=512,
        num_tasks=MODEL_CONFIG["num_tasks"],
        dropout=MODEL_CONFIG["dropout"],
    )
    model = model.to(device)

    criterion = MaskedMAELoss(gate_reg_weight=0.0)  # No gate reg for MLP
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_CONFIG["lr"],
        weight_decay=MODEL_CONFIG["weight_decay"],
    )

    def lr_lambda(current_step, warmup=500, total=5000):
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        progress = float(current_step - warmup) / float(max(1, total - warmup))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda(step, 500, 5000)
    )

    # Create datasets and dataloaders
    train_dataset = MolecularDataset(train_feat, y_matrix=y_train_split)
    val_dataset = MolecularDataset(val_feat, y_matrix=y_val_split)

    def collate_fn(batch):
        fps, targets, masks = zip(*batch)
        return torch.stack(fps), torch.stack(targets), torch.stack(masks)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Train
    print("Starting training...")
    model, best_val_score = train_model_mlp(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=200,
        device=device,
        patience=30,
    )
    print(f"Best validation MA-RAE: {best_val_score:.4f}")

    # Test inference
    print("Performing test inference...")
    test_dataset = MolecularDataset(test_feat)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    model.eval()
    all_test_preds = []
    with torch.no_grad():
        for fps, _, _ in test_loader:
            fps = fps.to(device)
            predictions = model(fps)
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
    submission.to_csv("./submission/submission.csv", index=False)
    print(f"Submission saved to ./submission/submission.csv")
    print(f"Submission shape: {submission.shape}")

    score = best_val_score
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()