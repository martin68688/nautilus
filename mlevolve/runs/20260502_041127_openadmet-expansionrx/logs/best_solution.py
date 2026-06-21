import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import math
import warnings
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ==============================================================================
# DATA PROCESSING AND FEATURE ENGINEERING
# ==============================================================================


def compute_morgan_fingerprints(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def compute_rdkit_descriptors(mol):
    if mol is None:
        return np.zeros(20)
    desc = []
    desc.append(Descriptors.MolWt(mol))
    desc.append(Descriptors.MolLogP(mol))
    desc.append(Descriptors.NumHDonors(mol))
    desc.append(Descriptors.NumHAcceptors(mol))
    desc.append(Descriptors.TPSA(mol))
    desc.append(Descriptors.NumRotatableBonds(mol))
    desc.append(Descriptors.NumAromaticRings(mol))
    desc.append(Descriptors.NumAliphaticRings(mol))
    desc.append(Descriptors.NumSaturatedRings(mol))
    desc.append(Descriptors.NumHeteroatoms(mol))
    desc.append(Descriptors.FractionCSP3(mol))
    desc.append(Descriptors.HeavyAtomCount(mol))
    desc.append(Descriptors.NHOHCount(mol))
    desc.append(Descriptors.NOCount(mol))
    desc.append(Descriptors.RingCount(mol))
    try:
        desc.append(rdMolDescriptors.CalcChi0n(mol))
    except:
        desc.append(0.0)
    try:
        desc.append(rdMolDescriptors.CalcChi1n(mol))
    except:
        desc.append(0.0)
    try:
        desc.append(rdMolDescriptors.CalcKappa1(mol))
    except:
        desc.append(0.0)
    try:
        desc.append(rdMolDescriptors.CalcHallKierAlpha(mol))
    except:
        desc.append(0.0)
    try:
        desc.append(rdMolDescriptors.CalcNumRings(mol))
    except:
        desc.append(0.0)
    return np.array(desc)


def compute_molecular_features(smiles_list):
    n_mols = len(smiles_list)
    fps = []
    rdkit_desc = []
    valid_mask = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fps.append(compute_morgan_fingerprints(smi))
            rdkit_desc.append(compute_rdkit_descriptors(mol))
            valid_mask.append(1)
        else:
            fps.append(np.zeros(2048))
            rdkit_desc.append(np.zeros(20))
            valid_mask.append(0)
    fps = np.array(fps)
    rdkit_desc = np.array(rdkit_desc)
    valid_mask = np.array(valid_mask)
    return fps, rdkit_desc, valid_mask


def preprocess_data():
    print("Loading data...")
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

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

    train_targets = train_df[target_cols].copy()

    print("Computing molecular features for training data...")
    train_fps, train_rdkit, train_valid = compute_molecular_features(
        train_df["SMILES"].values
    )

    print("Computing molecular features for test data...")
    test_fps, test_rdkit, test_valid = compute_molecular_features(
        test_df["SMILES"].values
    )

    train_features = np.hstack([train_fps, train_rdkit, train_valid.reshape(-1, 1)])
    test_features = np.hstack([test_fps, test_rdkit, test_valid.reshape(-1, 1)])

    fp_names = [f"fp_{i}" for i in range(2048)]
    rdkit_names = [
        "MolWt",
        "MolLogP",
        "NumHDonors",
        "NumHAcceptors",
        "TPSA",
        "NumRotatableBonds",
        "NumAromaticRings",
        "NumAliphaticRings",
        "NumSaturatedRings",
        "NumHeteroatoms",
        "FractionCSP3",
        "HeavyAtomCount",
        "NHOHCount",
        "NOCount",
        "RingCount",
        "Chi0n",
        "Chi1n",
        "Kappa1",
        "HallKierAlpha",
        "NumRings",
    ]
    feature_names = fp_names + rdkit_names + ["valid_mol"]

    train_features_df = pd.DataFrame(train_features, columns=feature_names)
    train_features_df["Molecule Name"] = train_df["Molecule Name"].values
    train_features_df["SMILES"] = train_df["SMILES"].values

    test_features_df = pd.DataFrame(test_features, columns=feature_names)
    test_features_df["Molecule Name"] = test_df["Molecule Name"].values
    test_features_df["SMILES"] = test_df["SMILES"].values

    print("Handling missing values in features...")
    feature_means = train_features_df[fp_names + rdkit_names].mean()
    train_features_df[fp_names + rdkit_names] = train_features_df[
        fp_names + rdkit_names
    ].fillna(feature_means)
    test_features_df[fp_names + rdkit_names] = test_features_df[
        fp_names + rdkit_names
    ].fillna(feature_means)

    print("Creating train/validation split...")
    # Split first using random split to avoid PCA/KMeans data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    # Use molecule names as groups for random grouping (no PCA/leakage)
    groups = np.arange(len(train_features_df))
    train_idx, val_idx = next(gss.split(train_features_df, groups=groups))

    train_df_final = train_features_df.iloc[train_idx].copy()
    val_df = train_features_df.iloc[val_idx].copy()

    train_df_final[target_cols] = train_targets.iloc[train_idx].values
    val_df[target_cols] = train_targets.iloc[val_idx].values

    # Fit scaler on training data only to prevent leakage
    scaler = RobustScaler()
    train_features_scaled = scaler.fit_transform(
        train_df_final[fp_names + rdkit_names]
    )
    test_features_scaled = scaler.transform(test_features_df[fp_names + rdkit_names])

    train_df_final[fp_names + rdkit_names] = train_features_scaled
    val_df[fp_names + rdkit_names] = scaler.transform(val_df[fp_names + rdkit_names])
    test_features_df[fp_names + rdkit_names] = test_features_scaled

    print(f"Train samples: {len(train_df_final)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_features_df)}")

    print("\nTarget missingness (%):")
    for col in target_cols:
        missing_pct = (train_df_final[col].isna().sum() / len(train_df_final)) * 100
        print(f"  {col}: {missing_pct:.1f}%")

    print("\nSaving processed data...")
    os.makedirs("./working", exist_ok=True)
    train_df_final.to_parquet("./working/train_processed.parquet")
    val_df.to_parquet("./working/val_processed.parquet")
    test_features_df.to_parquet("./working/test_processed.parquet")

    joblib.dump(scaler, "./working/scaler.pkl")
    joblib.dump(feature_means, "./working/feature_means.pkl")
    pd.DataFrame({"target_col": target_cols}).to_csv("./working/target_cols.csv", index=False)
    pd.DataFrame({"fp_name": fp_names}).to_csv("./working/fp_names.csv", index=False)
    pd.DataFrame({"rdkit_name": rdkit_names}).to_csv("./working/rdkit_names.csv", index=False)

    print("Data processing complete!")
    return train_df_final, val_df, test_features_df, target_cols


# ==============================================================================
# MODEL DESIGN (MultiTaskRegressor with Uncertainty Weighting)
# ==============================================================================


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, in_dim),
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TaskHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class TargetSpecificAttention(nn.Module):
    def __init__(self, input_dim, num_targets, hidden_dim=64):
        super().__init__()
        self.num_targets = num_targets
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_targets * input_dim),
        )

    def forward(self, x):
        attention_weights = self.attention_net(x)
        attention_weights = attention_weights.view(-1, self.num_targets, x.size(-1))
        attention_weights = F.softmax(attention_weights, dim=-1)
        outputs = []
        for t in range(self.num_targets):
            weighted = x * attention_weights[:, t, :]
            outputs.append(weighted)
        return torch.stack(outputs, dim=1)


class MultiTaskRegressor(nn.Module):
    def __init__(
        self,
        input_dim=2069,
        hidden_dim=1024,
        bottleneck_dim=256,
        num_targets=9,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.num_targets = num_targets

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate),
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.target_attention = TargetSpecificAttention(
            bottleneck_dim, num_targets, hidden_dim=64
        )

        self.task_heads = nn.ModuleList(
            [
                TaskHead(bottleneck_dim, hidden_dim=128, dropout_rate=dropout_rate)
                for _ in range(num_targets)
            ]
        )

        self.log_vars = nn.Parameter(torch.zeros(num_targets) - 0.5)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        shared_features = self.shared_encoder(x)
        bottleneck_features = self.bottleneck(shared_features)
        attended_features = self.target_attention(bottleneck_features)

        predictions = []
        for t in range(self.num_targets):
            task_input = attended_features[:, t, :]
            task_input = task_input + bottleneck_features * 0.1
            pred = self.task_heads[t](task_input)
            predictions.append(pred)

        predictions = torch.cat(predictions, dim=-1)
        return predictions

    def get_missing_mask_loss(self, predictions, targets, missing_mask):
        precision = torch.exp(-self.log_vars)
        abs_errors = torch.abs(predictions - targets)
        masked_errors = abs_errors * missing_mask
        weighted_errors = masked_errors * precision.unsqueeze(0)
        target_counts = missing_mask.sum(dim=0).clamp(min=1)
        per_target_loss = weighted_errors.sum(dim=0) / target_counts
        uncertainty_reg = 0.5 * self.log_vars
        total_loss = (per_target_loss + uncertainty_reg).mean()
        return total_loss


def create_model_and_optimizer(
    input_dim=2069,
    hidden_dim=1024,
    bottleneck_dim=256,
    num_targets=9,
    dropout_rate=0.3,
    learning_rate=1e-3,
    weight_decay=1e-5,
):
    model = MultiTaskRegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_targets=num_targets,
        dropout_rate=dropout_rate,
    )
    criterion = model.get_missing_mask_loss

    param_groups = [
        {
            "params": model.shared_encoder.parameters(),
            "lr": learning_rate * 0.5,
            "weight_decay": weight_decay * 2,
        },
        {
            "params": model.bottleneck.parameters(),
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": model.target_attention.parameters(),
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
    ]
    for head in model.task_heads:
        param_groups.append(
            {
                "params": head.parameters(),
                "lr": learning_rate * 1.5,
                "weight_decay": weight_decay * 0.5,
            }
        )
    param_groups.append(
        {"params": model.log_vars, "lr": learning_rate * 0.1, "weight_decay": 0}
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01
    )

    return model, criterion, optimizer, scheduler


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================


class MolecularDataset(Dataset):
    def __init__(self, features, targets=None, target_mask=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.target_mask = (
            torch.FloatTensor(target_mask) if target_mask is not None else None
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx], self.target_mask[idx]
        return self.features[idx]


def prepare_dataloaders(train_df, val_df, target_cols, batch_size=128):
    feature_cols = [
        c
        for c in train_df.columns
        if c.startswith("fp_")
        or c
        in [
            "MolWt",
            "MolLogP",
            "NumHDonors",
            "NumHAcceptors",
            "TPSA",
            "NumRotatableBonds",
            "NumAromaticRings",
            "NumAliphaticRings",
            "NumSaturatedRings",
            "NumHeteroatoms",
            "FractionCSP3",
            "HeavyAtomCount",
            "NHOHCount",
            "NOCount",
            "RingCount",
            "Chi0n",
            "Chi1n",
            "Kappa1",
            "HallKierAlpha",
            "NumRings",
            "valid_mol",
        ]
    ]

    train_features = train_df[feature_cols].values.astype(np.float32)
    train_targets = train_df[target_cols].values.astype(np.float32)
    train_mask = (~np.isnan(train_targets)).astype(np.float32)
    train_targets = np.nan_to_num(train_targets, nan=0.0)

    val_features = val_df[feature_cols].values.astype(np.float32)
    val_targets = val_df[target_cols].values.astype(np.float32)
    val_mask = (~np.isnan(val_targets)).astype(np.float32)
    val_targets = np.nan_to_num(val_targets, nan=0.0)

    train_dataset = MolecularDataset(train_features, train_targets, train_mask)
    val_dataset = MolecularDataset(val_features, val_targets, val_mask)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        train_features,
        val_features,
        train_targets,
        val_targets,
        train_mask,
        val_mask,
    )


def compute_target_ranges(train_targets, train_mask):
    num_targets = train_targets.shape[1]
    ranges = np.zeros(num_targets)
    for t in range(num_targets):
        mask = train_mask[:, t] == 1
        if mask.sum() > 0:
            ranges[t] = np.max(train_targets[mask, t]) - np.min(train_targets[mask, t])
            if ranges[t] == 0:
                ranges[t] = 1.0
        else:
            ranges[t] = 1.0
    return ranges


def compute_ma_rae_numpy(predictions, targets, masks, target_ranges):
    num_targets = predictions.shape[1]
    maes = []
    for t in range(num_targets):
        mask = masks[:, t] == 1
        if mask.sum() > 0:
            mae = np.mean(np.abs(predictions[mask, t] - targets[mask, t]))
            if target_ranges[t] > 0:
                rae = mae / target_ranges[t]
                maes.append(rae)
    return np.mean(maes) if maes else 0.0


def compute_per_target_mae(predictions, targets, masks):
    num_targets = predictions.shape[1]
    maes = {}
    for t in range(num_targets):
        mask = masks[:, t] == 1
        if mask.sum() > 0:
            mae = np.mean(np.abs(predictions[mask, t] - targets[mask, t]))
            maes[t] = mae
    return maes


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    val_features,
    val_targets,
    val_mask,
    target_ranges,
    num_epochs=300,
    patience=30,
    device="cuda",
):
    model = model.to(device)
    best_score = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_features, batch_targets, batch_mask in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets, batch_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        model.eval()
        with torch.no_grad():
            val_features_tensor = torch.FloatTensor(val_features).to(device)
            val_predictions = model(val_features_tensor).cpu().numpy()
            val_score = compute_ma_rae_numpy(
                val_predictions, val_targets, val_mask, target_ranges
            )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_train_loss:.4f} | Val MA-RAE: {val_score:.4f} | LR: {current_lr:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, "./working/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\nEarly stopping triggered after {epoch+1} epochs. Best Val MA-RAE: {best_score:.4f}"
                )
                break

    model.load_state_dict(best_model_state)
    model = model.to(device)
    return model, best_score


def main():
    print("Loading processed data...")
    train_df = pd.read_parquet("./working/train_processed.parquet")
    val_df = pd.read_parquet("./working/val_processed.parquet")
    test_df = pd.read_parquet("./working/test_processed.parquet")
    target_cols = pd.read_csv("./working/target_cols.csv")["target_col"].tolist()

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    print("\nPreparing data loaders...")
    (
        train_loader,
        val_loader,
        train_features,
        val_features,
        train_targets,
        val_targets,
        train_mask,
        val_mask,
    ) = prepare_dataloaders(train_df, val_df, target_cols, batch_size=128)

    print("Computing target ranges...")
    target_ranges = compute_target_ranges(train_targets, train_mask)
    for i, col in enumerate(target_cols):
        print(f"  {col}: range = {target_ranges[i]:.4f}")

    print("\nCreating model...")
    input_dim = train_features.shape[1]
    model, criterion, optimizer, scheduler = create_model_and_optimizer(
        input_dim=input_dim,
        hidden_dim=1024,
        bottleneck_dim=256,
        num_targets=len(target_cols),
        dropout_rate=0.3,
        learning_rate=1e-3,
        weight_decay=1e-5,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\nStarting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, best_score = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        val_features,
        val_targets,
        val_mask,
        target_ranges,
        num_epochs=300,
        patience=30,
        device=device,
    )

    print(f"\nBest validation MA-RAE: {best_score:.6f}")

    print("\nPer-target MAE on validation set:")
    model.eval()
    with torch.no_grad():
        val_features_tensor = torch.FloatTensor(val_features).to(device)
        val_predictions = model(val_features_tensor).cpu().numpy()
        per_target_maes = compute_per_target_mae(val_predictions, val_targets, val_mask)
        for t, mae in per_target_maes.items():
            print(f"  {target_cols[t]}: MAE = {mae:.4f}")

    print("\nPerforming test inference...")
    feature_cols = [
        c
        for c in test_df.columns
        if c.startswith("fp_")
        or c
        in [
            "MolWt",
            "MolLogP",
            "NumHDonors",
            "NumHAcceptors",
            "TPSA",
            "NumRotatableBonds",
            "NumAromaticRings",
            "NumAliphaticRings",
            "NumSaturatedRings",
            "NumHeteroatoms",
            "FractionCSP3",
            "HeavyAtomCount",
            "NHOHCount",
            "NOCount",
            "RingCount",
            "Chi0n",
            "Chi1n",
            "Kappa1",
            "HallKierAlpha",
            "NumRings",
            "valid_mol",
        ]
    ]
    test_features = test_df[feature_cols].values.astype(np.float32)
    test_dataset = MolecularDataset(test_features)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    model.eval()
    all_test_preds = []
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features.to(device)
            batch_preds = model(batch_features).cpu().numpy()
            all_test_preds.append(batch_preds)

    test_predictions = np.vstack(all_test_preds)

    print("\nCreating submission file...")
    submission = pd.DataFrame(
        {
            "Molecule Name": test_df["Molecule Name"].values,
            target_cols[0]: test_predictions[:, 0],
            target_cols[1]: test_predictions[:, 1],
            target_cols[2]: test_predictions[:, 2],
            target_cols[3]: test_predictions[:, 3],
            target_cols[4]: test_predictions[:, 4],
            target_cols[5]: test_predictions[:, 5],
            target_cols[6]: test_predictions[:, 6],
            target_cols[7]: test_predictions[:, 7],
            target_cols[8]: test_predictions[:, 8],
        }
    )

    submission = submission[
        [
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
    ]

    os.makedirs("./submission", exist_ok=True)
    submission.to_csv("./submission/submission.csv", index=False)
    print(f"Submission saved to ./submission/submission.csv")
    print(f"Submission shape: {submission.shape}")

    print(f"Final Validation Score: {best_score}")


if __name__ == "__main__":
    preprocess_data()
    main()