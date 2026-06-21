import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ========== DATA PROCESSING AND FEATURE ENGINEERING ==========
def calculate_morgan_fingerprints(mol, radius=2, nBits=2048):
    if mol is None:
        return np.zeros(nBits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))


def calculate_maccs_fingerprints(mol):
    if mol is None:
        return np.zeros(167)
    return np.array(MACCSkeys.GenMACCSKeys(mol))


def calculate_rdkit_fingerprints(mol, nBits=1024):
    if mol is None:
        return np.zeros(nBits)
    return np.array(Chem.RDKFingerprint(mol, fpSize=nBits))


def calculate_molecular_descriptors(mol):
    if mol is None:
        return {
            k: 0.0
            for k in [
                "MolWt",
                "HeavyAtomMolWt",
                "ExactMolWt",
                "NumHeavyAtoms",
                "NumHeteroatoms",
                "NumRotatableBonds",
                "NumHBD",
                "NumHBA",
                "NumAromaticRings",
                "NumAliphaticRings",
                "NumSaturatedRings",
                "NumRingAssemblies",
                "NumSpiroAtoms",
                "NumBridgeheadAtoms",
                "TPSA",
                "LogP",
                "FractionCsp3",
                "HeavyAtomCount",
                "NHOHCount",
                "NOCount",
                "RingCount",
                "MolMR",
                "MaxAbsPartialCharge",
                "MaxPartialCharge",
                "MinAbsPartialCharge",
                "MinPartialCharge",
                "NumValenceElectrons",
                "BalabanJ",
                "BertzCT",
                "Chi0",
                "Chi0n",
                "Chi0v",
                "Chi1",
                "Chi1n",
                "Chi1v",
                "Chi2n",
                "Chi2v",
                "Chi3n",
                "Chi3v",
                "Chi4n",
                "Chi4v",
                "HallKierAlpha",
                "Ipc",
                "Kappa1",
                "Kappa2",
                "Kappa3",
                "LabuteASA",
            ]
            + [f"PEOE_VSA{i}" for i in range(1, 15)]
            + [f"SMR_VSA{i}" for i in range(1, 11)]
            + [f"SlogP_VSA{i}" for i in range(1, 13)]
            + [f"EState_VSA{i}" for i in range(1, 12)]
            + [f"VSA_EState{i}" for i in range(1, 11)]
            + [
                "AvgIpc",
                "Chi0n_Chi0v",
                "FractionAromaticAtoms",
                "NumAromaticHeterocycles",
                "NumAromaticCarbocycles",
                "NumAliphaticHeterocycles",
                "NumAliphaticCarbocycles",
                "LargestRingSize",
            ]
        }

    descriptors = {}
    descriptors["MolWt"] = Descriptors.MolWt(mol)
    descriptors["HeavyAtomMolWt"] = Descriptors.HeavyAtomMolWt(mol)
    descriptors["ExactMolWt"] = Descriptors.ExactMolWt(mol)
    descriptors["NumHeavyAtoms"] = Descriptors.HeavyAtomCount(mol)
    descriptors["NumHeteroatoms"] = Descriptors.NumHeteroatoms(mol)
    descriptors["NumRotatableBonds"] = Descriptors.NumRotatableBonds(mol)
    descriptors["NumHBD"] = Descriptors.NumHDonors(mol)
    descriptors["NumHBA"] = Descriptors.NumHAcceptors(mol)
    descriptors["NumAromaticRings"] = Descriptors.NumAromaticRings(mol)
    descriptors["NumAliphaticRings"] = Descriptors.NumAliphaticRings(mol)
    descriptors["NumSaturatedRings"] = Descriptors.NumSaturatedRings(mol)
    # NumRingAssemblies not available in all RDKit versions, skip
    descriptors["NumRingAssemblies"] = 0
    descriptors["NumSpiroAtoms"] = Descriptors.NumSpiroAtoms(mol)
    descriptors["NumBridgeheadAtoms"] = Descriptors.NumBridgeheadAtoms(mol)
    descriptors["TPSA"] = Descriptors.TPSA(mol)
    descriptors["LogP"] = Descriptors.MolLogP(mol)
    descriptors["FractionCsp3"] = Descriptors.FractionCSP3(mol)
    descriptors["HeavyAtomCount"] = Descriptors.HeavyAtomCount(mol)
    descriptors["NHOHCount"] = Descriptors.NHOHCount(mol)
    descriptors["NOCount"] = Descriptors.NOCount(mol)
    descriptors["RingCount"] = Descriptors.RingCount(mol)
    descriptors["MolMR"] = Descriptors.MolMR(mol)
    descriptors["MaxAbsPartialCharge"] = Descriptors.MaxAbsPartialCharge(mol)
    descriptors["MaxPartialCharge"] = Descriptors.MaxPartialCharge(mol)
    descriptors["MinAbsPartialCharge"] = Descriptors.MinAbsPartialCharge(mol)
    descriptors["MinPartialCharge"] = Descriptors.MinPartialCharge(mol)
    descriptors["NumValenceElectrons"] = Descriptors.NumValenceElectrons(mol)
    descriptors["BalabanJ"] = Descriptors.BalabanJ(mol)
    descriptors["BertzCT"] = Descriptors.BertzCT(mol)
    descriptors["Chi0"] = Descriptors.Chi0(mol)
    descriptors["Chi0n"] = Descriptors.Chi0n(mol)
    descriptors["Chi0v"] = Descriptors.Chi0v(mol)
    descriptors["Chi1"] = Descriptors.Chi1(mol)
    descriptors["Chi1n"] = Descriptors.Chi1n(mol)
    descriptors["Chi1v"] = Descriptors.Chi1v(mol)
    descriptors["Chi2n"] = Descriptors.Chi2n(mol)
    descriptors["Chi2v"] = Descriptors.Chi2v(mol)
    descriptors["Chi3n"] = Descriptors.Chi3n(mol)
    descriptors["Chi3v"] = Descriptors.Chi3v(mol)
    descriptors["Chi4n"] = Descriptors.Chi4n(mol)
    descriptors["Chi4v"] = Descriptors.Chi4v(mol)
    descriptors["HallKierAlpha"] = Descriptors.HallKierAlpha(mol)
    descriptors["Ipc"] = Descriptors.Ipc(mol)
    descriptors["Kappa1"] = Descriptors.Kappa1(mol)
    descriptors["Kappa2"] = Descriptors.Kappa2(mol)
    descriptors["Kappa3"] = Descriptors.Kappa3(mol)
    descriptors["LabuteASA"] = Descriptors.LabuteASA(mol)
    for i in range(1, 15):
        desc_name = f"PEOE_VSA{i}"
        desc_func = getattr(Descriptors, desc_name, None)
        descriptors[desc_name] = desc_func(mol) if desc_func else 0.0
    for i in range(1, 11):
        desc_name = f"SMR_VSA{i}"
        desc_func = getattr(Descriptors, desc_name, None)
        descriptors[desc_name] = desc_func(mol) if desc_func else 0.0
    for i in range(1, 13):
        desc_name = f"SlogP_VSA{i}"
        desc_func = getattr(Descriptors, desc_name, None)
        descriptors[desc_name] = desc_func(mol) if desc_func else 0.0
    for i in range(1, 12):
        desc_name = f"EState_VSA{i}"
        desc_func = getattr(Descriptors, desc_name, None)
        descriptors[desc_name] = desc_func(mol) if desc_func else 0.0
    for i in range(1, 11):
        desc_name = f"VSA_EState{i}"
        desc_func = getattr(Descriptors, desc_name, None)
        descriptors[desc_name] = desc_func(mol) if desc_func else 0.0
    descriptors["AvgIpc"] = descriptors["Ipc"] / max(descriptors["NumHeavyAtoms"], 1)
    descriptors["Chi0n_Chi0v"] = (
        descriptors["Chi0n"] - descriptors["Chi0v"] if descriptors["Chi0n"] != 0 else 0
    )
    descriptors["FractionAromaticAtoms"] = len(mol.GetAromaticAtoms()) / max(
        len(mol.GetAtoms()), 1
    )
    ring_info = mol.GetRingInfo()
    aromatic_hetero = 0
    aromatic_carbo = 0
    aliphatic_hetero = 0
    aliphatic_carbo = 0
    for ring in ring_info.AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
        is_aromatic = all(atom.GetIsAromatic() for atom in ring_atoms)
        has_hetero = any(atom.GetAtomicNum() != 6 for atom in ring_atoms)
        if is_aromatic:
            if has_hetero:
                aromatic_hetero += 1
            else:
                aromatic_carbo += 1
        else:
            if has_hetero:
                aliphatic_hetero += 1
            else:
                aliphatic_carbo += 1
    descriptors["NumAromaticHeterocycles"] = aromatic_hetero
    descriptors["NumAromaticCarbocycles"] = aromatic_carbo
    descriptors["NumAliphaticHeterocycles"] = aliphatic_hetero
    descriptors["NumAliphaticCarbocycles"] = aliphatic_carbo
    ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
    descriptors["LargestRingSize"] = max(ring_sizes) if ring_sizes else 0
    return descriptors


def calculate_atom_mol_features(mol):
    if mol is None:
        return {
            k: 0.0
            for k in [
                "C_count",
                "N_count",
                "O_count",
                "S_count",
                "F_count",
                "Cl_count",
                "Br_count",
                "I_count",
                "P_count",
                "NumAromaticAtoms",
                "NumChiralCenters",
                "NumStereocenters",
                "NumSaturatedAtoms",
                "NumHeterocycles",
                "AromaticDensity",
                "HeteroatomDensity",
                "SaturationDensity",
                "AromaticToAliphaticRatio",
            ]
        }
    atoms = mol.GetAtoms()
    total_atoms = len(atoms)
    atom_counts = {
        "C": 0,
        "N": 0,
        "O": 0,
        "S": 0,
        "F": 0,
        "Cl": 0,
        "Br": 0,
        "I": 0,
        "P": 0,
    }
    aromatic_atoms = 0
    chiral_centers = 0
    saturated_atoms = 0
    for atom in atoms:
        symbol = atom.GetSymbol()
        if symbol in atom_counts:
            atom_counts[symbol] += 1
        if atom.GetIsAromatic():
            aromatic_atoms += 1
        if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            chiral_centers += 1
        if (
            atom.GetDegree() >= 4
            and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3
        ):
            saturated_atoms += 1
    ring_info = mol.GetRingInfo()
    heterocycle_atoms = set()
    for ring in ring_info.AtomRings():
        has_hetero = any(
            mol.GetAtomWithIdx(i).GetAtomicNum() not in [1, 6] for i in ring
        )
        if has_hetero:
            heterocycle_atoms.update(ring)
    heterocycles = len(heterocycle_atoms)
    return {
        "C_count": atom_counts["C"],
        "N_count": atom_counts["N"],
        "O_count": atom_counts["O"],
        "S_count": atom_counts["S"],
        "F_count": atom_counts["F"],
        "Cl_count": atom_counts["Cl"],
        "Br_count": atom_counts["Br"],
        "I_count": atom_counts["I"],
        "P_count": atom_counts["P"],
        "NumAromaticAtoms": aromatic_atoms,
        "NumChiralCenters": chiral_centers,
        "NumStereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "NumSaturatedAtoms": saturated_atoms,
        "NumHeterocycles": heterocycles,
        "AromaticDensity": aromatic_atoms / max(total_atoms, 1),
        "HeteroatomDensity": (sum(atom_counts.values()) - atom_counts["C"])
        / max(total_atoms, 1),
        "SaturationDensity": saturated_atoms / max(total_atoms, 1),
        "AromaticToAliphaticRatio": aromatic_atoms
        / max(total_atoms - aromatic_atoms, 1),
    }


def calculate_smiles_features(smiles):
    return {
        "smiles_length": len(smiles),
        "smiles_complexity": len(set(smiles)) / max(len(smiles), 1),
        "smiles_branch_count": smiles.count("(") + smiles.count(")"),
        "smiles_ring_count": smiles.count("1")
        + smiles.count("2")
        + smiles.count("3")
        + smiles.count("4"),
        "smiles_charge_count": smiles.count("+") + smiles.count("-"),
        "smiles_stereo_count": smiles.count("@")
        + smiles.count("/")
        + smiles.count("\\"),
        "smiles_bond_count": smiles.count("=") + smiles.count("#") + smiles.count(":"),
        "smiles_heteroatom_ratio": sum(1 for c in smiles if c in "NOSPFClBrI")
        / max(len(smiles), 1),
        "smiles_digit_ratio": sum(1 for c in smiles if c.isdigit())
        / max(len(smiles), 1),
    }


def process_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    all_features = {}
    all_features.update(calculate_molecular_descriptors(mol))
    all_features.update(calculate_atom_mol_features(mol))
    all_features.update(calculate_smiles_features(smiles))
    morgan_fp = calculate_morgan_fingerprints(mol)
    maccs_fp = calculate_maccs_fingerprints(mol)
    rdkit_fp = calculate_rdkit_fingerprints(mol)
    return all_features, morgan_fp, maccs_fp, rdkit_fp


def create_splits(df, n_splits=5, random_state=42):
    scaffolds = []
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                scaffold = Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except:
                scaffold = smi
        else:
            scaffold = smi
        scaffolds.append(scaffold)
    df_with_scaffolds = df.copy()
    df_with_scaffolds["scaffold"] = scaffolds
    scaffold_groups = df_with_scaffolds.groupby("scaffold").ngroup()
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for train_idx, val_idx in gkf.split(df_with_scaffolds, groups=scaffold_groups):
        splits.append((train_idx, val_idx))
    return splits, df_with_scaffolds


# ========== MODEL DEFINITION ==========
class MultiTaskADMETNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dims=[1024, 512, 256], dropout_rate=0.3, num_tasks=9
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim
        self.shared_backbone = nn.Sequential(*layers)
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dims[-1], 128),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate * 0.5),
                    nn.Linear(128, 1),
                )
                for _ in range(num_tasks)
            ]
        )
        self.num_tasks = num_tasks
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        shared_features = self.shared_backbone(x)
        outputs = [head(shared_features) for head in self.task_heads]
        return torch.cat(outputs, dim=1)


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, mask):
        diff = (predictions - targets) ** 2
        masked_diff = diff * mask.float()
        num_valid = mask.sum().float()
        if num_valid == 0:
            return torch.tensor(0.0, requires_grad=True, device=predictions.device)
        return masked_diff.sum() / num_valid


# ========== TARGET COLUMNS AND LOG TRANSFORM DEFINITIONS ==========
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
LOG_TRANSFORM_TARGETS = {
    "LogD": False,
    "KSOL": True,
    "HLM CLint": True,
    "MLM CLint": True,
    "Caco-2 Permeability Papp A>B": True,
    "Caco-2 Permeability Efflux": True,
    "MPPB": True,
    "MBPB": True,
    "MGMB": True,
}


# ========== DATASET CLASS ==========
class ADMETDataset(Dataset):
    def __init__(self, data_df, feature_cols, target_cols, smiles_col="SMILES", augment=False):
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.augment = augment
        # Store the canonical (original) features
        self.base_features = data_df[feature_cols].values.astype(np.float32)
        # Store SMILES for augmentation
        self.smiles_list = data_df[smiles_col].values.tolist() if smiles_col in data_df.columns else None

        self.target_values = np.zeros(
            (len(data_df), len(target_cols)), dtype=np.float32
        )
        self.target_masks = np.zeros((len(data_df), len(target_cols)), dtype=np.float32)
        for i, col in enumerate(target_cols):
            vals = data_df[col + "_proc"].values
            mask = ~np.isnan(vals)
            self.target_values[mask, i] = vals[mask]
            self.target_masks[mask, i] = 1.0

    def __len__(self):
        return len(self.base_features)

    def __getitem__(self, idx):
        if self.augment and self.smiles_list is not None and np.random.rand() < 0.8:
            # On-the-fly SMILES enumeration: generate random non-canonical SMILES
            smi = self.smiles_list[idx]
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    aug_smi = Chem.MolToSmiles(mol, doRandom=True)
                    # Recompute all features from augmented SMILES
                    features_dict, morgan_fp, maccs_fp, rdkit_fp = process_smiles(aug_smi)
                    # Build feature vector in the same order as feature_cols
                    feature_vec = np.zeros(len(self.feature_cols), dtype=np.float32)
                    for i, col_name in enumerate(self.feature_cols):
                        if col_name in features_dict:
                            feature_vec[i] = features_dict[col_name]
                    # Handle fingerprint features
                    for i, col_name in enumerate(self.feature_cols):
                        if col_name.startswith("MORGAN_"):
                            fp_idx = int(col_name.split("_")[1])
                            feature_vec[i] = morgan_fp[fp_idx]
                        elif col_name.startswith("MACCS_"):
                            fp_idx = int(col_name.split("_")[1])
                            feature_vec[i] = maccs_fp[fp_idx]
                        elif col_name.startswith("RDKIT_"):
                            fp_idx = int(col_name.split("_")[1])
                            feature_vec[i] = rdkit_fp[fp_idx]
                except:
                    feature_vec = self.base_features[idx].copy()
            else:
                feature_vec = self.base_features[idx].copy()
        else:
            feature_vec = self.base_features[idx]

        return {
            "features": torch.FloatTensor(feature_vec),
            "targets": torch.FloatTensor(self.target_values[idx]),
            "masks": torch.FloatTensor(self.target_masks[idx]),
        }


# ========== MAIN EXECUTION ==========
def main():
    print("Loading data...")
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Process training data
    print("Processing training molecules...")
    train_features = []
    train_morgan_fps = []
    train_maccs_fps = []
    train_rdkit_fps = []
    for i, smi in enumerate(train_df["SMILES"]):
        if i % 1000 == 0:
            print(f"  Processing train molecule {i}/{len(train_df)}")
        features, morgan_fp, maccs_fp, rdkit_fp = process_smiles(smi)
        train_features.append(features)
        train_morgan_fps.append(morgan_fp)
        train_maccs_fps.append(maccs_fp)
        train_rdkit_fps.append(rdkit_fp)

    train_desc_df = pd.DataFrame(train_features)
    train_morgan_df = pd.DataFrame(
        train_morgan_fps, columns=[f"MORGAN_{i}" for i in range(2048)]
    )
    train_maccs_df = pd.DataFrame(
        train_maccs_fps, columns=[f"MACCS_{i}" for i in range(167)]
    )
    train_rdkit_fp_df = pd.DataFrame(
        train_rdkit_fps, columns=[f"RDKIT_{i}" for i in range(1024)]
    )
    train_full = pd.concat(
        [
            train_df[["Molecule Name", "SMILES"]],
            train_desc_df,
            train_morgan_df,
            train_maccs_df,
            train_rdkit_fp_df,
            train_df[TARGET_COLUMNS],
        ],
        axis=1,
    )

    # Process test data
    print("Processing test molecules...")
    test_features = []
    test_morgan_fps = []
    test_maccs_fps = []
    test_rdkit_fps = []
    for i, smi in enumerate(test_df["SMILES"]):
        if i % 1000 == 0:
            print(f"  Processing test molecule {i}/{len(test_df)}")
        features, morgan_fp, maccs_fp, rdkit_fp = process_smiles(smi)
        test_features.append(features)
        test_morgan_fps.append(morgan_fp)
        test_maccs_fps.append(maccs_fp)
        test_rdkit_fps.append(rdkit_fp)

    test_desc_df = pd.DataFrame(test_features)
    test_morgan_df = pd.DataFrame(
        test_morgan_fps, columns=[f"MORGAN_{i}" for i in range(2048)]
    )
    test_maccs_df = pd.DataFrame(
        test_maccs_fps, columns=[f"MACCS_{i}" for i in range(167)]
    )
    test_rdkit_fp_df = pd.DataFrame(
        test_rdkit_fps, columns=[f"RDKIT_{i}" for i in range(1024)]
    )
    test_full = pd.concat(
        [
            test_df[["Molecule Name", "SMILES"]],
            test_desc_df,
            test_morgan_df,
            test_maccs_df,
            test_rdkit_fp_df,
        ],
        axis=1,
    )

    # Feature columns
    exclude_cols = ["Molecule Name", "SMILES"] + TARGET_COLUMNS
    feature_cols = [c for c in train_full.columns if c not in exclude_cols]
    print(f"Number of features: {len(feature_cols)}")

    # Create train/validation split
    print("Creating stratified splits...")
    splits, _ = create_splits(train_full)
    train_idx, val_idx = splits[0]
    train_set = train_full.iloc[train_idx].reset_index(drop=True)
    val_set = train_full.iloc[val_idx].reset_index(drop=True)
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    train_set[feature_cols] = scaler.fit_transform(train_set[feature_cols])
    val_set[feature_cols] = scaler.transform(val_set[feature_cols])
    test_full[feature_cols] = scaler.transform(test_full[feature_cols])

    # Target preprocessing (compute stats on training data only)
    target_stats = {}
    for col in TARGET_COLUMNS:
        train_vals = train_set[col].dropna().values
        if LOG_TRANSFORM_TARGETS[col]:
            train_vals = np.log1p(np.maximum(train_vals, 0))
        mean_val = np.mean(train_vals)
        std_val = np.std(train_vals) + 1e-8
        target_stats[col] = {"mean": mean_val, "std": std_val}

    def preprocess_targets(values, col_name):
        v = values.copy()
        mask = ~np.isnan(v)
        if LOG_TRANSFORM_TARGETS[col_name]:
            v[mask] = np.log1p(np.maximum(v[mask], 0))
        stats = target_stats[col_name]
        v[mask] = (v[mask] - stats["mean"]) / stats["std"]
        return v

    def inverse_transform_predictions(preds, col_name):
        stats = target_stats[col_name]
        v = preds * stats["std"] + stats["mean"]
        if LOG_TRANSFORM_TARGETS[col_name]:
            v = np.expm1(np.maximum(v, -10))
        return v

    # Apply preprocessing to train, val, and test sets
    for col in TARGET_COLUMNS:
        train_set[col + "_proc"] = preprocess_targets(train_set[col].values, col)
        val_set[col + "_proc"] = preprocess_targets(val_set[col].values, col)
        test_full[col + "_proc"] = preprocess_targets(
            np.full(len(test_full), np.nan), col
        )

    # Data loaders
    batch_size = 64
    train_dataset = ADMETDataset(train_set, feature_cols, TARGET_COLUMNS)
    val_dataset = ADMETDataset(val_set, feature_cols, TARGET_COLUMNS)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    input_dim = len(feature_cols)
    num_tasks = len(TARGET_COLUMNS)
    model = MultiTaskADMETNet(input_dim=input_dim, num_tasks=num_tasks).to(device)
    criterion = MaskedMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    def compute_ma_rae(predictions, targets, masks, target_cols_list):
        per_task_rae = []
        for i, col in enumerate(target_cols_list):
            task_mask = masks[:, i] == 1
            if task_mask.sum() == 0:
                continue
            task_preds = predictions[task_mask, i]
            task_targets = targets[task_mask, i]
            task_preds_orig = inverse_transform_predictions(task_preds, col)
            task_targets_orig = inverse_transform_predictions(task_targets, col)
            mae = np.mean(np.abs(task_preds_orig - task_targets_orig))
            mean_abs_target = np.mean(np.abs(task_targets_orig))
            rae = mae / max(mean_abs_target, 1e-8)
            per_task_rae.append(rae)
        return np.mean(per_task_rae) if per_task_rae else 0.0

    num_epochs = 200
    best_val_score = float("inf")
    best_model_state = None
    patience = 20
    no_improve_count = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["masks"].to(device)
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)

        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        val_masks = []
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                masks = batch["masks"].to(device)
                predictions = model(features)
                val_predictions.append(predictions.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
                val_masks.append(masks.cpu().numpy())

        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_masks = np.concatenate(val_masks, axis=0)
        val_score = compute_ma_rae(
            val_predictions, val_targets, val_masks, TARGET_COLUMNS
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val MA-RAE: {val_score:.4f}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
            print(f"  -> New best model! Val MA-RAE: {val_score:.4f}")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    print(f"Best validation MA-RAE: {best_val_score:.4f}")

    # Test inference
    print("Generating test predictions...")
    test_dataset = ADMETDataset(test_full, feature_cols, TARGET_COLUMNS)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            predictions = model(features)
            test_predictions.append(predictions.cpu().numpy())
    test_predictions = np.concatenate(test_predictions, axis=0)

    # Inverse transform predictions
    submission_predictions = np.zeros_like(test_predictions)
    for i, col in enumerate(TARGET_COLUMNS):
        submission_predictions[:, i] = inverse_transform_predictions(
            test_predictions[:, i], col
        )
        if col != "LogD":
            submission_predictions[:, i] = np.maximum(submission_predictions[:, i], 0)

    # Save submission
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame()
    submission_df["Molecule Name"] = test_df["Molecule Name"].values
    for i, col in enumerate(TARGET_COLUMNS):
        submission_df[col] = submission_predictions[:, i]
    sample_submission = pd.read_csv("./input/sample_submission.csv")
    submission_df = submission_df[sample_submission.columns.tolist()]
    submission_df.to_csv("./submission/submission.csv", index=False)
    print(
        f"Submission saved to ./submission/submission.csv, shape: {submission_df.shape}"
    )

    # Final validation score
    model.eval()
    val_predictions_final = []
    val_targets_final = []
    val_masks_final = []
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["masks"].to(device)
            predictions = model(features)
            val_predictions_final.append(predictions.cpu().numpy())
            val_targets_final.append(targets.cpu().numpy())
            val_masks_final.append(masks.cpu().numpy())

    val_predictions_final = np.concatenate(val_predictions_final, axis=0)
    val_targets_final = np.concatenate(val_targets_final, axis=0)
    val_masks_final = np.concatenate(val_masks_final, axis=0)
    final_val_score = compute_ma_rae(
        val_predictions_final, val_targets_final, val_masks_final, TARGET_COLUMNS
    )
    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()