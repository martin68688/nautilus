import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew, kurtosis
from PIL import Image
import torchvision.transforms as T
from transformers import AutoModel, AutoProcessor

warnings.filterwarnings("ignore")

# ============================================
# CONFIGURATION
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 64
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 20
LR = 2e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================
# 1. LOAD DATA
# ============================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(
    f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Sample sub: {sample_sub.shape}"
)


# ============================================
# 2. NORMALIZE COLUMN NAMES
# ============================================
def normalize_col_names(df):
    cols = {}
    for c in df.columns:
        new_c = c.replace("_", "").lower()
        cols[c] = new_c
    return df.rename(columns=cols)


train_df = normalize_col_names(train_df)
test_df = normalize_col_names(test_df)

# Identify feature columns for each group
margin_cols = [c for c in train_df.columns if "margin" in c and c != "id"]
shape_cols = [c for c in train_df.columns if "shape" in c and c != "id"]
texture_cols = [c for c in train_df.columns if "texture" in c and c != "id"]

# Sort columns numerically
margin_cols.sort(key=lambda x: int("".join(filter(str.isdigit, x))))
shape_cols.sort(key=lambda x: int("".join(filter(str.isdigit, x))))
texture_cols.sort(key=lambda x: int("".join(filter(str.isdigit, x))))

print(
    f"Margin cols: {len(margin_cols)}, Shape cols: {len(shape_cols)}, Texture cols: {len(texture_cols)}"
)


# ============================================
# 3. FEATURE ENGINEERING
# ============================================
def engineer_features(
    df,
    margin_cols,
    shape_cols,
    texture_cols,
    is_train=True,
    pca_models=None,
    scalers=None,
):
    feat_dict = {}
    all_groups = {"margin": margin_cols, "shape": shape_cols, "texture": texture_cols}

    if pca_models is None:
        pca_models = {}
    if scalers is None:
        scalers = {}

    all_feat_names = []
    all_feat_arrays = []

    for group_name, cols in all_groups.items():
        group_data = df[cols].values.astype(np.float64)

        # Statistical features per group
        group_stats = []
        for row in group_data:
            mean_val = np.mean(row)
            std_val = np.std(row)
            var_val = np.var(row)
            entropy = -np.sum(row * np.log(row + 1e-10))
            row_norm = row / (np.sum(row) + 1e-10)
            entropy_norm = -np.sum(row_norm * np.log(row_norm + 1e-10))
            pcts = np.percentile(row, [10, 25, 50, 75, 90])
            sk = skew(row)
            kur = kurtosis(row)
            rng = np.max(row) - np.min(row)
            iqr = pcts[3] - pcts[1]
            cv = std_val / (mean_val + 1e-10)
            energy = np.sqrt(np.sum(row**2))
            l2_norm = np.linalg.norm(row)
            peak_val = np.max(row)
            peak_pos = np.argmax(row)
            peak_ratio = peak_val / (np.sum(row) + 1e-10)
            sign_changes = np.sum(np.diff(np.signbit(row - mean_val)))

            group_stats.append(
                [
                    mean_val,
                    std_val,
                    var_val,
                    entropy,
                    entropy_norm,
                    pcts[0],
                    pcts[1],
                    pcts[2],
                    pcts[3],
                    pcts[4],
                    sk,
                    kur,
                    rng,
                    iqr,
                    cv,
                    energy,
                    l2_norm,
                    peak_val,
                    peak_pos,
                    peak_ratio,
                    sign_changes,
                ]
            )

        group_stats = np.array(group_stats)
        feat_dict[f"{group_name}_stats"] = group_stats
        all_feat_arrays.append(group_stats)
        all_feat_names.extend(
            [f"{group_name}_stat_{i}" for i in range(group_stats.shape[1])]
        )

        # PCA features
        if is_train:
            scaler = StandardScaler()
            group_scaled = scaler.fit_transform(group_data)
            pca = PCA(n_components=min(32, len(cols)))
            pca_components = pca.fit_transform(group_scaled)
            pca_models[group_name] = pca
            scalers[group_name] = scaler
        else:
            group_scaled = scalers[group_name].transform(group_data)
            pca_components = pca_models[group_name].transform(group_scaled)

        feat_dict[f"{group_name}_pca"] = pca_components
        all_feat_arrays.append(pca_components)
        all_feat_names.extend(
            [f"{group_name}_pca_{i}" for i in range(pca_components.shape[1])]
        )

    # Cross-group correlation features
    margin_l2 = feat_dict["margin_stats"]
    shape_l2 = feat_dict["shape_stats"]
    texture_l2 = feat_dict["texture_stats"]

    cross_corr = []
    for i in range(len(margin_l2)):
        corr_ms = (
            np.corrcoef(margin_l2[i][:5], shape_l2[i][:5])[0, 1]
            if np.std(margin_l2[i][:5]) > 0 and np.std(shape_l2[i][:5]) > 0
            else 0
        )
        corr_mt = (
            np.corrcoef(margin_l2[i][:5], texture_l2[i][:5])[0, 1]
            if np.std(margin_l2[i][:5]) > 0 and np.std(texture_l2[i][:5]) > 0
            else 0
        )
        corr_st = (
            np.corrcoef(shape_l2[i][:5], texture_l2[i][:5])[0, 1]
            if np.std(shape_l2[i][:5]) > 0 and np.std(texture_l2[i][:5]) > 0
            else 0
        )
        cross_corr.append(
            [
                corr_ms,
                corr_mt,
                corr_st,
                corr_ms * corr_mt,
                corr_ms * corr_st,
                corr_mt * corr_st,
            ]
        )

    cross_corr = np.array(cross_corr)
    all_feat_arrays.append(cross_corr)
    all_feat_names.extend([f"corr_{i}" for i in range(cross_corr.shape[1])])

    # Group sums
    margin_sum = np.sum(margin_l2, axis=1, keepdims=True)
    shape_sum = np.sum(shape_l2, axis=1, keepdims=True)
    texture_sum = np.sum(texture_l2, axis=1, keepdims=True)
    group_sums = np.hstack([margin_sum, shape_sum, texture_sum])
    all_feat_arrays.append(group_sums)
    all_feat_names.extend(["sum_margin", "sum_shape", "sum_texture"])

    # Difference statistics
    margin_diff = np.abs(np.diff(margin_l2, axis=1))
    shape_diff = np.abs(np.diff(shape_l2, axis=1))
    texture_diff = np.abs(np.diff(texture_l2, axis=1))

    diff_stats = []
    for i in range(len(margin_diff)):
        diff_stats.append(
            [
                np.mean(margin_diff[i]),
                np.std(margin_diff[i]),
                np.max(margin_diff[i]),
                np.mean(shape_diff[i]),
                np.std(shape_diff[i]),
                np.max(shape_diff[i]),
                np.mean(texture_diff[i]),
                np.std(texture_diff[i]),
                np.max(texture_diff[i]),
            ]
        )
    diff_stats = np.array(diff_stats)
    all_feat_arrays.append(diff_stats)
    all_feat_names.extend([f"diff_{i}" for i in range(diff_stats.shape[1])])

    engineered = np.hstack(all_feat_arrays)
    return engineered, all_feat_names, pca_models, scalers


# ============================================
# 4. LABEL ENCODING AND SPLIT
# ============================================
le = LabelEncoder()
train_df["species_encoded"] = le.fit_transform(train_df["species"])
label_classes = le.classes_
n_classes = len(label_classes)

y_full = train_df["species_encoded"].values

# Stratified split - save indices
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(sss.split(train_df, y_full))
train_split_indices = train_idx
val_split_indices = val_idx

# Sanity check
assert len(set(train_idx) & set(val_idx)) == 0, "Data leakage in split!"

print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

# ============================================
# 5. ENGINEER FEATURES FOR TRAIN/VAL/TEST
# ============================================
train_engineered, feat_names, pca_models, scalers = engineer_features(
    train_df.iloc[train_idx], margin_cols, shape_cols, texture_cols, is_train=True
)
val_engineered, _, _, _ = engineer_features(
    train_df.iloc[val_idx],
    margin_cols,
    shape_cols,
    texture_cols,
    is_train=False,
    pca_models=pca_models,
    scalers=scalers,
)
test_engineered, _, _, _ = engineer_features(
    test_df,
    margin_cols,
    shape_cols,
    texture_cols,
    is_train=False,
    pca_models=pca_models,
    scalers=scalers,
)

# Standard scale all engineered features
scaler_final = StandardScaler()
train_engineered_scaled = scaler_final.fit_transform(train_engineered)
val_engineered_scaled = scaler_final.transform(val_engineered)
test_engineered_scaled = scaler_final.transform(test_engineered)

# Feature selection using F-score (fit on train only)
selector = SelectKBest(f_classif, k=min(500, train_engineered_scaled.shape[1]))
train_selected = selector.fit_transform(train_engineered_scaled, y_full[train_idx])
val_selected = selector.transform(val_engineered_scaled)
test_selected = selector.transform(test_engineered_scaled)

# ============================================
# 6. SIGLIP2 IMAGE FEATURE EXTRACTION
# ============================================
print("Loading SigLIP2 model...")
siglip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
siglip_model.to(device)
siglip_model.eval()
for param in siglip_model.parameters():
    param.requires_grad = False

processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-256")

siglip_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


def extract_image_features(df, ids):
    """Extract SigLIP2 features for given image IDs"""
    features = []
    for img_id in ids:
        img_path = f"./input/images/{int(img_id)}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = siglip_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = siglip_model.get_image_features(pixel_values=img_tensor)
            features.append(feat.cpu().numpy().squeeze())
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            features.append(np.zeros(1152, dtype=np.float32))
    return np.vstack(features)


print("Extracting image features for train/val/test...")
train_img_ids = train_df.iloc[train_idx]["id"].values
val_img_ids = train_df.iloc[val_idx]["id"].values
test_img_ids = test_df["id"].values

cache_path = "./working/siglip_features.npz"
if os.path.exists(cache_path):
    cached = np.load(cache_path)
    train_img_feat = cached["train_img_feat"]
    val_img_feat = cached["val_img_feat"]
    test_img_feat = cached["test_img_feat"]
    print("Loaded cached image features")
else:
    train_img_feat = extract_image_features(train_df, train_img_ids)
    val_img_feat = extract_image_features(train_df, val_img_ids)
    test_img_feat = extract_image_features(test_df, test_img_ids)
    np.savez(
        cache_path,
        train_img_feat=train_img_feat,
        val_img_feat=val_img_feat,
        test_img_feat=test_img_feat,
    )
    print("Image features extracted and cached")

print(
    f"Image feature shapes: train={train_img_feat.shape}, val={val_img_feat.shape}, test={test_img_feat.shape}"
)


# ============================================
# 7. PREPARE RAW TABULAR FEATURES (L2-normalized per group)
# ============================================
def get_raw_tabular(df, indices=None):
    if indices is not None:
        df_sub = df.iloc[indices]
    else:
        df_sub = df

    margin = df_sub[margin_cols].values.astype(np.float32)
    shape = df_sub[shape_cols].values.astype(np.float32)
    texture = df_sub[texture_cols].values.astype(np.float32)

    margin = margin / (np.linalg.norm(margin, axis=1, keepdims=True) + 1e-10)
    shape = shape / (np.linalg.norm(shape, axis=1, keepdims=True) + 1e-10)
    texture = texture / (np.linalg.norm(texture, axis=1, keepdims=True) + 1e-10)

    return margin, shape, texture


train_margin, train_shape, train_texture = get_raw_tabular(train_df, train_idx)
val_margin, val_shape, val_texture = get_raw_tabular(train_df, val_idx)
test_margin, test_shape, test_texture = get_raw_tabular(test_df)


# ============================================
# 8. MODEL DEFINITION - Multi-Branch Fusion
# ============================================
class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=192, output_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        return self.net(x)


class LeafFusionModel(nn.Module):
    def __init__(
        self, n_classes=99, tabular_embed_dim=128, image_embed_dim=256, dropout=0.35
    ):
        super().__init__()
        self.margin_encoder = TabularEncoder(
            64, hidden_dim=192, output_dim=tabular_embed_dim, dropout=dropout
        )
        self.shape_encoder = TabularEncoder(
            64, hidden_dim=192, output_dim=tabular_embed_dim, dropout=dropout
        )
        self.texture_encoder = TabularEncoder(
            64, hidden_dim=192, output_dim=tabular_embed_dim, dropout=dropout
        )

        self.image_proj = nn.Sequential(
            nn.Linear(1152, image_embed_dim),
            nn.LayerNorm(image_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
        )

        fusion_dim = 3 * tabular_embed_dim + image_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if isinstance(self.classifier[-1], nn.Linear):
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, margin, shape, texture, image_feat):
        margin_emb = self.margin_encoder(margin)
        shape_emb = self.shape_encoder(shape)
        texture_emb = self.texture_encoder(texture)
        image_emb = self.image_proj(image_feat)
        fused = torch.cat([margin_emb, shape_emb, texture_emb, image_emb], dim=1)
        return self.classifier(fused)


# ============================================
# 9. CREATE DATA LOADERS
# ============================================
model = LeafFusionModel(n_classes=n_classes).to(device)

train_dataset = TensorDataset(
    torch.FloatTensor(train_margin),
    torch.FloatTensor(train_shape),
    torch.FloatTensor(train_texture),
    torch.FloatTensor(train_img_feat),
    torch.LongTensor(y_full[train_idx]),
)
val_dataset = TensorDataset(
    torch.FloatTensor(val_margin),
    torch.FloatTensor(val_shape),
    torch.FloatTensor(val_texture),
    torch.FloatTensor(val_img_feat),
    torch.LongTensor(y_full[val_idx]),
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# ============================================
# 10. LOSS, OPTIMIZER, SCHEDULER
# ============================================
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if len(param.shape) < 2 or "bias" in name or "norm" in name or "ln" in name.lower():
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = AdamW(
    [
        {"params": decay_params, "lr": LR, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "lr": LR, "weight_decay": 0.0},
    ]
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=LR * 0.01
)

scaler = GradScaler()

# ============================================
# 11. TRAINING LOOP
# ============================================
best_val_ll = float("inf")
best_epoch = -1
patience_counter = 0

print(
    f"Starting training: {len(train_dataset)} train samples, {len(val_dataset)} val samples"
)
print(
    f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for margin_b, shape_b, texture_b, img_b, labels_b in train_loader:
        margin_b = margin_b.to(device)
        shape_b = shape_b.to(device)
        texture_b = texture_b.to(device)
        img_b = img_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(margin_b, shape_b, texture_b, img_b)
            loss = criterion(logits, labels_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * margin_b.size(0)
        train_correct += (logits.argmax(dim=1) == labels_b).sum().item()
        train_total += margin_b.size(0)

    scheduler.step()

    # Validation
    model.eval()
    val_probs = []
    val_labels_list = []

    with torch.no_grad():
        for margin_b, shape_b, texture_b, img_b, labels_b in val_loader:
            margin_b = margin_b.to(device)
            shape_b = shape_b.to(device)
            texture_b = texture_b.to(device)
            img_b = img_b.to(device)

            with autocast():
                logits = model(margin_b, shape_b, texture_b, img_b)
                probs = F.softmax(logits, dim=1).float()

            val_probs.append(probs.cpu().numpy())
            val_labels_list.append(labels_b.numpy())

    val_probs = np.vstack(val_probs)
    val_labels = np.concatenate(val_labels_list)

    val_ll = log_loss(val_labels, val_probs, labels=np.arange(n_classes))
    train_acc = train_correct / train_total
    train_loss_avg = train_loss / train_total

    print(
        f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | Val LogLoss: {val_ll:.4f}"
    )

    if val_ll < best_val_ll:
        best_val_ll = val_ll
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (best was epoch {best_epoch+1})")
            break

# ============================================
# 12. LOAD BEST MODEL AND EVALUATE
# ============================================
print(f"\nLoading best model from epoch {best_epoch+1}")
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

val_probs = []
with torch.no_grad():
    for margin_b, shape_b, texture_b, img_b, _ in val_loader:
        margin_b = margin_b.to(device)
        shape_b = shape_b.to(device)
        texture_b = texture_b.to(device)
        img_b = img_b.to(device)

        with autocast():
            logits = model(margin_b, shape_b, texture_b, img_b)
            probs = F.softmax(logits, dim=1).float()

        val_probs.append(probs.cpu().numpy())

val_probs = np.vstack(val_probs)
score = log_loss(y_full[val_idx], val_probs, labels=np.arange(n_classes))
print(f"Best validation log loss: {score:.4f}")

# ============================================
# 13. TEST INFERENCE
# ============================================
print("Performing test inference...")
test_dataset = TensorDataset(
    torch.FloatTensor(test_margin),
    torch.FloatTensor(test_shape),
    torch.FloatTensor(test_texture),
    torch.FloatTensor(test_img_feat),
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

test_probs = []
with torch.no_grad():
    for margin_b, shape_b, texture_b, img_b in test_loader:
        margin_b = margin_b.to(device)
        shape_b = shape_b.to(device)
        texture_b = texture_b.to(device)
        img_b = img_b.to(device)

        with autocast():
            logits = model(margin_b, shape_b, texture_b, img_b)
            probs = F.softmax(logits, dim=1).float()

        test_probs.append(probs.cpu().numpy())

test_probs = np.vstack(test_probs)
print(f"Test predictions shape: {test_probs.shape}")

# ============================================
# 14. GENERATE SUBMISSION
# ============================================
submission = pd.DataFrame(test_probs, columns=label_classes)
submission.insert(0, "id", test_img_ids.astype(int))

# Reorder columns to match sample submission exactly
submission = submission[sample_sub.columns]

# Clip probabilities and normalize rows
submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(1e-15, 1 - 1e-15)
row_sums = submission.iloc[:, 1:].sum(axis=1)
submission.iloc[:, 1:] = submission.iloc[:, 1:].div(row_sums, axis=0)

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")

print(f"Final Validation Score: {score}")
