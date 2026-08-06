import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
import os, json, random

# ========== SETUP ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Fixed seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ========== DATA LOADING ==========
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# Define feature columns (no underscores)
margin_cols = [f"margin{i}" for i in range(1, 65)]
shape_cols = [f"shape{i}" for i in range(1, 65)]
texture_cols = [f"texture{i}" for i in range(1, 65)]
all_feature_cols = margin_cols + shape_cols + texture_cols
species_cols = sample_sub.columns[1:].tolist()

# Extract raw features
X_train_raw = train_df[all_feature_cols].values.astype(np.float32)
X_test_raw = test_df[all_feature_cols].values.astype(np.float32)
y_train = train_df["species"].values
train_ids = train_df["id"].values
test_ids = test_df["id"].values


# ========== FEATURE ENGINEERING ==========
def engineer_features(X):
    """Add statistical features and group aggregates."""
    n_samples = X.shape[0]
    margin = X[:, :64]
    shape = X[:, 64:128]
    texture = X[:, 128:192]

    features_list = [X]

    for group in [margin, shape, texture]:
        features_list.append(np.mean(group, axis=1, keepdims=True))
        features_list.append(np.std(group, axis=1, keepdims=True))
        features_list.append(np.median(group, axis=1, keepdims=True))
        features_list.append(np.max(group, axis=1, keepdims=True))
        features_list.append(np.min(group, axis=1, keepdims=True))

        mean_centered = group - np.mean(group, axis=1, keepdims=True)
        std = np.std(group, axis=1, keepdims=True) + 1e-8
        features_list.append(np.mean((mean_centered / std) ** 3, axis=1, keepdims=True))
        features_list.append(
            np.mean((mean_centered / std) ** 4, axis=1, keepdims=True) - 3
        )

        features_list.append(np.percentile(group, 10, axis=1, keepdims=True))
        features_list.append(np.percentile(group, 25, axis=1, keepdims=True))
        features_list.append(np.percentile(group, 75, axis=1, keepdims=True))
        features_list.append(np.percentile(group, 90, axis=1, keepdims=True))

    # Energy ratios
    m_energy = np.sum(margin**2, axis=1, keepdims=True) + 1e-8
    s_energy = np.sum(shape**2, axis=1, keepdims=True) + 1e-8
    t_energy = np.sum(texture**2, axis=1, keepdims=True) + 1e-8

    features_list.append(m_energy / (s_energy + t_energy + 1e-8))
    features_list.append(s_energy / (m_energy + t_energy + 1e-8))
    features_list.append(t_energy / (m_energy + s_energy + 1e-8))

    # Smoothness (first differences)
    features_list.append(
        np.mean(np.abs(np.diff(margin, axis=1)), axis=1, keepdims=True)
    )
    features_list.append(np.mean(np.abs(np.diff(shape, axis=1)), axis=1, keepdims=True))
    features_list.append(
        np.mean(np.abs(np.diff(texture, axis=1)), axis=1, keepdims=True)
    )

    return np.concatenate(features_list, axis=1)


X_train_eng = engineer_features(X_train_raw)
X_test_eng = engineer_features(X_test_raw)
print(f"Engineered feature shape: {X_train_eng.shape}")

# ========== STRATIFIED SPLIT ==========
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(X_train_eng, y_train)))

X_train = X_train_eng[train_idx]
X_val = X_train_eng[val_idx]
y_train_split = y_train[train_idx]
y_val = y_train[val_idx]
train_ids_split = train_ids[train_idx]
val_ids_split = train_ids[val_idx]

assert len(set(train_idx) & set(val_idx)) == 0, "Data leakage in split!"

# ========== SCALING ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_eng)

# ========== LABEL ENCODING ==========
unique_species = sorted(set(y_train))
species_to_idx = {sp: i for i, sp in enumerate(unique_species)}
idx_to_species = {i: sp for sp, i in species_to_idx.items()}
n_classes = len(unique_species)

y_train_encoded = np.array([species_to_idx[sp] for sp in y_train_split])
y_val_encoded = np.array([species_to_idx[sp] for sp in y_val])

# ========== IMAGE LOADING ==========
img_size = 256
transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def load_images(ids):
    images = []
    for img_id in ids:
        img_path = f"./input/images/{img_id}.jpg"
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            images.append(transform(img))
        else:
            images.append(torch.zeros(3, img_size, img_size))
    return torch.stack(images)


print("Loading images...")
X_train_images = load_images(train_ids_split)
X_val_images = load_images(val_ids_split)
X_test_images = load_images(test_ids)

# ========== SIGLIP2 FEATURE EXTRACTION ==========
print("Extracting SigLIP2 image features...")
siglip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
siglip_model.eval()
siglip_model.to(device)


def extract_siglip_features(image_tensors, batch_size=16):
    features = []
    with torch.no_grad():
        for i in range(0, len(image_tensors), batch_size):
            batch = image_tensors[i : i + batch_size].to(device)
            with torch.cuda.amp.autocast():
                pooled = siglip_model.get_image_features(pixel_values=batch)
            features.append(pooled.float().cpu())
    return torch.cat(features, dim=0)


train_img_feats = extract_siglip_features(X_train_images)
val_img_feats = extract_siglip_features(X_val_images)
test_img_feats = extract_siglip_features(X_test_images)
print(
    f"Image features: train={train_img_feats.shape}, val={val_img_feats.shape}, test={test_img_feats.shape}"
)

del X_train_images, X_val_images, X_test_images, siglip_model
torch.cuda.empty_cache()


# ========== MODEL DEFINITION ==========
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, l, c = x.shape
        w = torch.mean(x, dim=1)
        w = self.excitation(w).unsqueeze(1)
        return x * w


class TabularEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.se = SEBlock(output_dim)
        self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        res = self.residual(x)
        out = self.net(x)
        out = out.unsqueeze(1)
        out = self.se(out).squeeze(1)
        return out + res


class ImageProjector(nn.Module):
    def __init__(self, input_dim=1152, hidden_dim=512, output_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class LeafFusionModel(nn.Module):
    def __init__(
        self,
        tabular_dim=64,
        image_dim=1152,
        num_classes=99,
        hidden_dim=384,
        dropout=0.3,
        label_smoothing=0.1,
    ):
        super().__init__()
        self.margin_encoder = TabularEncoder(tabular_dim, hidden_dim, 128, dropout)
        self.shape_encoder = TabularEncoder(tabular_dim, hidden_dim, 128, dropout)
        self.texture_encoder = TabularEncoder(tabular_dim, hidden_dim, 128, dropout)
        self.image_projector = ImageProjector(image_dim, 512, 256, dropout)

        fusion_dim = 384 + 256
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_dim, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, margin, shape, texture, image_features):
        margin_emb = self.margin_encoder(margin)
        shape_emb = self.shape_encoder(shape)
        texture_emb = self.texture_encoder(texture)
        image_emb = self.image_projector(image_features)
        fused = torch.cat([margin_emb, shape_emb, texture_emb, image_emb], dim=1)
        return self.classifier(fused)


# ========== MODEL INITIALIZATION ==========
model = LeafFusionModel(
    tabular_dim=64,
    image_dim=1152,
    num_classes=n_classes,
    hidden_dim=384,
    dropout=0.3,
    label_smoothing=0.1,
).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

# ========== DATA PREPARATION ==========
# Split tabular features: first 64 margin, 64:128 shape, 128:192 texture, rest engineered
X_train_margin = torch.tensor(X_train_scaled[:, :64], dtype=torch.float32)
X_train_shape = torch.tensor(X_train_scaled[:, 64:128], dtype=torch.float32)
X_train_texture = torch.tensor(X_train_scaled[:, 128:192], dtype=torch.float32)

X_val_margin = torch.tensor(X_val_scaled[:, :64], dtype=torch.float32)
X_val_shape = torch.tensor(X_val_scaled[:, 64:128], dtype=torch.float32)
X_val_texture = torch.tensor(X_val_scaled[:, 128:192], dtype=torch.float32)

X_test_margin = torch.tensor(X_test_scaled[:, :64], dtype=torch.float32)
X_test_shape = torch.tensor(X_test_scaled[:, 64:128], dtype=torch.float32)
X_test_texture = torch.tensor(X_test_scaled[:, 128:192], dtype=torch.float32)

y_train_t = torch.tensor(y_train_encoded, dtype=torch.long)
y_val_t = torch.tensor(y_val_encoded, dtype=torch.long)

# ========== DATALOADERS ==========
train_dataset = TensorDataset(
    X_train_margin, X_train_shape, X_train_texture, train_img_feats, y_train_t
)
val_dataset = TensorDataset(
    X_val_margin, X_val_shape, X_val_texture, val_img_feats, y_val_t
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)


# ========== MIXUP ==========
def mixup_data(x_margin, x_shape, x_texture, img_feats, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x_margin.size(0)
    index = torch.randperm(batch_size).to(x_margin.device)
    mixed_margin = lam * x_margin + (1 - lam) * x_margin[index]
    mixed_shape = lam * x_shape + (1 - lam) * x_shape[index]
    mixed_texture = lam * x_texture + (1 - lam) * x_texture[index]
    mixed_img = lam * img_feats + (1 - lam) * img_feats[index]
    return mixed_margin, mixed_shape, mixed_texture, mixed_img, y, y[index], lam


# ========== TRAINING LOOP ==========
epochs = 50
patience = 15
best_val_logloss = float("inf")
best_epoch = -1
no_improve = 0
scaler_amp = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        x_margin, x_shape, x_texture, img_feats, targets = [b.to(device) for b in batch]

        x_margin_m, x_shape_m, x_texture_m, img_feats_m, targets_a, targets_b, lam = (
            mixup_data(x_margin, x_shape, x_texture, img_feats, targets, alpha=0.2)
        )

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(x_margin_m, x_shape_m, x_texture_m, img_feats_m)
            loss = lam * F.cross_entropy(logits, targets_a, label_smoothing=0.1) + (
                1 - lam
            ) * F.cross_entropy(logits, targets_b, label_smoothing=0.1)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        train_loss += loss.item()
        n_batches += 1

    avg_train_loss = train_loss / max(n_batches, 1)

    # Validation
    model.eval()
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            x_margin, x_shape, x_texture, img_feats, targets = [
                b.to(device) for b in batch
            ]
            with torch.cuda.amp.autocast():
                logits = model(x_margin, x_shape, x_texture, img_feats)
                probs = F.softmax(logits, dim=1)
            val_probs.append(probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_probs = np.concatenate(val_probs, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
    val_logloss = -np.mean(
        np.log(val_probs_clipped[np.arange(len(val_targets)), val_targets])
    )
    val_acc = (val_probs.argmax(axis=1) == val_targets).mean()

    scheduler.step()

    print(
        f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_logloss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_logloss < best_val_logloss:
        best_val_logloss = val_logloss
        best_epoch = epoch + 1
        no_improve = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(
            f"Early stopping at epoch {epoch+1}, best was epoch {best_epoch} with logloss {best_val_logloss:.4f}"
        )
        break

# ========== LOAD BEST MODEL & FINAL PREDICTIONS ==========
model.load_state_dict(torch.load("./working/best_model.pt", map_location=device))
model.eval()

# Validation predictions
val_final_probs = []
with torch.no_grad():
    for batch in val_loader:
        x_margin, x_shape, x_texture, img_feats, _ = [b.to(device) for b in batch]
        with torch.cuda.amp.autocast():
            logits = model(x_margin, x_shape, x_texture, img_feats)
            probs = F.softmax(logits, dim=1)
        val_final_probs.append(probs.cpu().numpy())
val_final_probs = np.concatenate(val_final_probs, axis=0)

eps = 1e-15
val_probs_clipped = np.clip(val_final_probs, eps, 1 - eps)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
final_val_logloss = -np.mean(
    np.log(val_probs_clipped[np.arange(len(y_val_encoded)), y_val_encoded])
)
print(f"Final validation log loss: {final_val_logloss:.6f}")

# Test predictions
test_dataset = TensorDataset(
    X_test_margin,
    X_test_shape,
    X_test_texture,
    test_img_feats,
    torch.zeros(len(X_test_margin), dtype=torch.long),
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

test_final_probs = []
with torch.no_grad():
    for batch in test_loader:
        x_margin, x_shape, x_texture, img_feats, _ = [b.to(device) for b in batch]
        with torch.cuda.amp.autocast():
            logits = model(x_margin, x_shape, x_texture, img_feats)
            probs = F.softmax(logits, dim=1)
        test_final_probs.append(probs.cpu().numpy())
test_final_probs = np.concatenate(test_final_probs, axis=0)

# ========== CREATE SUBMISSION ==========
test_preds = np.clip(test_final_probs, eps, 1 - eps)
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# Map to species order from sample submission
model_species_order = [idx_to_species[i] for i in range(n_classes)]
reordered_test_probs = np.zeros((len(test_ids), len(species_cols)))
for i, species in enumerate(model_species_order):
    if species in species_cols:
        col_idx = species_cols.index(species)
        reordered_test_probs[:, col_idx] = test_preds[:, i]

row_sums = reordered_test_probs.sum(axis=1, keepdims=True)
reordered_test_probs = reordered_test_probs / row_sums

submission_df = pd.DataFrame(reordered_test_probs, columns=species_cols)
submission_df.insert(0, "id", test_ids)
submission_df.to_csv("./submission/submission.csv", index=False)
print(
    f"Submission saved to ./submission/submission.csv with shape {submission_df.shape}"
)

print(f"Final Validation Score: {final_val_logloss}")
