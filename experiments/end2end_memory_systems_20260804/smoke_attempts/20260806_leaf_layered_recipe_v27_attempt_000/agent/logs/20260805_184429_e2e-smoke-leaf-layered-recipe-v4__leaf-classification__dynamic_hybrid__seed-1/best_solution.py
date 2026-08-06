import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import warnings
import os

warnings.filterwarnings("ignore")

# ==================== Paths and Device ====================
input_dir = Path("./input")
images_dir = input_dir / "images"
submission_dir = Path("./submission")
submission_dir.mkdir(exist_ok=True)
working_dir = Path("./working")
working_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== Data Loading and Preprocessing ====================
print("Loading data...")
train_df = pd.read_csv(input_dir / "train.csv")
test_df = pd.read_csv(input_dir / "test.csv")
sample_sub = pd.read_csv(input_dir / "sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Feature columns
margin_cols = [f"margin{i}" for i in range(1, 65)]
shape_cols = [f"shape{i}" for i in range(1, 65)]
texture_cols = [f"texture{i}" for i in range(1, 65)]

# Verify columns
for col_list in [margin_cols, shape_cols, texture_cols]:
    assert all(c in train_df.columns for c in col_list), f"Missing columns"

# Labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["species"].values)
class_names = label_encoder.classes_
n_classes = len(class_names)
print(f"Number of classes: {n_classes}")

# Raw features
X_margin_raw = train_df[margin_cols].values.astype(np.float32)
X_shape_raw = train_df[shape_cols].values.astype(np.float32)
X_texture_raw = train_df[texture_cols].values.astype(np.float32)
X_margin_test_raw = test_df[margin_cols].values.astype(np.float32)
X_shape_test_raw = test_df[shape_cols].values.astype(np.float32)
X_texture_test_raw = test_df[texture_cols].values.astype(np.float32)

train_ids = train_df["id"].values
test_ids = test_df["id"].values


# Create engineered features
def create_eng_features(margin, shape, texture):
    features_list = []
    for feature_mat in [margin, shape, texture]:
        features_list.append(np.mean(feature_mat, axis=1, keepdims=True))
        features_list.append(np.std(feature_mat, axis=1, keepdims=True))
        features_list.append(np.percentile(feature_mat, 25, axis=1, keepdims=True))
        features_list.append(np.percentile(feature_mat, 75, axis=1, keepdims=True))
        features_list.append(np.median(feature_mat, axis=1, keepdims=True))

    corr_ms = np.array([np.corrcoef(m, s)[0, 1] for m, s in zip(margin, shape)])
    corr_mt = np.array([np.corrcoef(m, t)[0, 1] for m, t in zip(margin, texture)])
    corr_st = np.array([np.corrcoef(s, t)[0, 1] for s, t in zip(shape, texture)])
    features_list.extend(
        [corr_ms.reshape(-1, 1), corr_mt.reshape(-1, 1), corr_st.reshape(-1, 1)]
    )
    features_list.append(np.linalg.norm(margin, axis=1, keepdims=True))
    features_list.append(np.linalg.norm(shape, axis=1, keepdims=True))
    features_list.append(np.linalg.norm(texture, axis=1, keepdims=True))
    return np.concatenate(features_list, axis=1)


eng_train = create_eng_features(X_margin_raw, X_shape_raw, X_texture_raw)
eng_test = create_eng_features(X_margin_test_raw, X_shape_test_raw, X_texture_test_raw)
eng_dim = eng_train.shape[1]
print(f"Engineered features shape: {eng_train.shape}")

# Image loading
img_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def load_images(img_ids):
    imgs = []
    for img_id in img_ids:
        p = images_dir / f"{img_id}.jpg"
        if p.exists():
            img = Image.open(p).convert("RGB")
            imgs.append(img_transform(img).numpy())
        else:
            imgs.append(np.zeros((3, 256, 256), dtype=np.float32))
    return np.stack(imgs)


print("Loading images...")
train_images_array = load_images(train_ids)
test_images_array = load_images(test_ids)
print(f"Train images shape: {train_images_array.shape}")

# ==================== SigLIP2 Embedding Extraction ====================
from transformers import AutoProcessor, AutoModel

print("Loading SigLIP2 model...")
siglip_processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-256")
siglip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
siglip_model = siglip_model.to(device)
siglip_model.eval()


def process_batch_for_siglip(images_tensor):
    """Convert [B,3,256,256] tensor to processor inputs"""
    denorm = images_tensor * 0.5 + 0.5
    pil_images = []
    for j in range(len(denorm)):
        img = denorm[j].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img))
    return pil_images


def extract_siglip_features(images_array):
    """Extract SigLIP2 embeddings [N,3,256,256] -> [N,1152]"""
    all_features = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(images_array), batch_size):
            batch = images_array[i : i + batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            pil_images = process_batch_for_siglip(batch_tensor)
            if len(pil_images) > 0:
                inputs = siglip_processor(images=pil_images, return_tensors="pt").to(
                    device
                )
                pooled = siglip_model.get_image_features(**inputs)
                all_features.append(pooled.cpu().numpy())
    return np.vstack(all_features)


print("Extracting SigLIP2 embeddings...")
# Process in chunks to manage memory
n_train = len(train_ids)
n_test = len(test_ids)

train_img_feats = extract_siglip_features(train_images_array)
test_img_feats = extract_siglip_features(test_images_array)
print(
    f"Train image features: {train_img_feats.shape}, Test image features: {test_img_feats.shape}"
)


# ==================== Model Definition ====================
class SigLIP2Branch(nn.Module):
    def __init__(self, img_dim=1152, latent_dim=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.proj(x)


class TabularBranch(nn.Module):
    def __init__(self, input_dim=64, latent_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class ModalityAttentionFusion(nn.Module):
    def __init__(self, latent_dim=256, n_heads=4, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, tokens):
        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out
        ffn_out = self.ffn(self.norm2(tokens))
        tokens = tokens + ffn_out
        return tokens.mean(dim=1)


class LeafMultimodalClassifier(nn.Module):
    def __init__(
        self,
        num_classes=99,
        latent_dim=256,
        n_heads=4,
        dropout=0.4,
        img_dim=1152,
        use_engineered=True,
        eng_dim=35,
    ):
        super().__init__()
        self.margin_branch = TabularBranch(
            input_dim=64, latent_dim=latent_dim, dropout=dropout
        )
        self.shape_branch = TabularBranch(
            input_dim=64, latent_dim=latent_dim, dropout=dropout
        )
        self.texture_branch = TabularBranch(
            input_dim=64, latent_dim=latent_dim, dropout=dropout
        )
        self.siglip2_branch = SigLIP2Branch(
            img_dim=img_dim, latent_dim=latent_dim, dropout=dropout
        )

        self.use_engineered = use_engineered
        if use_engineered:
            self.eng_proj = nn.Sequential(
                nn.Linear(eng_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, latent_dim),
                nn.LayerNorm(latent_dim),
            )

        self.fusion = ModalityAttentionFusion(
            latent_dim=latent_dim, n_heads=n_heads, dropout=dropout * 0.5
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(latent_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, margin, shape, texture, image_feat, engineered=None):
        margin_tok = self.margin_branch(margin)
        shape_tok = self.shape_branch(shape)
        texture_tok = self.texture_branch(texture)
        image_tok = self.siglip2_branch(image_feat)
        tokens = torch.stack([margin_tok, shape_tok, texture_tok, image_tok], dim=1)
        fused = self.fusion(tokens)

        if self.use_engineered and engineered is not None:
            eng_tok = self.eng_proj(engineered)
            fused = fused + 0.5 * eng_tok

        logits = self.classifier(fused)
        return logits


# ==================== Setup Folds ====================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(skf.split(np.arange(n_train), train_labels_encoded))

folds_data = []
for fold_idx, (tr_idx, va_idx) in enumerate(fold_indices):
    fold_data = {
        "train_idx": tr_idx,
        "val_idx": va_idx,
        "X_margin_train": X_margin_raw[tr_idx],
        "X_shape_train": X_shape_raw[tr_idx],
        "X_texture_train": X_texture_raw[tr_idx],
        "X_eng_train": eng_train[tr_idx],
        "X_margin_val": X_margin_raw[va_idx],
        "X_shape_val": X_shape_raw[va_idx],
        "X_texture_val": X_texture_raw[va_idx],
        "X_eng_val": eng_train[va_idx],
        "y_train": train_labels_encoded[tr_idx],
        "y_val": train_labels_encoded[va_idx],
        "img_feat_train": train_img_feats[tr_idx],
        "img_feat_val": train_img_feats[va_idx],
    }
    folds_data.append(fold_data)

test_data = {
    "test_ids": test_ids,
    "X_margin_test": X_margin_test_raw,
    "X_shape_test": X_shape_test_raw,
    "X_texture_test": X_texture_test_raw,
    "X_eng_test": eng_test,
    "img_feat_test": test_img_feats,
}

# ==================== Training Setup ====================
EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12
LABEL_SMOOTHING = 0.1

n_folds = len(folds_data)
OOF_preds = np.zeros((n_train, n_classes), dtype=np.float32)
test_preds = np.zeros((n_test, n_classes), dtype=np.float32)
best_val_losses = []

# ==================== Training Loop ====================
for fold_idx, fd in enumerate(folds_data):
    print(f"\n=== Fold {fold_idx+1}/{n_folds} ===")

    # Scale tabular features on this fold
    scaler_m = StandardScaler().fit(fd["X_margin_train"])
    scaler_s = StandardScaler().fit(fd["X_shape_train"])
    scaler_t = StandardScaler().fit(fd["X_texture_train"])
    scaler_e = StandardScaler().fit(fd["X_eng_train"])

    X_m_train = scaler_m.transform(fd["X_margin_train"]).astype(np.float32)
    X_s_train = scaler_s.transform(fd["X_shape_train"]).astype(np.float32)
    X_t_train = scaler_t.transform(fd["X_texture_train"]).astype(np.float32)
    X_e_train = scaler_e.transform(fd["X_eng_train"]).astype(np.float32)

    X_m_val = scaler_m.transform(fd["X_margin_val"]).astype(np.float32)
    X_s_val = scaler_s.transform(fd["X_shape_val"]).astype(np.float32)
    X_t_val = scaler_t.transform(fd["X_texture_val"]).astype(np.float32)
    X_e_val = scaler_e.transform(fd["X_eng_val"]).astype(np.float32)

    X_m_test = scaler_m.transform(test_data["X_margin_test"]).astype(np.float32)
    X_s_test = scaler_s.transform(test_data["X_shape_test"]).astype(np.float32)
    X_t_test = scaler_t.transform(test_data["X_texture_test"]).astype(np.float32)
    X_e_test = scaler_e.transform(test_data["X_eng_test"]).astype(np.float32)

    # Convert to tensors
    X_m_train_t = torch.from_numpy(X_m_train).float()
    X_s_train_t = torch.from_numpy(X_s_train).float()
    X_t_train_t = torch.from_numpy(X_t_train).float()
    X_e_train_t = torch.from_numpy(X_e_train).float()
    X_i_train_t = torch.from_numpy(fd["img_feat_train"]).float()
    y_train_t = torch.from_numpy(fd["y_train"]).long()

    X_m_val_t = torch.from_numpy(X_m_val).float()
    X_s_val_t = torch.from_numpy(X_s_val).float()
    X_t_val_t = torch.from_numpy(X_t_val).float()
    X_e_val_t = torch.from_numpy(X_e_val).float()
    X_i_val_t = torch.from_numpy(fd["img_feat_val"]).float()
    y_val_t = torch.from_numpy(fd["y_val"]).long()

    X_m_test_t = torch.from_numpy(X_m_test).float()
    X_s_test_t = torch.from_numpy(X_s_test).float()
    X_t_test_t = torch.from_numpy(X_t_test).float()
    X_e_test_t = torch.from_numpy(X_e_test).float()
    X_i_test_t = torch.from_numpy(test_data["img_feat_test"]).float()

    # Create dataloaders
    train_dataset = TensorDataset(
        X_m_train_t, X_s_train_t, X_t_train_t, X_i_train_t, X_e_train_t, y_train_t
    )
    val_dataset = TensorDataset(
        X_m_val_t, X_s_val_t, X_t_val_t, X_i_val_t, X_e_val_t, y_val_t
    )
    test_dataset = TensorDataset(
        X_m_test_t, X_s_test_t, X_t_test_t, X_i_test_t, X_e_test_t
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Initialize model
    model = LeafMultimodalClassifier(
        num_classes=n_classes,
        latent_dim=256,
        n_heads=4,
        dropout=0.4,
        img_dim=1152,
        use_engineered=True,
        eng_dim=eng_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    best_val_ll = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch_m, batch_s, batch_t, batch_i, batch_e, batch_y in train_loader:
            batch_m = batch_m.to(device)
            batch_s = batch_s.to(device)
            batch_t = batch_t.to(device)
            batch_i = batch_i.to(device)
            batch_e = batch_e.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                logits = model(batch_m, batch_s, batch_t, batch_i, batch_e)
                loss = criterion(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        train_loss_avg = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        n_val_batches = 0

        with torch.no_grad():
            for batch_m, batch_s, batch_t, batch_i, batch_e, batch_y in val_loader:
                batch_m = batch_m.to(device)
                batch_s = batch_s.to(device)
                batch_t = batch_t.to(device)
                batch_i = batch_i.to(device)
                batch_e = batch_e.to(device)
                batch_y = batch_y.to(device)

                with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    logits = model(batch_m, batch_s, batch_t, batch_i, batch_e)
                    loss = criterion(logits, batch_y)

                val_loss += loss.item()
                probs = F.softmax(logits.float(), dim=1)
                val_preds.append(probs.cpu().numpy())
                val_true.append(batch_y.cpu().numpy())
                n_val_batches += 1

        val_loss_avg = val_loss / max(n_val_batches, 1)
        val_preds_np = np.vstack(val_preds)
        val_true_np = np.concatenate(val_true)

        eps = 1e-15
        val_preds_clipped = np.clip(val_preds_np, eps, 1 - eps)
        val_preds_norm = val_preds_clipped / val_preds_clipped.sum(
            axis=1, keepdims=True
        )
        val_ll = log_loss(val_true_np, val_preds_norm, labels=np.arange(n_classes))

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Val LL: {val_ll:.4f}"
        )

        if val_ll < best_val_ll:
            best_val_ll = val_ll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # OOF predictions
    model.eval()
    oof_preds_this = []
    with torch.no_grad():
        for batch_m, batch_s, batch_t, batch_i, batch_e, _ in val_loader:
            batch_m = batch_m.to(device)
            batch_s = batch_s.to(device)
            batch_t = batch_t.to(device)
            batch_i = batch_i.to(device)
            batch_e = batch_e.to(device)
            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                logits = model(batch_m, batch_s, batch_t, batch_i, batch_e)
            probs = F.softmax(logits.float(), dim=1)
            oof_preds_this.append(probs.cpu().numpy())

    oof_preds_fold = np.vstack(oof_preds_this)
    OOF_preds[fd["val_idx"]] = oof_preds_fold
    best_val_losses.append(best_val_ll)

    # Test predictions
    model.eval()
    test_preds_fold = []
    with torch.no_grad():
        for batch_m, batch_s, batch_t, batch_i, batch_e in test_loader:
            batch_m = batch_m.to(device)
            batch_s = batch_s.to(device)
            batch_t = batch_t.to(device)
            batch_i = batch_i.to(device)
            batch_e = batch_e.to(device)
            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                logits = model(batch_m, batch_s, batch_t, batch_i, batch_e)
            probs = F.softmax(logits.float(), dim=1)
            test_preds_fold.append(probs.cpu().numpy())

    test_preds_fold_np = np.vstack(test_preds_fold)
    test_preds += test_preds_fold_np / n_folds

    print(f"Fold {fold_idx+1} best val LL: {best_val_ll:.4f}")

# ==================== OOF Validation Score ====================
eps = 1e-15
OOF_clipped = np.clip(OOF_preds, eps, 1 - eps)
OOF_norm = OOF_clipped / OOF_clipped.sum(axis=1, keepdims=True)
y_true_full = np.zeros(n_train, dtype=np.int64)
for fd in folds_data:
    y_true_full[fd["val_idx"]] = fd["y_val"]

final_ll = log_loss(y_true_full, OOF_norm, labels=np.arange(n_classes))
print(f"\n=== OOF Log Loss: {final_ll:.6f} ===")
print(f"Average fold best val LL: {np.mean(best_val_losses):.6f}")

# ==================== Generate Submission ====================
submission_cols = sample_sub.columns.tolist()
assert submission_cols[0] == "id", "First column must be id"

submission = pd.DataFrame({"id": test_data["test_ids"]})
for i, col in enumerate(submission_cols[1:]):
    submission[col] = test_preds[:, i]

submission = submission[submission_cols]

# Normalize probabilities
prob_cols = submission_cols[1:]
submission[prob_cols] = submission[prob_cols].div(
    submission[prob_cols].sum(axis=1), axis=0
)

submission.to_csv(submission_dir / "submission.csv", index=False)
print(
    f'Submission saved to {submission_dir / "submission.csv"} with shape {submission.shape}'
)

# Verify submission
assert submission.shape == (
    n_test,
    n_classes + 1,
), f"Submission shape {submission.shape} != {(n_test, n_classes+1)}"
assert (
    submission.iloc[:, 1:].sum(axis=1) - 1.0
).abs().max() < 0.01, "Probabilities don't sum to 1"

print(f"Final Validation Score: {final_ll}")
