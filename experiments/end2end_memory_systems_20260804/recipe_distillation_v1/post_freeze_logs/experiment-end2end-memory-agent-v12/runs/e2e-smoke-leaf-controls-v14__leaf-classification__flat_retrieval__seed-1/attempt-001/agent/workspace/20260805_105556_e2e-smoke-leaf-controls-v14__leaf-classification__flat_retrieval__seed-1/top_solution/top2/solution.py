import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os
import pickle
import time

# =============================================================================
# 1. LOAD RAW DATA
# =============================================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# Detect feature columns dynamically
raw_feature_cols = [c for c in train_df.columns if c not in ["id", "species"]]
print(f"Raw feature columns: {len(raw_feature_cols)}")


# Identify feature groups (margin, shape, texture)
def get_group_name(col):
    return "".join([c for c in col if not c.isdigit()])


group_cols = {}
for col in raw_feature_cols:
    g = get_group_name(col)
    if g not in group_cols:
        group_cols[g] = []
    group_cols[g].append(col)

for g in group_cols:
    group_cols[g] = sorted(
        group_cols[g], key=lambda x: int("".join(filter(str.isdigit, x)))
    )
    print(f"Group {g}: {len(group_cols[g])} features")

margin_cols = group_cols["margin"]
shape_cols = group_cols["shape"]
texture_cols = group_cols["texture"]


# =============================================================================
# 2. FEATURE ENGINEERING - extract per-group features
# =============================================================================
def extract_group_features(df, cols):
    return df[cols].values.astype(np.float32)


X_margin_all = extract_group_features(train_df, margin_cols)
X_shape_all = extract_group_features(train_df, shape_cols)
X_texture_all = extract_group_features(train_df, texture_cols)

X_test_margin = extract_group_features(test_df, margin_cols)
X_test_shape = extract_group_features(test_df, shape_cols)
X_test_texture = extract_group_features(test_df, texture_cols)

target = train_df["species"].values
print(
    f"Train features: margin {X_margin_all.shape}, shape {X_shape_all.shape}, texture {X_texture_all.shape}"
)
print(f"Test features: margin {X_test_margin.shape}")

# =============================================================================
# 3. ENCODE LABELS AND CREATE VALIDATION SPLIT
# =============================================================================
le = LabelEncoder()
y_encoded = le.fit_transform(target)
print(f"Num classes: {len(le.classes_)}")

# Stratified split: 80/20
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(np.zeros(len(y_encoded)), y_encoded)))
assert len(set(train_idx) & set(val_idx)) == 0, "Index overlap detected!"

X_margin_train, X_margin_val = X_margin_all[train_idx], X_margin_all[val_idx]
X_shape_train, X_shape_val = X_shape_all[train_idx], X_shape_all[val_idx]
X_texture_train, X_texture_val = X_texture_all[train_idx], X_texture_all[val_idx]
y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
train_ids = train_df["id"].values[train_idx]
val_ids = train_df["id"].values[val_idx]

print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

# =============================================================================
# 4. STANDARDIZE FEATURES (FIT ONLY ON TRAIN)
# =============================================================================
scaler_margin = StandardScaler()
scaler_shape = StandardScaler()
scaler_texture = StandardScaler()

X_margin_train_s = scaler_margin.fit_transform(X_margin_train)
X_margin_val_s = scaler_margin.transform(X_margin_val)
X_test_margin_s = scaler_margin.transform(X_test_margin)

X_shape_train_s = scaler_shape.fit_transform(X_shape_train)
X_shape_val_s = scaler_shape.transform(X_shape_val)
X_test_shape_s = scaler_shape.transform(X_test_shape)

X_texture_train_s = scaler_texture.fit_transform(X_texture_train)
X_texture_val_s = scaler_texture.transform(X_texture_val)
X_test_texture_s = scaler_texture.transform(X_test_texture)

print(
    f"Scaled features - margin: {X_margin_train_s.shape}, shape: {X_shape_train_s.shape}, texture: {X_texture_train_s.shape}"
)

# =============================================================================
# 5. EXTRACT SIGLIP2 IMAGE FEATURES
# =============================================================================
print("Loading SigLIP2 model for image feature extraction...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

img_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
img_processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-256")
img_model = img_model.to(device)
img_model.eval()

img_transform_norm = lambda x: (x / 255.0 - 0.5) / 0.5

image_dir = "./input/images"


def extract_image_features(image_ids, batch_size=32):
    features = []
    for i in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[i : i + batch_size]
        batch_images = []
        for img_id in batch_ids:
            img_path = os.path.join(image_dir, f"{img_id}.jpg")
            try:
                img = Image.open(img_path).convert("RGB").resize((256, 256))
                img_array = np.array(img).astype(np.float32)
                img_array = img_transform_norm(img_array)
                batch_images.append(img_array)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                batch_images.append(np.zeros((256, 256, 3), dtype=np.float32))

        pixel_values = (
            torch.FloatTensor(np.array(batch_images)).permute(0, 3, 1, 2).to(device)
        )
        with torch.no_grad():
            with autocast():
                pooled = img_model.get_image_features(pixel_values=pixel_values)
            features.append(pooled.float().cpu().numpy())
    return np.concatenate(features, axis=0)


print("Extracting train image features...")
train_img_feats = extract_image_features(train_ids)
print(f"Train image features: {train_img_feats.shape}")

print("Extracting val image features...")
val_img_feats = extract_image_features(val_ids)
print(f"Val image features: {val_img_feats.shape}")

print("Extracting test image features...")
test_ids = test_df["id"].values
test_img_feats = extract_image_features(test_ids)
print(f"Test image features: {test_img_feats.shape}")


# =============================================================================
# 6. MODEL DEFINITION - Multi-Branch Fusion Network
# =============================================================================
class MultiBranchFusionModel(nn.Module):
    def __init__(
        self,
        tab_dim: int = 64,
        img_dim: int = 1152,
        num_classes: int = 99,
        hidden_dim_tab: int = 128,
        embed_dim_tab: int = 64,
        hidden_dim_fusion: int = 512,
        dropout: float = 0.3,
        img_dropout: float = 0.2,
    ):
        super().__init__()

        self.tab_encoder_margin = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim_tab),
            nn.BatchNorm1d(hidden_dim_tab),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_tab, embed_dim_tab),
            nn.BatchNorm1d(embed_dim_tab),
            nn.GELU(),
        )

        self.tab_encoder_shape = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim_tab),
            nn.BatchNorm1d(hidden_dim_tab),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_tab, embed_dim_tab),
            nn.BatchNorm1d(embed_dim_tab),
            nn.GELU(),
        )

        self.tab_encoder_texture = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim_tab),
            nn.BatchNorm1d(hidden_dim_tab),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_tab, embed_dim_tab),
            nn.BatchNorm1d(embed_dim_tab),
            nn.GELU(),
        )

        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim_tab * 2),
            nn.BatchNorm1d(hidden_dim_tab * 2),
            nn.GELU(),
            nn.Dropout(img_dropout),
            nn.Linear(hidden_dim_tab * 2, embed_dim_tab * 2),
            nn.BatchNorm1d(embed_dim_tab * 2),
            nn.GELU(),
        )

        fusion_in_dim = embed_dim_tab * 3 + embed_dim_tab * 2
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim_fusion),
            nn.BatchNorm1d(hidden_dim_fusion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_fusion, hidden_dim_fusion // 2),
            nn.BatchNorm1d(hidden_dim_fusion // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_fusion // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, margin, shape, texture, img_feat):
        emb_margin = self.tab_encoder_margin(margin)
        emb_shape = self.tab_encoder_shape(shape)
        emb_texture = self.tab_encoder_texture(texture)
        emb_img = self.img_proj(img_feat)
        fused = torch.cat([emb_margin, emb_shape, emb_texture, emb_img], dim=1)
        logits = self.fusion_head(fused)
        return logits


# =============================================================================
# 7. TRAINING SETUP
# =============================================================================
model = MultiBranchFusionModel(
    tab_dim=64,
    img_dim=1152,
    num_classes=len(le.classes_),
    hidden_dim_tab=128,
    embed_dim_tab=64,
    hidden_dim_fusion=512,
    dropout=0.3,
    img_dropout=0.2,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scaler = GradScaler()

margin_train_t = torch.FloatTensor(X_margin_train_s).to(device)
shape_train_t = torch.FloatTensor(X_shape_train_s).to(device)
texture_train_t = torch.FloatTensor(X_texture_train_s).to(device)
img_train_t = torch.FloatTensor(train_img_feats).to(device)
y_train_t = torch.LongTensor(y_train).to(device)

margin_val_t = torch.FloatTensor(X_margin_val_s).to(device)
shape_val_t = torch.FloatTensor(X_shape_val_s).to(device)
texture_val_t = torch.FloatTensor(X_texture_val_s).to(device)
img_val_t = torch.FloatTensor(val_img_feats).to(device)
y_val_t = torch.LongTensor(y_val).to(device)

margin_test_t = torch.FloatTensor(X_test_margin_s).to(device)
shape_test_t = torch.FloatTensor(X_test_shape_s).to(device)
texture_test_t = torch.FloatTensor(X_test_texture_s).to(device)
img_test_t = torch.FloatTensor(test_img_feats).to(device)

# =============================================================================
# 8. TRAINING LOOP
# =============================================================================
best_val_loss = float("inf")
best_model_state = None
patience = 15
patience_counter = 0
num_epochs = 100
batch_size = 64

train_size = len(y_train)
val_size = len(y_val)
test_size = len(test_ids)

print(f"Starting training: {train_size} train, {val_size} val samples")

for epoch in range(num_epochs):
    model.train()
    train_loss_sum = 0.0
    num_batches = 0

    perm = torch.randperm(train_size)

    for i in range(0, train_size, batch_size):
        batch_indices = perm[i : i + batch_size]

        margin_batch = margin_train_t[batch_indices]
        shape_batch = shape_train_t[batch_indices]
        texture_batch = texture_train_t[batch_indices]
        img_batch = img_train_t[batch_indices]
        labels_batch = y_train_t[batch_indices]

        optimizer.zero_grad()

        with autocast():
            logits = model(margin_batch, shape_batch, texture_batch, img_batch)
            loss = criterion(logits, labels_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item()
        num_batches += 1

    avg_train_loss = train_loss_sum / num_batches

    # Validation
    model.eval()
    val_probs_list = []
    with torch.no_grad():
        for i in range(0, val_size, batch_size):
            end_idx = min(i + batch_size, val_size)
            with autocast():
                logits = model(
                    margin_val_t[i:end_idx],
                    shape_val_t[i:end_idx],
                    texture_val_t[i:end_idx],
                    img_val_t[i:end_idx],
                )
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            val_probs_list.append(probs)

    val_probs = np.concatenate(val_probs_list, axis=0)
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    val_log_loss = log_loss(
        y_val, val_probs_clipped, labels=list(range(len(le.classes_)))
    )

    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Log Loss: {val_log_loss:.4f}"
    )

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    scheduler.step()

# =============================================================================
# 9. LOAD BEST MODEL AND EVALUATE
# =============================================================================
model.load_state_dict(best_model_state)
model.eval()

# Final validation score
val_probs_list = []
with torch.no_grad():
    for i in range(0, val_size, batch_size):
        end_idx = min(i + batch_size, val_size)
        with autocast():
            logits = model(
                margin_val_t[i:end_idx],
                shape_val_t[i:end_idx],
                texture_val_t[i:end_idx],
                img_val_t[i:end_idx],
            )
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        val_probs_list.append(probs)

val_probs = np.concatenate(val_probs_list, axis=0)
eps = 1e-15
val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
final_val_score = log_loss(
    y_val, val_probs_clipped, labels=list(range(len(le.classes_)))
)

# =============================================================================
# 10. TEST INFERENCE
# =============================================================================
test_probs_list = []
with torch.no_grad():
    for i in range(0, test_size, batch_size):
        end_idx = min(i + batch_size, test_size)
        with autocast():
            logits = model(
                margin_test_t[i:end_idx],
                shape_test_t[i:end_idx],
                texture_test_t[i:end_idx],
                img_test_t[i:end_idx],
            )
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        test_probs_list.append(probs)

test_probs = np.concatenate(test_probs_list, axis=0)

# =============================================================================
# 11. GENERATE SUBMISSION
# =============================================================================
os.makedirs("./submission", exist_ok=True)

submission_df = pd.DataFrame(test_probs, columns=le.classes_)
submission_df.insert(0, "id", test_ids)

# Ensure column order matches sample submission
expected_cols = sample_sub.columns.tolist()
submission_df = submission_df[expected_cols]

# Clip and normalize probabilities
probs_array = submission_df.iloc[:, 1:].values
probs_array = np.clip(probs_array, eps, 1 - eps)
probs_array = probs_array / probs_array.sum(axis=1, keepdims=True)
submission_df.iloc[:, 1:] = probs_array

submission_df.to_csv("./submission/submission.csv", index=False)
print(
    f"Submission saved to ./submission/submission.csv with shape {submission_df.shape}"
)

print(f"Final Validation Score: {final_val_score}")
