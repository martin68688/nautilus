import os
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from transformers import AutoProcessor, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import time

# Setup directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)
os.makedirs("./input", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================
# 1. DATA LOADING
# =====================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# =====================================================
# 2. COLUMN DETECTION AND SORTING
# =====================================================
all_cols = train_df.columns.tolist()
margin_cols = [c for c in all_cols if "margin" in c.lower()]
shape_cols = [c for c in all_cols if "shape" in c.lower()]
texture_cols = [c for c in all_cols if "texture" in c.lower()]

def sort_feature_cols(cols):
    def get_num(c):
        match = re.search(r"(\d+)$", c)
        return int(match.group(1)) if match else float("inf")

    return sorted(cols, key=get_num)

margin_cols = sort_feature_cols(margin_cols)
shape_cols = sort_feature_cols(shape_cols)
texture_cols = sort_feature_cols(texture_cols)

assert len(margin_cols) == 64, f"Expected 64 margin features, got {len(margin_cols)}"
assert len(shape_cols) == 64, f"Expected 64 shape features, got {len(shape_cols)}"
assert len(texture_cols) == 64, f"Expected 64 texture features, got {len(texture_cols)}"

print(
    f"Found {len(margin_cols)} margin, {len(shape_cols)} shape, {len(texture_cols)} texture features"
)

# =====================================================
# 3. FEATURE GROUP EXTRACTION
# =====================================================
def extract_feature_groups(df, margin_cols, shape_cols, texture_cols):
    return {
        "margin_raw": df[margin_cols].values.astype(np.float32),
        "shape_raw": df[shape_cols].values.astype(np.float32),
        "texture_raw": df[texture_cols].values.astype(np.float32),
    }

train_features = extract_feature_groups(train_df, margin_cols, shape_cols, texture_cols)
test_features = extract_feature_groups(test_df, margin_cols, shape_cols, texture_cols)

# =====================================================
# 4. LABEL ENCODING
# =====================================================
class_order = sample_sub.columns[1:].tolist()
species_to_idx = {sp: i for i, sp in enumerate(class_order)}
train_species = train_df["species"].unique()
missing_species = set(train_species) - set(class_order)
assert len(missing_species) == 0, f"Missing species: {missing_species}"
train_labels = np.array(
    [species_to_idx[s] for s in train_df["species"]], dtype=np.int64
)
print(f"Number of classes: {len(class_order)}")

# =====================================================
# 5. IMAGE PATHS
# =====================================================
train_image_paths = [f"./input/images/{i}.jpg" for i in train_df["id"].values]
test_image_paths = [f"./input/images/{i}.jpg" for i in test_df["id"].values]
test_ids = test_df["id"].values

# =====================================================
# 6. STRATIFIED FOLDS
# =====================================================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_indices = []
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_labels)):
    fold_indices.append({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()})
    print(f"Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")

# =====================================================
# 7. SIGLIP2 EMBEDDING EXTRACTION (CACHED)
# =====================================================
print("Loading SigLIP2 processor and model...")
processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-256")

def extract_siglip2_embeddings(image_paths, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Extracting SigLIP2 embeddings for {len(image_paths)} images...")
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    all_embeddings = []
    batch_size = 16
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
            except Exception:
                batch_images.append(Image.new("RGB", (256, 256), (0, 0, 0)))

        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad(), autocast():
            outputs = model.get_image_features(**inputs)
            all_embeddings.append(outputs.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    np.save(cache_path, embeddings)
    print(f"Extracted embeddings shape: {embeddings.shape}")

    del model
    torch.cuda.empty_cache()
    return embeddings

train_siglip2 = extract_siglip2_embeddings(
    train_image_paths, "./working/train_siglip2_embeddings.npy"
)
test_siglip2 = extract_siglip2_embeddings(
    test_image_paths, "./working/test_siglip2_embeddings.npy"
)

# =====================================================
# 8. DATASET CLASS
# =====================================================
class LeafDataset(Dataset):
    def __init__(self, features_dict, siglip2_embeddings, labels=None):
        self.margin = features_dict["margin_raw"]
        self.shape = features_dict["shape_raw"]
        self.texture = features_dict["texture_raw"]
        self.siglip2 = siglip2_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.margin)

    def __getitem__(self, idx):
        margin = torch.tensor(self.margin[idx], dtype=torch.float32)
        shape = torch.tensor(self.shape[idx], dtype=torch.float32)
        texture = torch.tensor(self.texture[idx], dtype=torch.float32)
        siglip2 = torch.tensor(self.siglip2[idx], dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return margin, shape, texture, siglip2, label
        return margin, shape, texture, siglip2

# =====================================================
# 9. MODEL ARCHITECTURE
# =====================================================
class TabularEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=192, output_dim=192, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.norm(self.net(x))

class LeafFusionModel(nn.Module):
    def __init__(
        self,
        num_classes=99,
        tabular_dim=64,
        image_dim=1152,
        token_dim=192,
        num_heads=3,
        num_attention_layers=1,
        dropout=0.3,
        label_smoothing=0.1,
    ):
        super().__init__()
        self.margin_encoder = TabularEncoder(tabular_dim, 192, token_dim, dropout)
        self.shape_encoder = TabularEncoder(tabular_dim, 192, token_dim, dropout)
        self.texture_encoder = TabularEncoder(tabular_dim, 192, token_dim, dropout)

        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, token_dim),
            nn.LayerNorm(token_dim),
        )

        attention_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            attention_layer, num_layers=num_attention_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Dropout(dropout),
            nn.Linear(token_dim, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, margin, shape, texture, siglip2_emb):
        margin_token = self.margin_encoder(margin)
        shape_token = self.shape_encoder(shape)
        texture_token = self.texture_encoder(texture)
        image_token = self.image_projection(siglip2_emb)

        tokens = torch.stack(
            [margin_token, shape_token, texture_token, image_token], dim=1
        )
        fused_tokens = self.fusion_transformer(tokens)
        fused = fused_tokens.mean(dim=1)
        return self.classifier(fused)

    def compute_loss(self, logits, targets):
        return self.criterion(logits, targets)

# =====================================================
# 10. TRAINING UTILITIES
# =====================================================
def train_epoch(model, dataloader, optimizer, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for margin, shape, texture, siglip2, labels in dataloader:
        margin, shape, texture = margin.to(device), shape.to(device), texture.to(device)
        siglip2, labels = siglip2.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(margin, shape, texture, siglip2)
            loss = model.compute_loss(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total

def validate(model, dataloader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for margin, shape, texture, siglip2, labels in dataloader:
            margin, shape, texture = (
                margin.to(device),
                shape.to(device),
                texture.to(device),
            )
            siglip2, labels = siglip2.to(device), labels.to(device)
            with autocast():
                logits = model(margin, shape, texture, siglip2)
                probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_probs), np.concatenate(all_labels)

def predict_test(model, dataloader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for margin, shape, texture, siglip2 in dataloader:
            margin, shape, texture = (
                margin.to(device),
                shape.to(device),
                texture.to(device),
            )
            siglip2 = siglip2.to(device)
            with autocast():
                logits = model(margin, shape, texture, siglip2)
                probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)

# =====================================================
# 11. K-FOLD TRAINING
# =====================================================
num_classes = len(class_order)
batch_size = 32
epochs = 25
learning_rate = 3e-4
weight_decay = 1e-4
patience = 5

oof_probs = np.zeros((len(train_labels), num_classes), dtype=np.float32)
test_probs = np.zeros((len(test_ids), num_classes), dtype=np.float32)

print(f"\n=== Starting {n_splits}-Fold Training ===")
for fold in range(n_splits):
    print(f"\n--- Fold {fold} ---")
    train_idx = np.array(fold_indices[fold]["train_idx"])
    val_idx = np.array(fold_indices[fold]["val_idx"])
    assert len(set(train_idx) & set(val_idx)) == 0

    # Fold-specific scaling (fit on train only)
    scaler_margin = StandardScaler().fit(train_features["margin_raw"][train_idx])
    scaler_shape = StandardScaler().fit(train_features["shape_raw"][train_idx])
    scaler_texture = StandardScaler().fit(train_features["texture_raw"][train_idx])

    train_data = {
        "margin_raw": scaler_margin.transform(train_features["margin_raw"][train_idx]),
        "shape_raw": scaler_shape.transform(train_features["shape_raw"][train_idx]),
        "texture_raw": scaler_texture.transform(
            train_features["texture_raw"][train_idx]
        ),
    }
    val_data = {
        "margin_raw": scaler_margin.transform(train_features["margin_raw"][val_idx]),
        "shape_raw": scaler_shape.transform(train_features["shape_raw"][val_idx]),
        "texture_raw": scaler_texture.transform(train_features["texture_raw"][val_idx]),
    }

    train_dataset = LeafDataset(
        train_data, train_siglip2[train_idx], train_labels[train_idx]
    )
    val_dataset = LeafDataset(val_data, train_siglip2[val_idx], train_labels[val_idx])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = LeafFusionModel(num_classes=num_classes).to(device)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=epochs * len(train_loader),
        pct_start=0.3, div_factor=10, final_div_factor=100
    )
    scaler = GradScaler()

    best_val_loss = float("inf")
    best_val_probs = None
    no_improve = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler)
        val_probs, val_labels = validate(model, val_loader)

        eps = 1e-15
        val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
        val_loss = -np.mean(
            np.sum(np.eye(num_classes)[val_labels] * np.log(val_probs_clipped), axis=1)
        )
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_probs = val_probs.copy()
            no_improve = 0
            torch.save(model.state_dict(), f"./working/model_fold_{fold}.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (best: {best_val_loss:.4f})")
                break

    oof_probs[val_idx] = best_val_probs

    # Test predictions
    test_data = {
        "margin_raw": scaler_margin.transform(test_features["margin_raw"]),
        "shape_raw": scaler_shape.transform(test_features["shape_raw"]),
        "texture_raw": scaler_texture.transform(test_features["texture_raw"]),
    }
    test_dataset = LeafDataset(test_data, test_siglip2)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model.load_state_dict(
        torch.load(f"./working/model_fold_{fold}.pt", map_location=device)
    )
    test_probs += predict_test(model, test_loader) / n_splits

    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    print(f"Fold {fold} complete - Best Val Loss: {best_val_loss:.4f}")

# =====================================================
# 12. FINAL VALIDATION SCORE
# =====================================================
eps = 1e-15
oof_probs_clipped = np.clip(oof_probs, eps, 1 - eps)
val_score = -np.mean(
    np.sum(np.eye(num_classes)[train_labels] * np.log(oof_probs_clipped), axis=1)
)
print(f"\n=== Final OOF Log Loss: {val_score:.6f} ===")

# =====================================================
# 13. SUBMISSION GENERATION
# =====================================================
print("Generating submission file...")
test_probs_clipped = np.clip(test_probs, eps, 1 - eps)
test_probs_normalized = test_probs_clipped / test_probs_clipped.sum(
    axis=1, keepdims=True
)

submission_df = pd.DataFrame(test_probs_normalized, columns=class_order)
submission_df.insert(0, "id", test_ids)
submission_df.to_csv("./submission/submission.csv", index=False)

# Verify submission
assert list(submission_df.columns) == list(sample_sub.columns), "Column mismatch!"
assert len(submission_df) == len(
    sample_sub
), f"Row count mismatch: {len(submission_df)} vs {len(sample_sub)}"
print(f"Submission saved: {submission_df.shape}")

print(f"Final Validation Score: {val_score}")
