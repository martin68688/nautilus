import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
INPUT_DIR = "./input"
OUTPUT_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

BATCH_SIZE_FEAT = 32
BATCH_SIZE_TRAIN = 256
NUM_WORKERS = 4
SEED = 42
NUM_EPOCHS = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Step 1: Data Processing & Feature Engineering
# ============================================================
print("=" * 60)
print("Step 1: Data Processing & Feature Engineering")
print("=" * 60)

# Load CSVs
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

# Verify image paths
train_ids = train_df["id"].values
test_ids = test_df["id"].values
train_paths = [os.path.join(INPUT_DIR, "train", id) for id in train_ids]
test_paths = [os.path.join(INPUT_DIR, "test", id) for id in test_ids]

missing_train = [p for p in train_paths if not os.path.exists(p)]
if missing_train:
    raise ValueError(f"Missing training images: {len(missing_train)}")
missing_test = [p for p in test_paths if not os.path.exists(p)]
if missing_test:
    raise ValueError(f"Missing test images: {len(missing_test)}")

print(f"Verified: {len(train_paths)} training, {len(test_paths)} test images")

# Load EfficientNet-B0 backbone (safe, no hub dependency)
import timm
model_backbone = timm.create_model("efficientnet_b0", pretrained=True)
model_backbone.eval()
model_backbone.to(DEVICE)

# Preprocessing for DINOv3
normalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
preprocess = transforms.Compose(
    [
        transforms.Resize(
            (256, 256), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        normalize,
    ]
)

# Dataset class for feature extraction
class CactusFeatureDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def extract_features(model, dataloader, device="cuda"):
    model.eval()
    all_features = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            with torch.cuda.amp.autocast():
                features = model.forward_features(images)  # (B, 1280) global pool
                features = features.float()
            all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)

# Extract features
print("Extracting features for training images...")
train_dataset = CactusFeatureDataset(train_paths, transform=preprocess)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_FEAT,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
train_features_all = extract_features(model_backbone, train_loader, DEVICE)

print("Extracting features for test images...")
test_dataset = CactusFeatureDataset(test_paths, transform=preprocess)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE_FEAT,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_features_all = extract_features(model_backbone, test_loader, DEVICE)

# Split using StratifiedKFold (safe indexing)
labels = train_df["has_cactus"].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, val_idx = next(skf.split(np.zeros(len(train_df)), labels))
assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap detected!"

print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
print(f"Step 1 complete: features extracted successfully")

# ============================================================
# Step 2 & 3: Model Design & Training/Evaluation
# ============================================================
print("\n" + "=" * 60)
print("Step 2 & 3: Model Design & Training/Evaluation")
print("=" * 60)

# Build feature matrices (EfficientNet-B0 global pooled features)
train_features = train_features_all[train_idx]
val_features = train_features_all[val_idx]
test_features = test_features_all

train_labels = labels[train_idx]
val_labels = labels[val_idx]
test_ids_final = test_ids

print(f"Train features: {train_features.shape}")
print(f"Val features: {val_features.shape}")
print(f"Test features: {test_features.shape}")

# Dataset class for training
class FeatureDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

train_dataset = FeatureDataset(train_features, train_labels)
val_dataset = FeatureDataset(val_features, val_labels)
test_dataset = FeatureDataset(test_features)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# Model definition
class CactusHead(nn.Module):
    def __init__(self, input_dim=2048, dropout_rate=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)

model = CactusHead(input_dim=train_features.shape[1], dropout_rate=0.3).to(DEVICE)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} trainable parameters")
print(f"Backbone: EfficientNet-B0 (pretrained), Feature dim: {train_features.shape[1]}")

# Loss, Optimizer, Scheduler
criterion = nn.BCEWithLogitsLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)
scaler = torch.cuda.amp.GradScaler()

# Training loop
best_val_auc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")

print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    train_losses = []

    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(loss.item())

    # Validation
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(batch_features)
                probs = torch.sigmoid(logits)
            val_preds.extend(probs.cpu().numpy())

    val_auc = roc_auc_score(val_labels, np.array(val_preds))
    avg_loss = np.mean(train_losses)
    scheduler.step()

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS}: train_loss={avg_loss:.4f}, val_auc={val_auc:.4f}"
    )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(
            {"model_state_dict": model.state_dict(), "val_auc": val_auc},
            best_model_path,
        )
        print(f"  ✓ Best model saved (AUC: {val_auc:.4f})")

print(f"\nTraining complete. Best validation AUC: {best_val_auc:.4f}")

# ============================================================
# Inference & Submission
# ============================================================
print("\nLoading best model for inference...")
checkpoint = torch.load(best_model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate test predictions
test_preds = []
with torch.no_grad():
    for batch_features in test_loader:
        batch_features = batch_features.to(DEVICE)
        with torch.cuda.amp.autocast():
            logits = model(batch_features)
            probs = torch.sigmoid(logits)
        test_preds.extend(probs.cpu().numpy())

test_preds = np.array(test_preds)
print(f"Test predictions generated: {len(test_preds)} samples")

# Save submission
submission = pd.DataFrame({"id": test_ids_final, "has_cactus": test_preds})
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)
print(f"Submission saved to {os.path.join(SUBMISSION_DIR, 'submission.csv')}")
print(f"Submission shape: {submission.shape}")

# Final validation score with consistent variant
submission_variant = "efficientnet_b0_frozen_head"
print(
    f"Final Submission-Aligned Validation Score: {best_val_auc} | variant={submission_variant}"
)
