import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import timm

# ============================================================
# Configuration
# ============================================================
BASE_DIR = "./input"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 0
RANDOM_STATE = 42

NUM_CLASSES = 1
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 30
PATIENCE = 5
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_classifier.pth")

# ============================================================
# 1. Load Data and Labels
# ============================================================
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(SAMPLE_SUB)
test_image_ids = test_df["id"].tolist()

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Class distribution:\n{train_df['has_cactus'].value_counts()}")

# ============================================================
# 2. Create Train/Validation Split (Stratified)
# ============================================================
X = train_df["id"].values
y = train_df["has_cactus"].values

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
train_idx, val_idx = next(sss.split(X, y))
assert len(set(train_idx) & set(val_idx)) == 0, "Train/Val indices overlap!"

train_ids = X[train_idx]
train_labels = y[train_idx]
val_ids = X[val_idx]
val_labels = y[val_idx]

print(f"Train samples: {len(train_ids)}, Val samples: {len(val_ids)}")

# ============================================================
# 3. Dataset and Transform Definitions (End-to-End)
# ============================================================
# Training transforms with augmentation
train_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# Validation/Test transforms (no augmentation)
eval_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

class CactusDataset(Dataset):
    def __init__(self, image_dir, image_ids, labels=None, transform=None, is_train=False):
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform
        self.is_train = is_train
        # Use separate transforms for train vs eval if none explicitly provided
        if self.transform is None:
            self.transform = train_transform if is_train else eval_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_ids[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label, self.image_ids[idx]
        else:
            return image, self.image_ids[idx]

# ============================================================
# 4. Define End-to-End Model: EfficientNet-B0 with Custom Head
# ============================================================
print("Loading EfficientNet-B0 backbone...")
class EfficientNetClassifier(nn.Module):
    def __init__(self, dropout_rate=DROPOUT_RATE):
        super().__init__()
        # Load pretrained EfficientNet-B0 from timm
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        # Get the number of features from the backbone
        num_features = self.backbone.num_features  # Should be 1280 for EfficientNet-B0
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, NUM_CLASSES),
        )

    def forward(self, x):
        # Backbone returns features (B, num_features, H, W) for efficientnet
        features = self.backbone.forward_features(x)
        out = self.classifier(features)
        return out

model = EfficientNetClassifier(dropout_rate=DROPOUT_RATE)
model.to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================
# 5. Create Dataloaders (End-to-End)
# ============================================================
print("Creating datasets and dataloaders...")
train_dataset = CactusDataset(
    TRAIN_DIR, train_ids, train_labels, is_train=True
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

val_dataset = CactusDataset(
    TRAIN_DIR, val_ids, val_labels, transform=eval_transform
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_dataset = CactusDataset(
    TEST_DIR, test_image_ids, transform=eval_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

# ============================================================
# 6. Loss Function (Focal Loss for class imbalance)
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
            focal_weight = focal_weight * alpha_weight
        return (focal_weight * bce_loss).mean()

criterion = FocalLoss(gamma=2.0, alpha=0.75)
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler('cuda', enabled=DEVICE == "cuda")

# ============================================================
# 7. Training Loop (End-to-End)
# ============================================================
best_val_auc = 0.0
patience_counter = 0
print(f"Starting training on {DEVICE}...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    train_batches = 0

    for images, labels, _ in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=DEVICE == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_batches += 1

    avg_train_loss = train_loss / train_batches

    model.eval()
    val_loss = 0.0
    val_batches = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

            with torch.amp.autocast('cuda', enabled=DEVICE == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)

            val_loss += loss.item()
            val_batches += 1
            all_val_preds.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / val_batches
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_labels_np = np.concatenate(all_val_labels, axis=0)
    val_auc = roc_auc_score(val_labels_np, val_preds)

    scheduler.step()

    print(
        f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.6f}"
    )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  -> New best model saved (AUC: {val_auc:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  -> Early stopping triggered after {epoch} epochs")
            break

print(f"\nTraining complete. Best validation AUC: {best_val_auc:.6f}")

# ============================================================
# 8. Load Best Model and Final Validation
# ============================================================
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()

all_val_preds = []
all_val_labels = []
with torch.no_grad():
    for images, labels, _ in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        with torch.amp.autocast('cuda', enabled=DEVICE == "cuda"):
            logits = model(images)
            probs = torch.sigmoid(logits)
        all_val_preds.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

final_val_preds = np.concatenate(all_val_preds, axis=0)
final_val_labels = np.concatenate(all_val_labels, axis=0)
final_val_auc = roc_auc_score(final_val_labels, final_val_preds)
print(f"Final Validation AUC: {final_val_auc:.6f}")

# ============================================================
# 9. Test Inference
# ============================================================
print("Performing test inference...")
all_test_preds = []
all_test_ids = []

model.eval()
with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=DEVICE == "cuda"):
            logits = model(images)
            probs = torch.sigmoid(logits)
        all_test_preds.append(probs.cpu().numpy())
        all_test_ids.extend(ids)

test_predictions = np.concatenate(all_test_preds, axis=0).flatten()
test_ids_loaded = np.array(all_test_ids)

# ============================================================
# 10. Create Submission File
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame({"id": test_ids_loaded, "has_cactus": test_predictions})
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Test predictions shape: {test_predictions.shape}")

print(f"Final Validation Score: {final_val_auc:.6f}")
