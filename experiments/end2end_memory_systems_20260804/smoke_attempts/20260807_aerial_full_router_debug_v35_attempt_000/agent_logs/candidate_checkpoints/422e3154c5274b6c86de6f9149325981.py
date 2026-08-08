import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
INPUT_DIR = "./input"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_DIR = os.path.join(INPUT_DIR, "test")
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 25
LR_BACKBONE = 5e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
PATIENCE = 5
NUM_WORKERS = 2

# ========== LOAD DATA ==========
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(SAMPLE_SUB)
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# ========== STRATIFIED SPLIT (no leakage) ==========
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf.split(train_df, train_df["has_cactus"])))

assert len(set(train_idx) & set(val_idx)) == 0, "Index leakage detected!"

train_set = train_df.iloc[train_idx].reset_index(drop=True).copy()
val_set = train_df.iloc[val_idx].reset_index(drop=True).copy()
print(f"Train split: {len(train_set)}, Val split: {len(val_set)}")


# ========== DATASET CLASS ==========
class CactusDataset(Dataset):
    def __init__(self, image_dir, id_list, labels=None, augment=False):
        self.image_dir = image_dir
        self.id_list = id_list
        self.labels = labels
        self.augment = augment

        self.base_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.augment_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
            ]
        )

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_id = self.id_list[idx]
        img_path = os.path.join(self.image_dir, img_id)
        img = Image.open(img_path).convert("RGB")

        if self.augment:
            img = self.augment_transform(img)

        img = self.base_transform(img)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label
        return img


train_ids = train_set["id"].tolist()
train_labels = train_set["has_cactus"].values.astype(np.float32)
val_ids = val_set["id"].tolist()
val_labels = val_set["has_cactus"].values.astype(np.float32)
test_ids = test_df["id"].tolist()

train_dataset = CactusDataset(TRAIN_DIR, train_ids, train_labels, augment=True)
val_dataset = CactusDataset(TRAIN_DIR, val_ids, val_labels, augment=False)
test_dataset = CactusDataset(TEST_DIR, test_ids, labels=None, augment=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# ========== MODEL DEFINITION ==========
import timm


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 8)
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_att = self.sigmoid(
            self.bn2(self.conv2(F.relu(self.bn1(self.conv1(avg_pool)))))
            + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(max_pool)))))
        )
        x = x * channel_att
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]
        spatial_cat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_cat))
        x = x * spatial_att
        return x


class CactusClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, dropout_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, features_only=False, num_classes=0
        )
        backbone_out = self.backbone.num_features

        self.attention = SpatialAttention(backbone_out, reduction=16)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.attention(features)
        logits = self.head(features)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss


# ========== BUILD MODEL ==========
model = CactusClassifier(num_classes=1, pretrained=True, dropout_rate=0.3)
model = model.to(DEVICE)
criterion = FocalLoss(alpha=0.75, gamma=2.0)

backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "head" in name or "attention" in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ],
    weight_decay=WEIGHT_DECAY,
)

total_steps = len(train_loader) * EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=1e-6
)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ========== TRAINING LOOP ==========
best_auc = 0.0
best_epoch = 0
patience_counter = 0
save_path = os.path.join(WORKING_DIR, "best_model.pth")

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    num_batches = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        num_batches += 1

    avg_train_loss = train_loss / num_batches

    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            val_preds.extend(probs)
            val_targets.extend(labels.cpu().numpy())

    val_auc = roc_auc_score(val_targets, val_preds)
    print(
        f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val AUC: {val_auc:.6f}"
    )

    if val_auc > best_auc:
        best_auc = val_auc
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch+1} (best: {best_auc:.6f} at epoch {best_epoch})"
            )
            break

print(f"Best validation AUC: {best_auc:.6f} (epoch {best_epoch})")

# ========== RELOAD BEST MODEL ==========
model.load_state_dict(torch.load(save_path, map_location=DEVICE))
model.eval()

# ========== VALIDATION INFERENCE ==========
val_preds_final = []
with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        val_preds_final.extend(probs)

final_val_auc = roc_auc_score(val_labels, val_preds_final)
print(f"Final validation AUC (best model): {final_val_auc:.6f}")

# ========== TEST INFERENCE ==========
test_preds = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        test_preds.extend(probs)

test_preds = np.array(test_preds)

# ========== SAVE SUBMISSION ==========
submission = pd.DataFrame({"id": test_ids, "has_cactus": test_preds})
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)
print(
    f"Submission saved to {SUBMISSION_DIR}/submission.csv with {len(submission)} rows"
)

# Verify submission format
expected_ids = pd.read_csv(SAMPLE_SUB)["id"].tolist()
assert len(submission) == len(
    expected_ids
), f"Submission rows mismatch: {len(submission)} vs {len(expected_ids)}"

submission_variant = "efficientnet_b0_attention_finetuned"
score = final_val_auc
print(
    f"Final Submission-Aligned Validation Score: {score} | variant={submission_variant}"
)
