import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os
import pickle
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

feature_cols = (
    [f"margin{i}" for i in range(1, 65)]
    + [f"shape{i}" for i in range(1, 65)]
    + [f"texture{i}" for i in range(1, 65)]
)

missing_cols = [c for c in feature_cols if c not in train_df.columns]
if missing_cols:
    feature_cols_alt = (
        [f"margin_{i}" for i in range(1, 65)]
        + [f"shape_{i}" for i in range(1, 65)]
        + [f"texture_{i}" for i in range(1, 65)]
    )
    if all(c in train_df.columns for c in feature_cols_alt):
        print("Using alternative column naming with underscores")
        feature_cols = feature_cols_alt
    else:
        raise KeyError(f"Feature columns not found. Missing: {missing_cols[:5]}")

print(f"Found {len(feature_cols)} feature columns")

label_encoder = LabelEncoder()
train_df["label_encoded"] = label_encoder.fit_transform(train_df["species"])
n_classes = len(label_encoder.classes_)
species_cols = sample_sub.columns[1:].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading SigLIP2 model for image feature extraction...")
processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-256")
model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
model = model.to(device).eval()

image_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def extract_image_features(image_paths, batch_size=16):
    features_list = []
    dataloader = DataLoader(
        ImageDataset(image_paths, image_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            with torch.cuda.amp.autocast():
                pooled_feat = model.get_image_features(pixel_values=batch)
            features_list.append(pooled_feat.cpu().float().numpy())
    return np.vstack(features_list)


train_ids = train_df["id"].tolist()
test_ids = test_df["id"].tolist()

print("Extracting image features for training data...")
train_img_paths = [f"./input/images/{img_id}.jpg" for img_id in train_ids]
train_img_features = extract_image_features(train_img_paths, batch_size=16)
print(f"Train image features shape: {train_img_features.shape}")

print("Extracting image features for test data...")
test_img_paths = [f"./input/images/{img_id}.jpg" for img_id in test_ids]
test_img_features = extract_image_features(test_img_paths, batch_size=16)
print(f"Test image features shape: {test_img_features.shape}")

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Engineering tabular features...")


def create_statistical_features(df, feature_cols):
    margin_cols = [c for c in feature_cols if "margin" in c]
    shape_cols = [c for c in feature_cols if "shape" in c]
    texture_cols = [c for c in feature_cols if "texture" in c]

    for group_name, group_cols in [
        ("margin", margin_cols),
        ("shape", shape_cols),
        ("texture", texture_cols),
    ]:
        group_data = df[group_cols].values
        df[f"{group_name}_mean"] = np.mean(group_data, axis=1)
        df[f"{group_name}_std"] = np.std(group_data, axis=1)
        df[f"{group_name}_min"] = np.min(group_data, axis=1)
        df[f"{group_name}_max"] = np.max(group_data, axis=1)
        df[f"{group_name}_skew"] = stats.skew(group_data, axis=1)
        df[f"{group_name}_kurtosis"] = stats.kurtosis(group_data, axis=1)
        df[f"{group_name}_median"] = np.median(group_data, axis=1)
        df[f"{group_name}_range"] = df[f"{group_name}_max"] - df[f"{group_name}_min"]
        df[f"{group_name}_q25"] = np.percentile(group_data, 25, axis=1)
        df[f"{group_name}_q75"] = np.percentile(group_data, 75, axis=1)
        df[f"{group_name}_energy"] = np.sum(group_data**2, axis=1)
        norm_data = group_data / np.maximum(
            np.sum(group_data, axis=1, keepdims=True), 1e-10
        )
        df[f"{group_name}_entropy"] = -np.sum(
            norm_data * np.log(np.maximum(norm_data, 1e-10)), axis=1
        )
    return df


train_feat = create_statistical_features(train_df.copy(), feature_cols)
test_feat = create_statistical_features(test_df.copy(), feature_cols)

engineered_cols = [
    c for c in train_feat.columns if c not in ["id", "species", "label_encoded"]
]
tab_feat_cols = [c for c in engineered_cols if c not in feature_cols]
print(f"Statistical features: {len(tab_feat_cols)}")

X_tab_train = train_feat[engineered_cols].values.astype(np.float32)
X_tab_test = test_feat[engineered_cols].values.astype(np.float32)
y_train = train_feat["label_encoded"].values

print("Scaling features...")
scaler = StandardScaler()
X_tab_train_scaled = scaler.fit_transform(X_tab_train)
X_tab_test_scaled = scaler.transform(X_tab_test)

print("Creating PCA features...")
pca_features_train = []
pca_features_test = []
pca_components = 20

for group_name in ["margin", "shape", "texture"]:
    group_cols = [f"{group_name}{i}" for i in range(1, 65)]
    if not all(c in train_df.columns for c in group_cols):
        group_cols = [f"{group_name}_{i}" for i in range(1, 65)]
    pca = PCA(n_components=pca_components, random_state=42)
    pca_train = pca.fit_transform(train_df[group_cols].values)
    pca_test = pca.transform(test_df[group_cols].values)
    pca_features_train.append(pca_train)
    pca_features_test.append(pca_test)

X_pca_train = np.hstack(pca_features_train)
X_pca_test = np.hstack(pca_features_test)

pca_scaler = StandardScaler()
X_pca_train_scaled = pca_scaler.fit_transform(X_pca_train)
X_pca_test_scaled = pca_scaler.transform(X_pca_test)

X_train_combined = np.hstack(
    [X_tab_train_scaled, X_pca_train_scaled, train_img_features]
)
X_test_combined = np.hstack([X_tab_test_scaled, X_pca_test_scaled, test_img_features])
print(f"Combined feature dimensions: {X_train_combined.shape[1]}")

combined_scaler = StandardScaler()
X_train_final = combined_scaler.fit_transform(X_train_combined)
X_test_final = combined_scaler.transform(X_test_combined)

print("Creating validation split...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X_train_final, y_train))

assert len(set(train_idx) & set(val_idx)) == 0
print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

X_train_split = X_train_final[train_idx]
X_val_split = X_train_final[val_idx]
y_train_split = y_train[train_idx]
y_val_split = y_train[val_idx]

image_dim = 1152
tabular_dim = X_train_split.shape[1] - image_dim
print(f"Tabular dim: {tabular_dim}, Image dim: {image_dim}")

X_train_tab = X_train_split[:, :tabular_dim].astype(np.float32)
X_train_img = X_train_split[:, tabular_dim:].astype(np.float32)
X_val_tab = X_val_split[:, :tabular_dim].astype(np.float32)
X_val_img = X_val_split[:, tabular_dim:].astype(np.float32)
X_test_tab = X_test_final[:, :tabular_dim].astype(np.float32)
X_test_img = X_test_final[:, tabular_dim:].astype(np.float32)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc_squeeze = nn.Linear(channel, channel // reduction)
        self.fc_excite = nn.Linear(channel // reduction, channel)

    def forward(self, x):
        b, c = x.shape
        squeeze = x.mean(dim=1, keepdim=True).expand(b, c)
        excite = F.relu(self.fc_squeeze(squeeze))
        excite = torch.sigmoid(self.fc_excite(excite))
        return x * excite


class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.se1 = SELayer(hidden_dim, reduction=8)
        self.se2 = SELayer(hidden_dim, reduction=8)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.out_proj = nn.Linear(hidden_dim, 256)
        self.out_bn = nn.BatchNorm1d(256)
        self.out_dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.bn_in(F.gelu(self.fc_in(x)))
        residual = x
        x = self.bn1(F.gelu(self.fc1(x)))
        x = self.dropout1(x)
        x = self.se1(x)
        x = x + residual
        residual = x
        x = self.bn2(F.gelu(self.fc2(x)))
        x = self.dropout2(x)
        x = self.se2(x)
        x = x + residual
        x = self.out_bn(F.gelu(self.out_proj(x)))
        x = self.out_dropout(x)
        return x


class LeafClassifier(nn.Module):
    def __init__(
        self, tabular_dim, image_dim, n_classes, hidden_dim=256, dropout_rate=0.3
    ):
        super().__init__()
        self.tabular_encoder = TabularEncoder(tabular_dim, hidden_dim)
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        fusion_input_dim = hidden_dim + hidden_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, tabular_features, image_features):
        tabular_emb = self.tabular_encoder(tabular_features)
        image_emb = self.image_proj(image_features)
        fused = torch.cat([tabular_emb, image_emb], dim=1)
        logits = self.fusion(fused)
        return logits


class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(
                log_probs, self.smoothing / (self.n_classes - 1)
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        return loss


print("Initializing model...")
model = LeafClassifier(
    tabular_dim=tabular_dim,
    image_dim=image_dim,
    n_classes=n_classes,
    hidden_dim=256,
    dropout_rate=0.3,
).to(device)

criterion = LabelSmoothingLoss(n_classes=n_classes, smoothing=0.1)
optimizer = AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8
)

n_epochs = 50
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

batch_size = 32
train_dataset = TensorDataset(
    torch.from_numpy(X_train_tab),
    torch.from_numpy(X_train_img),
    torch.from_numpy(y_train_split).long(),
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val_tab),
    torch.from_numpy(X_val_img),
    torch.from_numpy(y_val_split).long(),
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)


def mixup_data(x_tab, x_img, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        batch_size_ = x_tab.size(0)
        index = torch.randperm(batch_size_).to(x_tab.device)
        mixed_x_tab = lam * x_tab + (1 - lam) * x_tab[index]
        mixed_x_img = lam * x_img + (1 - lam) * x_img[index]
        y_a, y_b = y, y[index]
        return mixed_x_tab, mixed_x_img, y_a, y_b, lam
    else:
        return x_tab, x_img, y, y, 1.0


scaler_amp = torch.amp.GradScaler("cuda")

print("Starting training...")
best_val_loss = float("inf")
best_model_state = None
patience = 15
patience_counter = 0

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    n_batches = len(train_loader)

    for batch_idx, (x_tab, x_img, y) in enumerate(train_loader):
        x_tab = x_tab.to(device)
        x_img = x_img.to(device)
        y = y.to(device)

        x_tab, x_img, y_a, y_b, lam = mixup_data(x_tab, x_img, y, alpha=0.2)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(x_tab, x_img)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        train_loss += loss.item()

    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for x_tab, x_img, y in val_loader:
            x_tab = x_tab.to(device)
            x_img = x_img.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(x_tab, x_img)
                probs = F.softmax(logits, dim=-1)
            val_preds.append(probs.cpu().numpy())
            val_targets.append(y.numpy())

    val_preds = np.vstack(val_preds)
    val_targets = np.concatenate(val_targets)
    val_score = log_loss(val_targets, val_preds, labels=range(n_classes))
    avg_train_loss = train_loss / n_batches

    scheduler.step()

    print(
        f"Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    if val_score < best_val_loss:
        best_val_loss = val_score
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"Best validation log loss: {best_val_loss:.4f}")

print("Loading best model...")
model.load_state_dict(best_model_state)
model = model.to(device)
model.eval()

val_preds = []
with torch.no_grad():
    for x_tab, x_img, _ in val_loader:
        x_tab = x_tab.to(device)
        x_img = x_img.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(x_tab, x_img)
            probs = F.softmax(logits, dim=-1)
        val_preds.append(probs.cpu().numpy())

val_preds = np.vstack(val_preds)
final_val_score = log_loss(y_val_split, val_preds, labels=range(n_classes))
print(f"Final validation log loss: {final_val_score:.4f}")

print("Generating test predictions...")
test_dataset = TensorDataset(torch.from_numpy(X_test_tab), torch.from_numpy(X_test_img))
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

test_preds = []
with torch.no_grad():
    for x_tab, x_img in test_loader:
        x_tab = x_tab.to(device)
        x_img = x_img.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(x_tab, x_img)
            probs = F.softmax(logits, dim=-1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.vstack(test_preds)
print(f"Test predictions shape: {test_preds.shape}")

test_ids = test_df["id"].values
submission = pd.DataFrame(test_preds, columns=species_cols)
submission.insert(0, "id", test_ids)

submission_prob_cols = species_cols
submission[submission_prob_cols] = submission[submission_prob_cols].clip(
    1e-15, 1 - 1e-15
)
row_sums = submission[submission_prob_cols].sum(axis=1)
for col in submission_prob_cols:
    submission[col] = submission[col] / row_sums

submission = submission[["id"] + list(sample_sub.columns[1:])]

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")

print(f"Final Validation Score: {final_val_score}")
