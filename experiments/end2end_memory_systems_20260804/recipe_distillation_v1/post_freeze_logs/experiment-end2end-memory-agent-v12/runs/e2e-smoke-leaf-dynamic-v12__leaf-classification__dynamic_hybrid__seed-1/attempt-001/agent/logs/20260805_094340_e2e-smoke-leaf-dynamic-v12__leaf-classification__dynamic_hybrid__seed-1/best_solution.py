import numpy as np
import pandas as pd
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from scipy.stats import skew, kurtosis
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, AutoModel

# ============================================
# 1. Load Data
# ============================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# Identify feature columns
feature_cols = [c for c in train_df.columns if c not in ["id", "species"]]
margin_cols = [c for c in feature_cols if c.startswith("margin")]
shape_cols = [c for c in feature_cols if c.startswith("shape")]
texture_cols = [c for c in feature_cols if c.startswith("texture")]


# ============================================
# 2. Feature Engineering
# ============================================
def engineer_features(df):
    """Create rich feature set from margin/shape/texture + derived features."""
    df = df.copy()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Per-group statistics
    for group_name, group_cols in [
        ("margin", margin_cols),
        ("shape", shape_cols),
        ("texture", texture_cols),
    ]:
        group_data = df[group_cols].values
        df[f"{group_name}_mean"] = group_data.mean(axis=1)
        df[f"{group_name}_std"] = group_data.std(axis=1)
        df[f"{group_name}_min"] = group_data.min(axis=1)
        df[f"{group_name}_max"] = group_data.max(axis=1)
        df[f"{group_name}_range"] = df[f"{group_name}_max"] - df[f"{group_name}_min"]
        df[f"{group_name}_skew"] = skew(group_data, axis=1)
        df[f"{group_name}_kurt"] = kurtosis(group_data, axis=1)
        df[f"{group_name}_median"] = np.median(group_data, axis=1)
        df[f"{group_name}_p25"] = np.percentile(group_data, 25, axis=1)
        df[f"{group_name}_p75"] = np.percentile(group_data, 75, axis=1)
        df[f"{group_name}_iqr"] = df[f"{group_name}_p75"] - df[f"{group_name}_p25"]
        eps = 1e-12
        df[f"{group_name}_energy"] = np.sum(group_data**2, axis=1)
        norm_data = group_data / np.maximum(
            np.sum(group_data, axis=1, keepdims=True), eps
        )
        df[f"{group_name}_entropy"] = -np.sum(
            norm_data * np.log(norm_data + eps), axis=1
        )
        fft_data = np.fft.rfft(group_data, axis=1)
        fft_mag = np.abs(fft_data)[:, 1:]
        df[f"{group_name}_fft_dominant"] = np.argmax(fft_mag, axis=1) / max(
            fft_mag.shape[1], 1
        )
        df[f"{group_name}_fft_energy"] = np.sum(fft_mag**2, axis=1)

    # Cross-group correlations
    margin_data = df[margin_cols].values
    shape_data = df[shape_cols].values
    texture_data = df[texture_cols].values
    df["corr_margin_shape"] = np.array(
        [np.corrcoef(m, s)[0, 1] for m, s in zip(margin_data, shape_data)]
    )
    df["corr_margin_texture"] = np.array(
        [np.corrcoef(m, t)[0, 1] for m, t in zip(margin_data, texture_data)]
    )
    df["corr_shape_texture"] = np.array(
        [np.corrcoef(s, t)[0, 1] for s, t in zip(shape_data, texture_data)]
    )

    # Ratios
    eps = 1e-12
    df["ratio_margin_shape"] = np.sum(margin_data, axis=1) / (
        np.sum(shape_data, axis=1) + eps
    )
    df["ratio_margin_texture"] = np.sum(margin_data, axis=1) / (
        np.sum(texture_data, axis=1) + eps
    )
    df["ratio_shape_texture"] = np.sum(shape_data, axis=1) / (
        np.sum(texture_data, axis=1) + eps
    )

    return df


train_eng = engineer_features(train_df)
test_eng = engineer_features(test_df)

# ============================================
# 3. Prepare Features and Labels
# ============================================
engineered_feature_cols = [c for c in train_eng.columns if c not in ["id", "species"]]
X_full = train_eng[engineered_feature_cols].copy()
X_test = test_eng[engineered_feature_cols].copy()
col_medians = X_full.median()
X_full = X_full.fillna(col_medians)
X_test = X_test.fillna(col_medians)

label_encoder = LabelEncoder()
y_full = label_encoder.fit_transform(train_eng["species"])
n_classes = len(label_encoder.classes_)
species_cols = list(sample_sub.columns[1:])

# Stratified split: 85% train, 15% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.15, random_state=42, stratify=y_full
)

# Save indices for image alignment
train_idx = X_train.index.values
val_idx = X_val.index.values
assert len(set(train_idx) & set(val_idx)) == 0, "Index overlap detected!"

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(
    f"Train shape: {X_train_scaled.shape}, Val shape: {X_val_scaled.shape}, Test shape: {X_test_scaled.shape}"
)

# ============================================
# 4. Extract Image Features using SigLIP2
# ============================================
print("Loading SigLIP2 model for feature extraction...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
siglip2_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
siglip2_model.to(device)
siglip2_model.eval()
processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-256")

train_ids = train_eng["id"].values
test_ids = test_eng["id"].values


def extract_image_features_batch(image_ids, base_path="./input/images", batch_size=16):
    """Extract features from all images using SigLIP2 in batches"""
    all_features = []
    for i in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[i : i + batch_size]
        batch_images = []
        for img_id in batch_ids:
            img_path = os.path.join(base_path, f"{img_id}.jpg")
            img = Image.open(img_path).convert("RGB")
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            batch_images.append(np.array(img) / 255.0)
        pixel_values = processor(
            images=batch_images, return_tensors="pt", do_resize=False
        ).pixel_values.to(device)
        with torch.no_grad():
            features = siglip2_model.get_image_features(pixel_values=pixel_values)
        all_features.append(features.cpu().numpy())
    return np.vstack(all_features)


cache_path = "./working/siglip2_features.npz"
os.makedirs("./working", exist_ok=True)
if os.path.exists(cache_path):
    cache = np.load(cache_path)
    train_img_features = cache["train_img_features"]
    test_img_features = cache["test_img_features"]
else:
    print("Extracting training image features...")
    train_img_features = extract_image_features_batch(train_ids)
    print("Extracting test image features...")
    test_img_features = extract_image_features_batch(test_ids)
    np.savez(
        cache_path,
        train_img_features=train_img_features,
        test_img_features=test_img_features,
    )

print(
    f"Image features: train={train_img_features.shape}, test={test_img_features.shape}"
)

# ============================================
# 5. Combine Features
# ============================================
train_img_split = train_img_features[train_idx]
val_img_split = train_img_features[val_idx]

X_train_combined = np.hstack([X_train_scaled, train_img_split])
X_val_combined = np.hstack([X_val_scaled, val_img_split])
X_test_combined = np.hstack([X_test_scaled, test_img_features])

print(
    f"Combined features: train={X_train_combined.shape}, val={X_val_combined.shape}, test={X_test_combined.shape}"
)

# ============================================
# 6. Train Ensemble Models
# ============================================
print("Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=n_classes,
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss",
    early_stopping_rounds=20,
)
xgb_model.fit(
    X_train_combined, y_train, eval_set=[(X_val_combined, y_val)], verbose=False
)
xgb_val_preds = xgb_model.predict_proba(X_val_combined)
xgb_val_loss = log_loss(y_val, xgb_val_preds)
print(f"  XGBoost val logloss: {xgb_val_loss:.4f}")

print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    objective="multiclass",
    num_class=n_classes,
    metric="multi_logloss",
)
lgb_model.fit(
    X_train_combined,
    y_train,
    eval_set=[(X_val_combined, y_val)],
    callbacks=[lgb.early_stopping(20, verbose=False)],
    eval_metric="multi_logloss",
)
lgb_val_preds = lgb_model.predict_proba(X_val_combined)
lgb_val_loss = log_loss(y_val, lgb_val_preds)
print(f"  LightGBM val logloss: {lgb_val_loss:.4f}")

print("Training RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features=0.6,
    n_jobs=-1,
    random_state=42,
)
rf_model.fit(X_train_combined, y_train)
rf_val_preds = rf_model.predict_proba(X_val_combined)
if rf_val_preds.shape[1] < n_classes:
    padded = np.zeros((rf_val_preds.shape[0], n_classes))
    for i, cls in enumerate(rf_model.classes_):
        padded[:, cls] = rf_val_preds[:, i]
    rf_val_preds = padded
rf_val_loss = log_loss(y_val, rf_val_preds)
print(f"  RandomForest val logloss: {rf_val_loss:.4f}")

# ============================================
# 7. Ensemble Weight Optimization
# ============================================
print("Optimizing ensemble weights...")
best_weight_score = float("inf")
best_weights = None
for w1 in [0.2, 0.3, 0.4, 0.5]:
    for w2 in [0.2, 0.3, 0.4, 0.5]:
        for w3 in [0.2, 0.3, 0.4, 0.5]:
            total = w1 + w2 + w3
            w1n, w2n, w3n = w1 / total, w2 / total, w3 / total
            ensemble_preds = (
                w1n * xgb_val_preds + w2n * lgb_val_preds + w3n * rf_val_preds
            )
            loss = log_loss(y_val, ensemble_preds)
            if loss < best_weight_score:
                best_weight_score = loss
                best_weights = (w1n, w2n, w3n)
print(f"Best ensemble weights: {best_weights}, val logloss: {best_weight_score:.4f}")

# ============================================
# 8. Generate Test Predictions and Submission
# ============================================
print("Generating test predictions...")
xgb_test_preds = xgb_model.predict_proba(X_test_combined)
lgb_test_preds = lgb_model.predict_proba(X_test_combined)
rf_test_preds = rf_model.predict_proba(X_test_combined)
if rf_test_preds.shape[1] < n_classes:
    padded = np.zeros((rf_test_preds.shape[0], n_classes))
    for i, cls in enumerate(rf_model.classes_):
        padded[:, cls] = rf_test_preds[:, i]
    rf_test_preds = padded

test_preds = (
    best_weights[0] * xgb_test_preds
    + best_weights[1] * lgb_test_preds
    + best_weights[2] * rf_test_preds
)

# Clip and renormalize
eps = 1e-15
test_preds = np.clip(test_preds, eps, 1.0 - eps)
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# Create submission with correct column order
submission = pd.DataFrame(test_preds, columns=species_cols)
submission.insert(0, "id", test_ids)
submission = submission[["id"] + species_cols]

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved with shape: {submission.shape}")

# ============================================
# 9. Final Validation Score
# ============================================
final_val_preds = (
    best_weights[0] * xgb_val_preds
    + best_weights[1] * lgb_val_preds
    + best_weights[2] * rf_val_preds
)
final_val_preds = np.clip(final_val_preds, eps, 1.0 - eps)
final_val_preds = final_val_preds / final_val_preds.sum(axis=1, keepdims=True)
score = log_loss(y_val, final_val_preds)

print(f"Final Validation Score: {score}")
