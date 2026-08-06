import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import skew, kurtosis
from pathlib import Path
import cv2
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# ============================================================
# 2. FEATURE GROUP DEFINITIONS
# ============================================================
margin_cols = [f"margin{i}" for i in range(1, 65)]
shape_cols = [f"shape{i}" for i in range(1, 65)]
texture_cols = [f"texture{i}" for i in range(1, 65)]


# ============================================================
# 3. IMAGE-DERIVED FEATURES (Hu moments, shape descriptors)
# ============================================================
def extract_image_features(image_path):
    """Extract contour-based shape descriptors from binary leaf image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(10)

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros(10)

    cnt = max(contours, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = np.log(np.abs(hu_moments) + 1e-10)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0

    return np.concatenate([hu_moments, [circularity, aspect_ratio, extent]])


def compute_image_features(df, image_dir="./input/images"):
    """Extract image features for all IDs in dataframe."""
    n = len(df)
    feat_matrix = np.zeros((n, 10))
    for i, img_id in enumerate(df["id"].values):
        img_path = Path(image_dir) / f"{img_id}.jpg"
        if img_path.exists():
            feat_matrix[i] = extract_image_features(img_path)
    return feat_matrix


print("Extracting image features...")
train_img_feats = compute_image_features(train_df)
test_img_feats = compute_image_features(test_df)


# ============================================================
# 4. STATISTICAL FEATURES + CROSS-PRODUCTS
# ============================================================
def add_statistical_features(df):
    """Add mean, std, max, min, median, sum, skew, kurtosis for each feature group."""
    for prefix in ["margin", "shape", "texture"]:
        cols = [f"{prefix}{i}" for i in range(1, 65)]
        group = df[cols].values
        df[f"{prefix}_mean"] = np.mean(group, axis=1)
        df[f"{prefix}_std"] = np.std(group, axis=1)
        df[f"{prefix}_max"] = np.max(group, axis=1)
        df[f"{prefix}_min"] = np.min(group, axis=1)
        df[f"{prefix}_median"] = np.median(group, axis=1)
        df[f"{prefix}_sum"] = np.sum(group, axis=1)
        df[f"{prefix}_skew"] = skew(group, axis=1)
        df[f"{prefix}_kurt"] = kurtosis(group, axis=1)
    return df


train_df = add_statistical_features(train_df)
test_df = add_statistical_features(test_df)


def add_cross_products(df):
    """Add cross-group interaction features."""
    for g1 in ["margin", "shape", "texture"]:
        for g2 in ["margin", "shape", "texture"]:
            if g1 < g2:
                df[f"{g1}_{g2}_corr"] = (
                    df[f"{g1}_mean"]
                    * df[f"{g2}_mean"]
                    / (df[f"{g1}_std"] * df[f"{g2}_std"] + 1e-8)
                )
    return df


train_df = add_cross_products(train_df)
test_df = add_cross_products(test_df)

# Add image features to dataframes
image_feat_names = [f"img_hu_{i}" for i in range(7)] + [
    "img_circularity",
    "img_aspect_ratio",
    "img_extent",
]
for i, name in enumerate(image_feat_names):
    train_df[name] = train_img_feats[:, i]
    test_df[name] = test_img_feats[:, i]


def add_advanced_features(df):
    """Add advanced engineered features."""
    margin_data = df[[f"margin{i}" for i in range(1, 65)]].values
    diff_margin = np.diff(margin_data, axis=1)
    df["margin_diff_mean"] = np.mean(diff_margin, axis=1)
    df["margin_diff_std"] = np.std(diff_margin, axis=1)

    shape_data = df[[f"shape{i}" for i in range(1, 65)]].values
    half = 32
    left_part = shape_data[:, :half]
    right_part = shape_data[:, half:]
    df["shape_symmetry"] = np.mean(np.abs(left_part - right_part[:, ::-1]), axis=1)

    texture_data = df[[f"texture{i}" for i in range(1, 65)]].values
    diff_texture = np.diff(texture_data, axis=1)
    df["texture_roughness"] = np.mean(np.abs(diff_texture), axis=1)

    df["shape_energy"] = np.sum(shape_data**2, axis=1)
    df["texture_energy"] = np.sum(texture_data**2, axis=1)

    eps = 1e-10
    norm_shape = shape_data / (np.sum(shape_data, axis=1, keepdims=True) + eps)
    df["shape_entropy"] = -np.sum(norm_shape * np.log(norm_shape + eps), axis=1)
    norm_texture = texture_data / (np.sum(texture_data, axis=1, keepdims=True) + eps)
    df["texture_entropy"] = -np.sum(norm_texture * np.log(norm_texture + eps), axis=1)

    return df


train_df = add_advanced_features(train_df)
test_df = add_advanced_features(test_df)

# ============================================================
# 5. ENCODE TARGET LABELS
# ============================================================
species_names = sample_sub.columns[1:].tolist()
species_to_idx = {sp: i for i, sp in enumerate(species_names)}
train_df["species_idx"] = train_df["species"].map(species_to_idx)

# ============================================================
# 6. FEATURE COLUMNS SELECTION
# ============================================================
exclude_cols = ["id", "species", "species_idx"]
feature_cols = [c for c in train_df.columns if c not in exclude_cols]
print(f"Total engineered features: {len(feature_cols)}")

X_all = train_df[feature_cols].values
y_all = train_df["species_idx"].values
X_test = test_df[feature_cols].values

# ============================================================
# 7. SPLIT DATA (5-fold stratified)
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X_all, y_all))

# ============================================================
# 8. FOLD-WISE PREPROCESSING AND TRAINING
# ============================================================
n_classes = len(species_names)
LGB_WEIGHT = 0.6
LR_WEIGHT = 0.4

lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "n_estimators": 1000,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

lr_params = {
    "C": 0.5,
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "ovr",
    "random_state": 42,
    "n_jobs": -1,
}

original_cols = margin_cols + shape_cols + texture_cols
engineered_cols = [c for c in feature_cols if c not in original_cols]
orig_idx = [feature_cols.index(c) for c in original_cols]
eng_idx = [feature_cols.index(c) for c in engineered_cols]

oof_proba = np.zeros((len(train_df), n_classes))
test_proba_folds = []

print("Starting 5-fold cross-validation training...")

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    assert len(set(train_idx) & set(val_idx)) == 0, f"Leakage in fold {fold_idx}!"

    X_train_fold = X_all[train_idx]
    X_val_fold = X_all[val_idx]
    y_train_fold = y_all[train_idx]
    y_val_fold = y_all[val_idx]

    # Fit scaler + PCA on training fold only
    scaler = StandardScaler()
    pca = PCA(n_components=0.95, random_state=42)

    X_train_orig_scaled = scaler.fit_transform(X_train_fold[:, orig_idx])
    X_val_orig_scaled = scaler.transform(X_val_fold[:, orig_idx])

    X_train_orig_pca = pca.fit_transform(X_train_orig_scaled)
    X_val_orig_pca = pca.transform(X_val_orig_scaled)

    # Scale engineered features
    eng_scaler = StandardScaler()
    X_train_eng_scaled = eng_scaler.fit_transform(X_train_fold[:, eng_idx])
    X_val_eng_scaled = eng_scaler.transform(X_val_fold[:, eng_idx])

    # Combine features
    X_train_combined = np.hstack([X_train_orig_pca, X_train_eng_scaled])
    X_val_combined = np.hstack([X_val_orig_pca, X_val_eng_scaled])

    # Transform test
    X_test_orig_scaled = scaler.transform(X_test[:, orig_idx])
    X_test_orig_pca = pca.transform(X_test_orig_scaled)
    X_test_eng_scaled = eng_scaler.transform(X_test[:, eng_idx])
    X_test_combined = np.hstack([X_test_orig_pca, X_test_eng_scaled])

    # Train LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train_combined,
        y_train_fold,
        eval_set=[(X_val_combined, y_val_fold)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # Train Logistic Regression
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_train_combined, y_train_fold)

    # OOF predictions
    lgb_val_proba = lgb_model.predict_proba(X_val_combined)
    lr_val_proba = lr_model.predict_proba(X_val_combined)

    if lr_val_proba.shape[1] < n_classes:
        lr_val_full = np.zeros((len(X_val_combined), n_classes))
        lr_val_full[:, lr_model.classes_] = lr_val_proba
        val_blend = LGB_WEIGHT * lgb_val_proba + LR_WEIGHT * lr_val_full
    else:
        val_blend = LGB_WEIGHT * lgb_val_proba + LR_WEIGHT * lr_val_proba

    oof_proba[val_idx] = val_blend

    # Test predictions
    lgb_test_proba = lgb_model.predict_proba(X_test_combined)
    lr_test_proba = lr_model.predict_proba(X_test_combined)

    if lr_test_proba.shape[1] < n_classes:
        lr_test_full = np.zeros((len(X_test_combined), n_classes))
        lr_test_full[:, lr_model.classes_] = lr_test_proba
        lr_test_proba = lr_test_full

    test_blend = LGB_WEIGHT * lgb_test_proba + LR_WEIGHT * lr_test_proba
    test_proba_folds.append(test_blend)

    print(f"Fold {fold_idx+1}/5 complete (LGB best_iter: {lgb_model.best_iteration_})")

# Average test predictions across folds
test_proba = np.mean(test_proba_folds, axis=0)

# ============================================================
# 9. COMPUTE VALIDATION METRIC
# ============================================================
y_true = train_df["species_idx"].values

eps = 1e-15
oof_clipped = np.clip(oof_proba, eps, 1 - eps)
row_sums = oof_clipped.sum(axis=1, keepdims=True)
oof_normalized = oof_clipped / row_sums

val_score = log_loss(y_true, oof_normalized)

# ============================================================
# 10. GENERATE SUBMISSION
# ============================================================
test_clipped = np.clip(test_proba, eps, 1 - eps)
test_normalized = test_clipped / test_clipped.sum(axis=1, keepdims=True)

submission_df = pd.DataFrame(test_normalized, columns=species_names)
submission_df.insert(0, "id", test_df["id"].values)
submission_df["id"] = submission_df["id"].astype(int)

sample_cols = sample_sub.columns.tolist()
assert (
    list(submission_df.columns) == sample_cols
), "Column mismatch with sample submission!"

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv with {len(submission_df)} rows")

print(f"Final Validation Score: {val_score}")
