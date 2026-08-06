import numpy as np
import pandas as pd
import os
import warnings
import gc
import json
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ============================================
# SETUP AND DATA LOADING
# ============================================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

# Extract features and labels
margin_cols = [c for c in train_df.columns if c.lower().startswith("margin")]
shape_cols = [c for c in train_df.columns if c.lower().startswith("shape")]
texture_cols = [c for c in train_df.columns if c.lower().startswith("texture")]
feature_cols = margin_cols + shape_cols + texture_cols

X_all = train_df[feature_cols].values.astype(np.float32)
y_all = train_df["species"].values
X_test = test_df[feature_cols].values.astype(np.float32)
test_ids = test_df["id"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_all)
num_classes = len(le.classes_)
class_names = le.classes_

print(f"Train: {X_all.shape}, Test: {X_test.shape}, Classes: {num_classes}")


# ============================================
# FEATURE ENGINEERING (Shared for all models)
# ============================================
def engineer_features(X):
    """Add statistical features to raw features"""
    n_samples = X.shape[0]
    additional = []

    # Per-type statistics
    for start in [0, 64, 128]:  # margin, shape, texture
        X_type = X[:, start : start + 64]
        stats = np.column_stack(
            [
                np.mean(X_type, axis=1),
                np.std(X_type, axis=1),
                np.median(X_type, axis=1),
                np.percentile(X_type, 25, axis=1),
                np.percentile(X_type, 75, axis=1),
                np.max(X_type, axis=1),
                np.min(X_type, axis=1),
                np.sum(X_type, axis=1),
                np.sqrt(np.mean(X_type**2, axis=1)),
            ]
        )
        additional.append(stats)

    # Cross-type correlations and ratios
    margin_feat = X[:, 0:64]
    shape_feat = X[:, 64:128]
    texture_feat = X[:, 128:192]

    # Correlations between feature types
    corr_ms = np.array(
        [np.corrcoef(m, s)[0, 1] for m, s in zip(margin_feat, shape_feat)]
    )
    corr_mt = np.array(
        [np.corrcoef(m, t)[0, 1] for m, t in zip(margin_feat, texture_feat)]
    )
    corr_st = np.array(
        [np.corrcoef(s, t)[0, 1] for s, t in zip(shape_feat, texture_feat)]
    )

    # Ratios of means
    mean_m = np.mean(margin_feat, axis=1)
    mean_s = np.mean(shape_feat, axis=1)
    mean_t = np.mean(texture_feat, axis=1)

    additional.append(
        np.column_stack(
            [
                corr_ms,
                corr_mt,
                corr_st,
                mean_m / (mean_s + 1e-10),
                mean_t / (mean_s + 1e-10),
                mean_m / (mean_t + 1e-10),
                mean_m + mean_s + mean_t,
                np.std(margin_feat, axis=1)
                * np.std(shape_feat, axis=1)
                * np.std(texture_feat, axis=1),
            ]
        )
    )

    return np.hstack([X] + additional).astype(np.float32)


print("Engineering features...")
X_all_eng = engineer_features(X_all)
X_test_eng = engineer_features(X_test)
print(f"Engineered feature dims: Train {X_all_eng.shape}, Test {X_test_eng.shape}")


# ============================================
# NEURAL NETWORK MODEL
# ============================================
class LeafClassifier(nn.Module):
    def __init__(
        self, input_dim, num_classes, hidden_dims=[512, 256, 128], dropout=0.4
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


# ============================================
# TRAINING SETUP
# ============================================
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
EPOCHS = 30
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Store OOF predictions
oof_lgb = np.zeros((len(X_all), num_classes))
oof_xgb = np.zeros((len(X_all), num_classes))
oof_nn = np.zeros((len(X_all), num_classes))

test_lgb = np.zeros((len(X_test), num_classes))
test_xgb = np.zeros((len(X_test), num_classes))
test_nn = np.zeros((len(X_test), num_classes))

# ============================================
# CROSS-VALIDATION TRAINING
# ============================================
print(f"\nStarting {N_FOLDS}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all_eng, y_encoded)):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/{N_FOLDS}")
    print(f"{'='*50}")

    # Split data
    X_train_fold = X_all_eng[train_idx]
    y_train_fold = y_encoded[train_idx]
    X_val_fold = X_all_eng[val_idx]
    y_val_fold = y_encoded[val_idx]

    # Normalize features
    scaler = StandardScaler()
    X_train_fold_norm = scaler.fit_transform(X_train_fold)
    X_val_fold_norm = scaler.transform(X_val_fold)
    X_test_norm = scaler.transform(X_test_eng)

    # ============ LightGBM ============
    print(f"Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=16,
        max_depth=6,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
        objective="multiclass",
        num_class=num_classes,
    )
    lgb_model.fit(
        X_train_fold_norm,
        y_train_fold,
        eval_set=[(X_val_fold_norm, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val_fold_norm)
    test_lgb += lgb_model.predict_proba(X_test_norm) / N_FOLDS

    # ============ XGBoost ============
    print(f"Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42 + fold,
        n_jobs=-1,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb_model.fit(
        X_train_fold_norm,
        y_train_fold,
        eval_set=[(X_val_fold_norm, y_val_fold)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val_fold_norm)
    test_xgb += xgb_model.predict_proba(X_test_norm) / N_FOLDS

    # ============ Neural Network ============
    print(f"Training Neural Network...")
    nn_model = LeafClassifier(
        input_dim=X_train_fold_norm.shape[1],
        num_classes=num_classes,
        hidden_dims=[256, 128, 64],
        dropout=0.5,
    ).to(DEVICE)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_fold_norm), torch.LongTensor(y_train_fold)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_fold_norm), torch.LongTensor(y_val_fold)
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

    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float("inf")
    best_model_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(EPOCHS):
        nn_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = nn_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_X)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += len(batch_X)

        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = nn_model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * len(batch_X)
                val_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_loss /= len(val_idx)
        val_acc = (val_preds.argmax(axis=1) == np.array(val_labels)).mean()

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(
                f"  Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss/len(train_idx):.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in nn_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    nn_model.load_state_dict(best_model_state)
    nn_model.eval()

    with torch.no_grad():
        val_preds = []
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = nn_model(batch_X)
            val_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
        oof_nn[val_idx] = np.vstack(val_preds)

        test_preds = []
        test_dataset = TensorDataset(torch.FloatTensor(X_test_norm))
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        for (batch_X,) in test_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = nn_model(batch_X)
            test_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
        test_nn += np.vstack(test_preds) / N_FOLDS

    fold_logloss = log_loss(y_val_fold, oof_lgb[val_idx], labels=range(num_classes))
    print(f"  Fold {fold+1} LogLoss: {fold_logloss:.4f}")

    del lgb_model, xgb_model, nn_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================
# ENSEMBLE WEIGHT OPTIMIZATION
# ============================================
print("\nOptimizing ensemble weights...")


def negative_log_loss(weights):
    weights = np.abs(weights) / np.abs(weights).sum()
    ensemble_pred = weights[0] * oof_lgb + weights[1] * oof_xgb + weights[2] * oof_nn
    return log_loss(y_encoded, ensemble_pred, labels=range(num_classes))


init_weights = np.array([0.4, 0.4, 0.2])
result = minimize(negative_log_loss, init_weights, method="Nelder-Mead")
ensemble_weights = np.abs(result.x) / np.abs(result.x).sum()
print(f"Optimal ensemble weights (LGB, XGB, NN): {ensemble_weights}")

# ============================================
# FINAL VALIDATION METRIC
# ============================================
final_oof = (
    ensemble_weights[0] * oof_lgb
    + ensemble_weights[1] * oof_xgb
    + ensemble_weights[2] * oof_nn
)
final_oof_clipped = np.clip(final_oof, 1e-15, 1 - 1e-15)
val_score = log_loss(y_encoded, final_oof_clipped, labels=range(num_classes))
print(f"\nEnsemble validation LogLoss: {val_score:.6f}")

lgb_score = log_loss(
    y_encoded, np.clip(oof_lgb, 1e-15, 1 - 1e-15), labels=range(num_classes)
)
xgb_score = log_loss(
    y_encoded, np.clip(oof_xgb, 1e-15, 1 - 1e-15), labels=range(num_classes)
)
nn_score = log_loss(
    y_encoded, np.clip(oof_nn, 1e-15, 1 - 1e-15), labels=range(num_classes)
)
print(
    f"Individual scores - LGB: {lgb_score:.6f}, XGB: {xgb_score:.6f}, NN: {nn_score:.6f}"
)

# ============================================
# TEST PREDICTIONS AND SUBMISSION
# ============================================
print("\nGenerating test predictions...")
test_ensemble = (
    ensemble_weights[0] * test_lgb
    + ensemble_weights[1] * test_xgb
    + ensemble_weights[2] * test_nn
)
test_ensemble = np.clip(test_ensemble, 1e-15, 1 - 1e-15)
test_ensemble = test_ensemble / test_ensemble.sum(axis=1, keepdims=True)

submission = pd.DataFrame(test_ensemble, columns=class_names)
submission.insert(0, "id", test_ids)
submission = submission[sample_sub.columns]

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")

# ============================================
# SAVE MODEL ARTIFACTS
# ============================================
os.makedirs("./working", exist_ok=True)
np.save("./working/oof_lgb.npy", oof_lgb)
np.save("./working/oof_xgb.npy", oof_xgb)
np.save("./working/oof_nn.npy", oof_nn)
np.save("./working/test_lgb.npy", test_lgb)
np.save("./working/test_xgb.npy", test_xgb)
np.save("./working/test_nn.npy", test_nn)
np.save("./working/ensemble_weights.npy", ensemble_weights)

with open("./working/training_meta.json", "w") as f:
    json.dump(
        {
            "num_folds": N_FOLDS,
            "epochs": EPOCHS,
            "ensemble_weights": ensemble_weights.tolist(),
            "val_logloss": float(val_score),
            "lgb_logloss": float(lgb_score),
            "xgb_logloss": float(xgb_score),
            "nn_logloss": float(nn_score),
        },
        f,
        indent=2,
    )

print(f"\nTraining complete!")
print(f"Final Validation Score: {val_score}")
