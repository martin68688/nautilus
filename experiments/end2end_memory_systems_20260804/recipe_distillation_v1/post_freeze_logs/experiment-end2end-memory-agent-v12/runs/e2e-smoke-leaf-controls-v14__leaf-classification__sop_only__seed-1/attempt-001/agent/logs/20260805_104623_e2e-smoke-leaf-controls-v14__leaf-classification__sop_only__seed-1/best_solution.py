import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from scipy import stats
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import os, warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============ 1. Data Loading ============
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Identify feature columns
margin_cols = [f"margin{i}" for i in range(1, 65)]
shape_cols = [f"shape{i}" for i in range(1, 65)]
texture_cols = [f"texture{i}" for i in range(1, 65)]
all_feature_cols = margin_cols + shape_cols + texture_cols

# Check test has same columns (no species column)
test_missing = set(all_feature_cols) - set(test_df.columns)
if test_missing:
    print(f"Missing in test: {test_missing}")


# ============ 2. Feature Engineering ============
def engineer_features(df, feature_cols, margin_cols, shape_cols, texture_cols):
    X_raw = df[feature_cols].values.astype(np.float32)
    X_margin = df[margin_cols].values.astype(np.float32)
    X_shape = df[shape_cols].values.astype(np.float32)
    X_texture = df[texture_cols].values.astype(np.float32)
    n_samples = len(df)
    engineered = {}

    # Block statistics
    for block_name, block_data in [
        ("margin", X_margin),
        ("shape", X_shape),
        ("texture", X_texture),
    ]:
        engineered[f"{block_name}_mean"] = np.mean(block_data, axis=1)
        engineered[f"{block_name}_std"] = np.std(block_data, axis=1)
        engineered[f"{block_name}_max"] = np.max(block_data, axis=1)
        engineered[f"{block_name}_min"] = np.min(block_data, axis=1)
        engineered[f"{block_name}_range"] = np.ptp(block_data, axis=1)
        engineered[f"{block_name}_median"] = np.median(block_data, axis=1)
        engineered[f"{block_name}_sum"] = np.sum(block_data, axis=1)
        engineered[f"{block_name}_kurtosis"] = stats.kurtosis(block_data, axis=1)
        engineered[f"{block_name}_skew"] = stats.skew(block_data, axis=1)
        engineered[f"{block_name}_energy"] = np.sum(block_data**2, axis=1)
        engineered[f"{block_name}_entropy"] = -np.sum(
            block_data * np.log(np.abs(block_data) + 1e-10), axis=1
        )
        engineered[f"{block_name}_zero_crossings"] = np.sum(
            np.diff(np.signbit(block_data), axis=1), axis=1
        )

    # Inter-block correlations
    engineered["margin_shape_corr"] = np.array(
        [np.corrcoef(X_margin[i], X_shape[i])[0, 1] for i in range(n_samples)]
    )
    engineered["margin_texture_corr"] = np.array(
        [np.corrcoef(X_margin[i], X_texture[i])[0, 1] for i in range(n_samples)]
    )
    engineered["shape_texture_corr"] = np.array(
        [np.corrcoef(X_shape[i], X_texture[i])[0, 1] for i in range(n_samples)]
    )

    # Ratios
    engineered["margin_shape_ratio"] = np.mean(X_margin, axis=1) / (
        np.mean(X_shape, axis=1) + 1e-10
    )
    engineered["margin_texture_ratio"] = np.mean(X_margin, axis=1) / (
        np.mean(X_texture, axis=1) + 1e-10
    )
    engineered["shape_texture_ratio"] = np.mean(X_shape, axis=1) / (
        np.mean(X_texture, axis=1) + 1e-10
    )

    # FFT features
    fft_margin = np.abs(np.fft.fft(X_margin, axis=1))[:, :16]
    fft_shape = np.abs(np.fft.fft(X_shape, axis=1))[:, :16]
    fft_texture = np.abs(np.fft.fft(X_texture, axis=1))[:, :16]
    engineered["fft_margin_mean"] = np.mean(fft_margin[:, 1:], axis=1)
    engineered["fft_shape_mean"] = np.mean(fft_shape[:, 1:], axis=1)
    engineered["fft_texture_mean"] = np.mean(fft_texture[:, 1:], axis=1)
    engineered["fft_margin_std"] = np.std(fft_margin[:, 1:], axis=1)
    engineered["fft_shape_std"] = np.std(fft_shape[:, 1:], axis=1)
    engineered["fft_texture_std"] = np.std(fft_texture[:, 1:], axis=1)

    # RMS
    engineered["margin_rms"] = np.sqrt(np.mean(X_margin**2, axis=1))
    engineered["shape_rms"] = np.sqrt(np.mean(X_shape**2, axis=1))
    engineered["texture_rms"] = np.sqrt(np.mean(X_texture**2, axis=1))

    # Peak indices
    engineered["margin_peak_idx"] = np.argmax(X_margin, axis=1)
    engineered["shape_peak_idx"] = np.argmax(X_shape, axis=1)
    engineered["texture_peak_idx"] = np.argmax(X_texture, axis=1)

    # Distance metrics
    engineered["margin_shape_l2"] = np.sqrt(np.sum((X_margin - X_shape) ** 2, axis=1))
    engineered["margin_texture_l2"] = np.sqrt(
        np.sum((X_margin - X_texture) ** 2, axis=1)
    )
    engineered["shape_texture_l2"] = np.sqrt(np.sum((X_shape - X_texture) ** 2, axis=1))

    # Smoothness
    engineered["margin_smoothness"] = np.std(np.diff(X_margin, axis=1), axis=1)
    engineered["shape_smoothness"] = np.std(np.diff(X_shape, axis=1), axis=1)
    engineered["texture_smoothness"] = np.std(np.diff(X_texture, axis=1), axis=1)

    engineered_df = pd.DataFrame(engineered)
    return X_raw, engineered_df


print("Engineering features...")
X_train_raw, X_train_eng = engineer_features(
    train_df, all_feature_cols, margin_cols, shape_cols, texture_cols
)
X_test_raw, X_test_eng = engineer_features(
    test_df, all_feature_cols, margin_cols, shape_cols, texture_cols
)

X_all = np.hstack([X_train_raw, X_train_eng.values])
X_test = np.hstack([X_test_raw, X_test_eng.values])
print(f"Combined train: {X_all.shape}, test: {X_test.shape}")

# Encode labels
y_all = LabelEncoder().fit_transform(train_df["species"].values)
classes = np.unique(train_df["species"].values)
num_classes = len(classes)
test_ids = test_df["id"].values
print(f"Classes: {num_classes}")


# ============ 3. Model Definition ============
class LeafMLP(nn.Module):
    def __init__(
        self, input_dim, num_classes, hidden_dims=[512, 256, 128], dropout=0.3
    ):
        super(LeafMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_all.shape[1]
print(f"Device: {device}, Input dim: {input_dim}")

# ============ 4. 5-Fold Cross-Validation Training ============
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_indices = list(skf.split(X_all, y_all))

model_names = ["lgb", "xgb", "mlp", "lr"]
OOF_preds = {name: np.zeros((len(X_all), num_classes)) for name in model_names}
test_preds = {name: np.zeros((len(X_test), num_classes)) for name in model_names}

for fold, (train_idx, val_idx) in enumerate(fold_indices):
    print(f"\n=== Fold {fold+1}/{n_folds} ===")

    X_train_fold = X_all[train_idx]
    y_train_fold = y_all[train_idx]
    X_val_fold = X_all[val_idx]
    y_val_fold = y_all[val_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    X_test_scaled = scaler.transform(X_test)

    # --- LightGBM ---
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=8,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        X_train_scaled,
        y_train_fold,
        eval_set=[(X_val_scaled, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    OOF_preds["lgb"][val_idx] = lgb_model.predict_proba(X_val_scaled)
    test_preds["lgb"] += lgb_model.predict_proba(X_test_scaled) / n_folds
    print(f"  LGB val log_loss: {log_loss(y_val_fold, OOF_preds['lgb'][val_idx]):.4f}")

    # --- XGBoost ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_train_scaled,
        y_train_fold,
        eval_set=[(X_val_scaled, y_val_fold)],
        verbose=False,
    )
    OOF_preds["xgb"][val_idx] = xgb_model.predict_proba(X_val_scaled)
    test_preds["xgb"] += xgb_model.predict_proba(X_test_scaled) / n_folds
    print(f"  XGB val log_loss: {log_loss(y_val_fold, OOF_preds['xgb'][val_idx]):.4f}")

    # --- MLP ---
    mlp_model = LeafMLP(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train_fold)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val_fold)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 20

    for epoch in range(50):
        mlp_model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = mlp_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        mlp_model.eval()
        val_loss = 0
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = mlp_model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(X_batch)
                val_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_preds = np.vstack(val_preds)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_preds = val_preds.copy()

        if epoch - best_epoch >= patience:
            break

    OOF_preds["mlp"][val_idx] = best_val_preds

    # Test predictions for MLP
    mlp_model.eval()
    test_pred_list = []
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            outputs = mlp_model(X_batch)
            test_pred_list.append(torch.softmax(outputs, dim=1).cpu().numpy())
    test_preds["mlp"] += np.vstack(test_pred_list) / n_folds
    print(
        f"  MLP val log_loss: {log_loss(y_val_fold, best_val_preds):.4f} (best epoch {best_epoch})"
    )

    # --- Logistic Regression ---
    lr_model = LogisticRegression(
        C=1.0, max_iter=1000, multi_class="multinomial", solver="lbfgs", random_state=42
    )
    lr_model.fit(X_train_scaled, y_train_fold)
    OOF_preds["lr"][val_idx] = lr_model.predict_proba(X_val_scaled)
    test_preds["lr"] += lr_model.predict_proba(X_test_scaled) / n_folds
    print(f"  LR val log_loss: {log_loss(y_val_fold, OOF_preds['lr'][val_idx]):.4f}")

# ============ 5. Learn Ensemble Weights ============
print("\n=== Learning Optimal Ensemble Weights ===")


def objective(w):
    w = np.abs(w)
    w = w / w.sum()
    ensemble_pred = sum(w[i] * OOF_preds[name] for i, name in enumerate(model_names))
    ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
    ensemble_pred = ensemble_pred / ensemble_pred.sum(axis=1, keepdims=True)
    return log_loss(y_all, ensemble_pred)


w0 = np.ones(len(model_names)) / len(model_names)
bounds = [(0, 1)] * len(model_names)
constraint = {"type": "eq", "fun": lambda w: 1 - np.sum(w)}
result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraint)
optimal_weights = np.abs(result.x) / np.abs(result.x).sum()
print(f"Optimal weights: {dict(zip(model_names, optimal_weights))}")

# ============ 6. Final Predictions ============
print("\n=== Computing Final Predictions ===")

# Validation score
val_ensemble = sum(
    optimal_weights[i] * OOF_preds[name] for i, name in enumerate(model_names)
)
val_ensemble = np.clip(val_ensemble, 1e-15, 1 - 1e-15)
val_ensemble = val_ensemble / val_ensemble.sum(axis=1, keepdims=True)
val_score = log_loss(y_all, val_ensemble)
print(f"Ensemble Validation Log Loss: {val_score:.4f}")

# Test predictions
test_ensemble = sum(
    optimal_weights[i] * test_preds[name] for i, name in enumerate(model_names)
)
test_ensemble = np.clip(test_ensemble, 1e-15, 1 - 1e-15)
test_ensemble = test_ensemble / test_ensemble.sum(axis=1, keepdims=True)

# ============ 7. Generate Submission ============
print("\n=== Generating Submission ===")
submission_df = pd.DataFrame(test_ensemble, columns=classes)
submission_df.insert(0, "id", test_ids)
submission_df = submission_df[sample_sub.columns]
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

# Verify submission
assert (
    submission_df.shape == sample_sub.shape
), f"Shape mismatch: {submission_df.shape} vs {sample_sub.shape}"
assert list(submission_df.columns) == list(sample_sub.columns), "Column mismatch"
assert submission_df["id"].tolist() == sample_sub["id"].tolist(), "ID mismatch"
assert not submission_df.iloc[:, 1:].isna().any().any(), "NaN values in predictions"
print("Submission validation passed!")

print(f"\nFinal Validation Score: {val_score}")
