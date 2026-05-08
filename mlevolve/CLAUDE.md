# OpenADMET ExpansionRX Competition - Critical Guidelines

## MANDATORY: Log Transformation for Targets

**This is CRITICAL for correct evaluation metrics.**

### Background
The competition uses MA-RAE (Macro-Averaged Relative Absolute Error) for evaluation. According to the competition rules, **all non-log scale endpoints must be transformed to log scale**.

### Target Columns
- `LogD` - Already in log scale, **DO NOT transform**
- All other 8 targets - **MUST be log-transformed**:
  - KSOL
  - HLM CLint
  - MLM CLint
  - Caco-2 Permeability Papp A>B
  - Caco-2 Permeability Efflux
  - MPPB
  - MBPB
  - MGMB

### Implementation Requirements

#### 1. Forward Transformation (Training)
After loading training targets, apply log10 transformation:

```python
# Define columns needing transformation
log_transform_cols = [col for col in target_cols if col != 'LogD']

# Save original targets for MA-RAE calculation
train_targets_original = train_targets.copy()

# Apply log10 transformation
for col in log_transform_cols:
    mask = ~train_targets[col].isna()
    if mask.sum() > 0:
        train_targets.loc[mask, col] = np.log10(train_targets.loc[mask, col])
```

#### 2. Inverse Transformation (Validation & Testing)
Before calculating MA-RAE or creating submissions, inverse transform predictions:

```python
# Inverse transform validation predictions
for i_col, col in enumerate(target_cols):
    if col != 'LogD':
        val_predictions[:, i_col] = 10 ** val_predictions[:, i_col]

# Also inverse transform validation targets for MA-RAE
val_targets_original = val_targets.copy()
for i_col, col in enumerate(target_cols):
    if col != 'LogD':
        mask = val_mask[:, i_col] == 1
        val_targets_original[mask, i_col] = 10 ** val_targets[mask, i_col]

# For test predictions
for i_col, col in enumerate(target_cols):
    if col != 'LogD':
        test_predictions[:, i_col] = 10 ** test_predictions[:, i_col]
```

#### 3. Target Ranges for MA-RAE
Calculate target_ranges from **original scale** training data:

```python
target_ranges = compute_target_ranges(train_targets_original.iloc[train_idx].values, train_mask)
```

### Why This Matters
- Without log transformation: MA-RAE ≈ 0.04-0.06 (WRONG - 10x too good)
- With correct log transformation: MA-RAE ≈ 0.5 (matches leaderboard)

---

## RDKit API Corrections

### Common Error: GetValenceContribution
**WRONG:** `bond.GetValenceContribution()`  
**CORRECT:** `bond.GetValenceContrib()`

When extracting bond features for graph neural networks, use the correct method name without the "ution" suffix.

### Example
```python
# Correct bond feature extraction
bond_feats = [
    bond_type / 4.0,
    float(bond.GetIsAromatic()),
    float(bond.IsInRing()),
    float(bond.GetIsConjugated()),
    bond.GetValenceContrib() / 4.0  # NOT GetValenceContribution()
]
```

---

## Performance Guidelines

### Model Selection
- **Traditional ML (LightGBM, XGBoost)**: Fast, reliable, good baseline
- **Neural Networks**: Can achieve better results but require careful tuning
- **GNN**: Theoretically best for molecular data, but prone to implementation errors

### Execution Time
- Target: < 5 minutes per node
- If execution > 10 minutes, simplify the model or reduce epochs

### GPU Usage - CRITICAL

**MUST move ALL tensors to GPU to avoid device mismatch errors.**

#### Correct GPU Setup
```python
# 1. Set device at the beginning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Move model to GPU
model = YourModel().to(device)

# 3. Move ALL data to GPU during training
for batch in train_loader:
    features = features.to(device)  # ← CRITICAL
    targets = targets.to(device)    # ← CRITICAL
    mask = mask.to(device)          # ← CRITICAL
    
    outputs = model(features)
    loss = criterion(outputs, targets, mask)

# 4. Move validation data to GPU
val_features_tensor = torch.FloatTensor(val_features).to(device)  # ← CRITICAL
val_predictions = model(val_features_tensor).cpu().numpy()

# 5. Move test data to GPU
test_features_tensor = torch.FloatTensor(test_features).to(device)  # ← CRITICAL
test_predictions = model(test_features_tensor).cpu().numpy()
```

#### Common Error to Avoid
❌ **RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!**

This happens when:
- Model is on GPU but input data is on CPU
- Some tensors are created without `.to(device)`
- Intermediate variables are not moved to GPU

#### Checklist
- [ ] Model moved to GPU: `model.to(device)`
- [ ] Training features moved to GPU: `features.to(device)`
- [ ] Training targets moved to GPU: `targets.to(device)`
- [ ] Training masks moved to GPU: `mask.to(device)`
- [ ] Validation data moved to GPU: `val_data.to(device)`
- [ ] Test data moved to GPU: `test_data.to(device)`
- [ ] All custom tensors created with device: `torch.tensor(..., device=device)`

**Performance Notes:**
- Most time is spent on CPU preprocessing (RDKit features) - this is normal
- GPU training should be 20-30% of total time
- GPU utilization 30-50% is expected due to CPU preprocessing bottleneck

---

## Common Pitfalls to Avoid

1. ❌ Forgetting log transformation → Wrong metrics
2. ❌ Using `GetValenceContribution()` → AttributeError
3. ❌ Calculating target_ranges from log-transformed data → Wrong MA-RAE
4. ❌ Not inverse transforming predictions → Wrong submission
5. ❌ Overly complex models → Slow execution, low success rate
