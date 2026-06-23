---
name: small-data-transformer-finetuning
description: Procedural skill for fine-tuning transformer models on small NLP datasets under multi-class log-loss classification. Evolved from mlevolve solver execution traces via the Trace2Skill baseline (creation-from-scratch mode — this skeleton is intentionally minimal; all content below is distilled from traces).
---

# Small-Data Transformer Fine-tuning

Guidance for an automated ML solver fine-tuning large pretrained transformers
(e.g. DeBERTa-v3) on small text-classification datasets where overfitting is the
dominant risk and validation log loss is the objective.

## Core Workflow (Staged Execution)

Run the pipeline in stages, validating each before proceeding:

1. **Pre-flight environment check** — Verify key imports succeed. **NEVER import `AdamW` from `transformers`** — it was removed. Use `from torch.optim import AdamW`. If any import fails, adjust the script to match installed versions. Convert any numpy scalar types (e.g. `int64`) to native Python types (`int`, `float`) before serializing configs or metadata to JSON.
2. **Probe GPU memory before model selection** — Check `torch.cuda.mem_get_info()`. If free memory < 2 GB: fall back to TF-IDF + Logistic Regression (CPU). If < 8 GB: use DistilBERT or smaller models only. DO NOT select DeBERTa-v3-large without confirming ≥ 12 GB free.
3. **Start minimal, then scale** — First attempt must produce a valid submission. Use TF-IDF + Logistic Regression or DistilBERT. Scale up incrementally only after baseline succeeds. DO NOT combine heavy feature engineering with a large transformer on the first attempt.
4. **Data loading & splitting** — Load data, inspect label distribution. Create a stratified train/validation split. Verify shapes and dtypes. Use index-based splits (deterministic partitioning by row index). Do NOT use random splits or column-value-based splits — these cause subtle data leakage.
5. **Tokenization** — Tokenize raw text only for the transformer using the pretrained model's tokenizer. Do NOT add handcrafted features alongside transformer embeddings — they can dominate gradients and cause early overfitting. Tokenize within each CV fold to prevent leakage.
6. **Model setup** — Load via `AutoModelForSequenceClassification` with a strong pretrained backbone (e.g., DeBERTa-v3-large) and `num_labels` set. A simple linear head on the [CLS] token is the reliable default. Do NOT build custom model wrappers (manual pooling, mixture-of-experts, contrastive losses) unless a clean baseline has been benchmarked first. Freeze bottom ~33% of encoder layers (or keep only the last ~8 layers unfrozen).
7. **Fine-tuning** — Fine-tune with conservative learning rate (1e-5 to 3e-5), short training (2–4 epochs), and appropriate batch size (16 safe default). Use **differential learning rates** — lower for the backbone (~2e-5), higher for the head (~5e-5). Build in OOM recovery: wrap training in try-except catching `torch.cuda.OutOfMemoryError`, retry with halved batch size.
8. **Evaluation** — Compute validation log loss every epoch; save predictions as probabilities (not hard labels). Track train vs. validation loss divergence. Save the checkpoint that minimizes validation log loss.
9. **Ensembling** — Train each model independently and blend predictions (e.g., weighted average or logistic regression on logits). Generate out-of-fold (OOF) predictions via stratified K-fold cross-validation for each model to enable stacking without leakage. Grid-search ensemble weights over OOF predictions using the target metric.
10. **Verify scheduler total steps** — `total_steps = len(train_dataloader) * num_epochs`. DO NOT use integer division on dataset length — this underestimates actual steps.
11. **Verify dependencies** — Download or assert availability of external resources (e.g., `nltk.download('punkt_tab')`). DO NOT silently fall back to empty features.

For detailed memory-management patterns and incremental-complexity workflow, see [GPU Memory & Baseline Strategy](references/gpu-memory-and-baseline-strategy.md).

## Proven Strategies for Log-Loss Optimization

Beyond the staged workflow above, these strategies consistently improve validation
log loss on small datasets:

1. **Multi-task auxiliary regression head.** Fine-tune the transformer with both
   (a) label-smoothed cross-entropy on the target and (b) an auxiliary MSE
   regression head predicting handcrafted features from the CLS embedding, using
   a decaying alpha weight (e.g., 0.5 → 0.01). This regularizes the
   representation and reduces overfitting.
2. **Metric-aligned checkpoint selection.** Save the checkpoint that minimizes
   validation log loss (the competition metric), NOT training cross-entropy.
   Transformer models become overconfident during training, so training loss and
   evaluation log loss diverge. On small datasets, the best checkpoint often
   comes from early epochs.
3. **Post-hoc probability calibration.** After inference, apply temperature
   scaling (L-BFGS-optimized) or Platt scaling on logits to reduce
   overconfidence. These lightweight steps are critical for log-loss evaluation.
4. **Diverse multi-stream ensemble.** Combine a deep contextual model
   (DeBERTa-v3), gradient-boosted trees (XGBoost on CLS embeddings + tabular
   features), and sparse n-gram features (TF-IDF + Logistic Regression). Optimize
   ensemble weights directly against validation log loss using SLSQP or
   Nelder-Mead — do NOT use simple uniform averaging.
5. **Temporal checkpoint ensembling.** During extended fine-tuning, save
   checkpoints at regular intervals. Select the top-3 by validation log loss and
   combine predictions using inverse-loss weighting.
6. **High-confidence pseudolabeling.** Pseudolabel test samples with
   max probability > 0.8–0.95, append to training set, and retrain.
7. **Supervised Contrastive Loss.** Add a SupCon loss alongside cross-entropy
   (e.g., 0.7 × CE + 0.3 × SupCon) via a separate projection head. This enforces
   tighter class clustering and improves generalization.
8. **Two-phase full-data retraining.** Retrain the best checkpoint on 100% of
   the data (train + validation combined) for 2–3 epochs at a very low learning
   rate (e.g., 5e-6). This incorporates additional data without catastrophic
   forgetting.

For detailed implementation patterns for each strategy, see
[Proven Strategies](references/proven-strategies.md) and
[Proven Configurations](references/proven-configurations.md).

## Pre-Execution Validation (MANDATORY)

Before running any generated or edited script:

1. **No patch artifacts**: The file contains no `<<<<<<<`, `=======`, `>>>>>>>`, `<<<<<<< SEARCH`, or `>>>>>>> REPLACE` markers. These cause immediate `SyntaxError`.
2. **Complete file**: Emit the full file content for each iteration — never inline SEARCH/REPLACE patch directives.
3. **Syntax check**: Run `python -m py_compile <file>` or `import py_compile; py_compile.compile("script.py", doraise=True)` before submitting to the runtime.
4. **Variable lifecycle trace**: Scan top-to-bottom; for each variable reference, confirm an assignment exists earlier in the file. Focus on: `train_indices`, `val_indices`, `X_train`, `X_val`, `train_features`, `val_features`, `oof_predictions`.
5. **Import completeness**: Every class/function used is imported. After merging code segments, imports from one block often end up after usages in another.
6. **Plan consistency**: Confirm each intended modification (removed methods, changed hyperparameters) is actually reflected in the final source. Remove ALL dead code from previous approaches.
7. **Global rename check**: When introducing new variables, search the ENTIRE script for all references to old variable names. Do NOT rely on targeted patches.
8. **Function signature consistency**: After modifying any function signature, immediately locate ALL call sites and confirm argument count, order, and names match.
9. **No partial refactoring**: If a structural change invalidates old variable names, do a comprehensive find-and-replace across the whole script.
10. **Single-batch dry run**: Before entering the training loop, fetch one batch and inspect shapes and dtypes. Verify tensor shapes in custom modules — especially before `bmm`, `matmul`, or attention operations.

These checks are cheap and catch the most common failure: malformed code files that crash before any training logic runs.

See [Code Validation Patterns](references/code-validation.md) for detailed validation patterns and common code-generation pitfalls.

## Pipeline Ordering Rules

Generate training scripts in this top-down order to prevent `NameError` crashes:

1. Imports
2. Configuration block (every hyperparameter and constant)
3. Data loading & preprocessing
4. Label encoding
5. Cross-validation split (produces index variables)
6. FOR EACH FOLD: subset data → feature engineering → tokenization → model init → training → evaluation
7. Aggregate results
8. Save outputs

**Never reference a variable before its definition.** Before executing a long script, scan top-to-bottom and confirm every variable used in each section was assigned by an earlier statement.

## Critical: Self-Contained Feature Functions

**Every feature extraction function must be self-contained.** Do NOT assume a DataFrame column created in one function is available in another function's scope.

- **DO** compute needed columns locally inside each function, or pass the full DataFrame as an explicit argument.
- **DO NOT** reference columns like `features["word_count"]` unless that column was created within the same function or passed in.
- **Validate after each step**: `print(df.columns.tolist())` or `assert 'word_count' in df.columns` before chaining transformations.

For detailed patterns on safe feature engineering and incremental validation, see [Feature Engineering Patterns](references/feature-engineering-patterns.md).

## Model Construction: Use Config Objects

DO NOT pass hyperparameters like `dropout` or `attention_dropout` directly to `AutoModelForSequenceClassification.from_pretrained()`. Different transformer architectures have different configuration schemas — for example, `ModernBertForSequenceClassification` does NOT accept `hidden_dropout_prob` as a kwarg.

Instead, construct a `Config` object first:
```python
from transformers import AutoConfig, AutoModelForSequenceClassification

config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
config.hidden_dropout_prob = 0.1

model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
```

**Fallback pattern**: if `from_pretrained(**kwargs)` raises `TypeError`, retry with only `num_labels` (the one universally supported kwarg) so the pipeline still produces a baseline result.

For detailed per-architecture config parameter names, see [Model Config Guide](references/model-config-guide.md).

## Model Introspection & Parameter Grouping

Different HuggingFace model families use **different internal attribute paths** for the base model:

| Architecture | Base-model attribute |
|---|---|
| BERT | `model.bert` |
| RoBERTa | `model.roberta` |
| DeBERTa-v3 | `model.deberta` |
| ModernBERT | `model.model` |

### Before writing layer-specific code (freezing, unfreezing, discriminative LR):

1. **Print parameter names** to discover the actual structure.
2. **Verify every attribute path** your code will touch actually exists.
3. **Group parameters by name patterns**, never hard-coded paths.

**DO NOT** write `for name, param in model.parameters()` — this raises `ValueError` because `.parameters()` yields tensors, not tuples. Use `.named_parameters()`.

**DO NOT** filter parameters by `.name` attribute — use `isinstance(module, ...)` or `param.ndim` checks instead. Standard PyTorch parameters do not expose `.name`.

For detailed attribute-discovery patterns, see [Model Introspection Patterns](references/model-introspection-patterns.md).

## Evaluation Metric Safety

**Use `sklearn.metrics.log_loss` for all evaluation metrics.** Never implement log loss manually — manual implementations are susceptible to shape-mismatch crashes and domain errors (log of zero).

Before computing any metric, sanitize predictions:
```python
val_probs = np.nan_to_num(val_probs, nan=1.0/n_classes, posinf=1.0, neginf=0.0)
val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
```

**NaN diagnostic checkpoint:** If the model's training loss returns valid values but your evaluation metric returns NaN, the bug is in the metric calculation or data pipeline, not the model.

For metric-validation patterns and root-cause debugging, see [Metric & Training Diagnostics](references/metric-and-training-diagnostics.md).

## Critical Safeguards

1. **NaN loss prevention**: Always add gradient clipping (`torch.nn.utils.clip_grad_norm_`, max norm 1.0; use 0.5 for very small datasets), small epsilon to denominators in custom losses, and detect NaN in the first few steps to halt early.
2. **Checkpoint path consistency**: Use a single `best_model_path` variable for both saving and loading — never hardcode filenames independently.
3. **Missing-checkpoint fallback**: Before loading the best model for inference, check `os.path.exists(best_model_path)`. If missing, generate a baseline submission (uniform class probabilities).
4. **fp16-safe masking**: Never use `-1e9` for attention masking under `autocast()`. Use `-float('inf')` (PyTorch softmax correctly handles it) or `-1e4` as an alternative.
5. **GradScaler lifecycle**: `scaler.scale(loss).backward()` → at most ONE `scaler.unscale_(optimizer)` → `scaler.step(optimizer)` → `scaler.update()`.
6. **Fault-tolerant CV**: Wrap each fold in try-except so a single fold failure does NOT abort the entire pipeline.
7. **AWP NaN warning**: DO NOT use AWP (Adversarial Weight Perturbation) with small datasets — the double-backward mechanism corrupts gradients after parameter restoration, producing NaN values.

For concrete code patterns implementing these safeguards, see [Training Safeguards](references/training-safeguards.md) and [Numerical Stability Patterns](references/numerical-stability-patterns.md).

## Environment Constraints and Resource Budgeting

- **Always use `num_workers=0`** in DataLoaders — multi-worker DataLoaders rely on shared memory which frequently exhausts in constrained environments.
- **Set `pin_memory=False`** when working with high-dimensional sparse matrices or under strict memory constraints.
- **DO NOT** combine a large transformer with massive handcrafted feature sets simultaneously.
- **Catch and retry resource errors**: wrap training in try/except for `RuntimeError` (bus error, CUDA OOM). On catch, retry with reduced batch size.
- **Proactively manage resources**: call `gc.collect()` and `torch.cuda.empty_cache()` between epochs.
- **Set `os.environ["TOKENIZERS_PARALLELISM"] = "false"`** at the top of the script before importing transformers/tokenizers.

## GPU Memory Safety

- **Start conservative**: use batch size ≤ 8 and max_seq_len ≤ 256 for the first transformer run.
- **Enable memory-saving techniques**: gradient checkpointing, mixed precision (fp16), and dynamic padding.
- **Pre-load GPU memory check**: Before loading any large model, query free GPU memory via `torch.cuda.mem_get_info()`.
- **If OOM occurs**: halve batch size, reduce max_seq_len by 64, or switch to a smaller model variant.
- **OOM resolution order**: Remove adversarial weight perturbation → remove auxiliary losses → downgrade model size → reduce batch size.

**Model selection rules based on free memory:**
- **< 4 GB free**: Small/base model, max_seq_len 128, batch_size 8, gradient checkpointing.
- **4–12 GB free**: Base model variant, max_seq_len 256, batch_size 16–32.
- **> 12 GB free**: Large variant may be viable; still use max_seq_len ≤ 512.

For copy-pasteable snippets and edge cases, see [GPU Memory & Device Fallback](references/gpu-memory-and-device-fallback.md).

## Overfitting Controls

- Use a held-out validation split (stratified) and track log loss every epoch.
- Prefer low learning rates (1e-5 to 2e-5) with weight decay and small batch sizes.
- Use dropout (0.1–0.3) in the classification head.
- Consider early stopping based on validation log loss; stop when it stops improving.
- For very small datasets (< 1k samples), freeze early transformer layers or use feature extraction mode.
- Consider layer-wise learning rate decay if fine-tuning all layers.
- Monitor train vs. validation loss divergence.
- **Label smoothing (0.05)** is the first-line intervention for overfitting. Use 0.1 only if overfitting is severe.
- **Reduce regularization when loss plateaus**: excessive dropout (>0.3) and label smoothing (>0.15) can cause underfitting even when the model appears to overfit. If validation loss stalls, lower dropout to 0.2 and label smoothing to 0.06.

### Anti-Overfitting Defaults for Small Datasets

| Parameter            | Typical Range         |
|---------------------|----------------------|
| Learning rate        | 1e-5 – 2e-5          |
| Epochs               | 2 – 4 (use early stopping) |
| Batch size           | 16 – 32 (gradient accum if GPU-limited) |
| Weight decay         | 0.01 – 0.1           |
| Max sequence length  | 128 – 512 (dataset-dependent) |
| Dropout              | Use model defaults unless tuned |
| Label smoothing      | 0.05 (0.1 if severe overfitting) |
| LR scheduler         | Linear warmup + cosine decay |
| Gradient clipping    | max_norm 0.5–1.0     |

## Regularization Stack (essential for small data)

On datasets with ~10K–20K samples, combine ALL of the following to prevent overfitting:
- **Label smoothing**: 0.05 recommended (mild regularization while preserving confident predictions for log-loss). Use 0.1 only if overfitting is severe.
- **Multi-Sample Dropout**: Apply K=4 independent dropout masks to the pooled representation and average logits across masks.
- **Stochastic Weight Averaging (SWA)** — start late in training (e.g., last 25% of epochs) to find flatter minima.
- **Focal Loss**: Optionally replace standard `CrossEntropyLoss` with Focal Loss (gamma=2.0) to focus training on hard-to-classify examples.
- **Mixed precision**: `torch.cuda.amp` or `fp16` in Trainer.
- **Weight decay**: 0.01–0.1 to regularize the classifier head.
- **Gradient clipping** (max_norm=1.0) to stabilize training.
- **Differential learning rates**: backbone `2e-5`, classification head `5e-5`.
- **Early stopping**: patience 3–5 on validation log loss.

## Scheduler Configuration

Use `SequentialLR` composing `LinearLR(start_factor=0.1)` for warmup (~10% of steps or ~2 epochs) then `CosineAnnealingLR(eta_min=1e-6)`. Call `scheduler.step()` exactly once per batch after `optimizer.step()`.

**Do NOT use `CosineAnnealingWarmRestarts`** — mid-training LR spikes destabilize fine-tuning. DO NOT combine manual warmup logic inside the training loop with PyTorch schedulers — this causes the LR to freeze at a constant value mid-training. DO NOT apply warmup twice.

Alternatives: **ReduceLROnPlateau** (factor=0.5, patience=2, min_lr=1e-7) or **OneCycleLR** (`max_lr=2e-5`, `pct_start=0.1`, per-epoch stepping) have also proven effective.

**Diagnose training stalls via LR inspection**: When validation loss plateaus, inspect the actual learning-rate trajectory. A plateau caused by a buggy scheduler freezing the LR is a model-capacity illusion.

## Early Stopping & Best-Checkpoint Restoration

Save model state_dict whenever validation log loss improves. On early stopping trigger, **load the best checkpoint** before generating predictions. Never evaluate on the final degraded epoch. Track EMA (Exponential Moving Average, decay≈0.999) checkpoints alongside raw weights.

## Critical: Pipeline Isolation

Wrap each independent model training step in try-except so a single model failure does NOT abort the entire pipeline.

## Critical: Verify Library API Compatibility

- **XGBoost**: Do NOT pass `early_stopping_rounds` to `.fit()`. Pass it to the constructor.
- **DO NOT** pass `verbose=True` to `ReduceLROnPlateau` or any PyTorch scheduler — the parameter was removed.
- **DO NOT** pass `correct_bias` to `AdamW` — it was removed.
- Prefer constructor parameters over fit-time keyword arguments.

## Critical Pandas Indexing Rules

- **NEVER** pass a boolean mask to `.iloc[]` — it raises `NotImplementedError`. Use `.loc[]` or `df[mask]`.
- **ALWAYS** preserve index arrays returned by `train_test_split`.
- **DO NOT** pass `axis=1` to Series reduction methods (`.any()`, `.all()`, `.sum()`).

For detailed pandas split/index patterns, see [Pandas Data Splitting](references/pandas-data-splitting.md).

## Custom Feature Extraction — Shape Safety

When writing custom feature extraction functions, the output **must** be a 2D array of shape `(n_samples, n_features)`. Never return a flat or 1D array. Test on a small sample and assert the shape before integrating.

## Device Consistency

Ensure **every** tensor that interacts with GPU-resident modules is explicitly moved to the correct device. Common sources of crashes: validation tensors, labels, and logits built from numpy conversions.

## Dependency Safety (Critical)

1. **Prefer standard library** for simple parsing — use `text.split()` instead of `nltk.word_tokenize()`.
2. **Provision at runtime** — if you must use NLTK, call `nltk.download('punkt')` and `nltk.download('punkt_tab')` at the top of the script, wrapped in `try/except`.
3. **Fail fast** — run a quick smoke test of imports and data loads before launching long training loops.
4. **Isolate non-essential steps** — wrap optional feature extraction in `try/except`.

See [Robust Pipeline Patterns](references/robust-pipeline-patterns.md) for concrete code snippets.

## Code Generation Discipline

- **Never embed edit directives in executable code.** Phrases like "Now replace the training loop..." must be applied as actual edits.
- **Never place natural-language reasoning inside code blocks.** Use `#` comments or docstrings.
- After generating or editing code, scan for any natural-language instruction fragments.
- When debugging, read the **exact exception type and traceback** before planning a fix.
- **Fix syntax errors surgically.** When a pipeline fails due to embedded non-code text or syntax errors, locate the exact offending lines, remove them, and complete any missing code structure. Do NOT rewrite the pipeline from scratch — preserve working sections.

## Smoke Test Checkpoint

Before launching the full training loop, run a quick initialization check: instantiate the model and perform one forward pass on a tiny batch.

## Verify Initialization After Refactors

- **Trace every variable reference**: Before running, statically verify that each variable called in the training/evaluation loop has a corresponding initialization.
- **Maintain strict variable naming consistency.**
- **Update both sides**: If replacing a component, ensure the new object is defined AND assigned before it is referenced.
- **DO NOT** leave dangling references to variables that were removed or renamed.

## Defensive Error Handling

Wrap non-critical operations in `try/except` blocks. **Critical operations** (data loading, model creation, training loop, evaluation) should NOT be silently swallowed. **Non-critical operations** (caching, logging, debug artifacts) should degrade gracefully.

## Critical Warnings

### Avoid Complex Custom Architectures
On small datasets, prefer the pretrained transformer + linear head pattern. Do NOT add multi-branch fusion layers unless you explicitly verify every tensor shape. Always benchmark a clean cross-entropy baseline before adding auxiliary loss terms.

### Gradient Checkpointing + Gradient Accumulation
**DO NOT combine `gradient_checkpointing_enable()` with gradient accumulation.** This causes a fatal `RuntimeError`. If you must combine them, explicitly set `use_reentrant=False`.

### Regex Metacharacters in Pandas String Accessors
`Series.str.count`, `.str.replace`, `.str.contains` all compile their pattern argument as regex. Escape literal special characters. For literal character counting, prefer `text.apply(lambda s: s.count("("))`.

See [Safe Text Processing](references/safe-text-processing.md) for detailed patterns.

### BatchNorm Crash Prevention
- DO NOT use `BatchNorm` layers in custom architectures — prefer `LayerNorm`.
- Always set `drop_last=True` on training DataLoaders to avoid a final batch of size 1.

### In-Memory Model Preservation
If the model architecture is modified dynamically (e.g. custom pooling layers, attention heads added at runtime), DO NOT save/reload via `state_dict`. Custom keys (e.g. `attention_pool`) will cause load mismatches. Keep the trained model instance in memory and use it directly for evaluation and inference.

## Critical Pitfalls

### Label Smoothing — Use the Loss Function, Not the Model Constructor
`AutoModelForSequenceClassification.from_pretrained()` does **not** accept `label_smoothing_factor`. Use `nn.CrossEntropyLoss(label_smoothing=0.05)` in the training loop.

### torch.load and PyTorch 2.6+ `weights_only` Default
PyTorch 2.6 changed `torch.load`'s default `weights_only` from `False` to `True`. Explicitly set `weights_only=False` for non-tensor objects.

### Caching Intermediate Artifacts
**DO NOT** cache complex framework-specific objects (e.g., `BatchEncoding`, tokenizers) via `torch.save` or `pickle`. **PREFER** portable formats (`.npy`, `.npz`, CSV, JSON).

### NumPy Serialization Pitfall
When saving arrays containing strings, NumPy serializes them as object dtype. Reload with `np.load(path, allow_pickle=True)` — both save and load flags must be set. Proactively cast all numpy scalars to native Python types before any `json.dump` or `json.dumps` call.

### Common Runtime Pitfalls
- **NameError from ordering**: The most frequent crash. If code references `train_indices`, the KFold loop must appear earlier.
- **Complexity-induced defects**: Bundling 3+ advanced techniques into one script makes integration bugs hard to track.
- **Tokenizing before splitting**: Tokenize within each CV fold to avoid data leakage.
- **Too many epochs**: On small data, 2–4 epochs with early stopping usually suffices.
- **Near-random performance**: If validation log loss is close to the random baseline (e.g., ~1.099 for 3-class), the problem is a fundamental architecture or data-pipeline flaw — not a hyperparameter issue. Replace the model/pipeline rather than incrementally tuning.
- **Missing OOF/test-ensemble code**: A common failure is training folds successfully but missing the downstream prediction/submission code that completes the pipeline.

For correct script ordering patterns, see [Code Structure & Validation](references/code-structure-and-validation.md).

## HuggingFace Model Output Handling

`AutoModelForSequenceClassification` returns a `SequenceClassifierOutput`, **not** a raw tensor. Always extract `.logits`. Alternatively, pass `labels` directly to use built-in loss. DO NOT apply tensor functions directly to Hugging Face model outputs — always extract `.logits` first: use `torch.softmax(output.logits, dim=-1)`, NOT `torch.softmax(output, dim=-1)`.

## Timeout Safety Net

Always integrate elapsed-time checks into the training loop so progress is saved before a hard system timeout. Track elapsed time and break out of remaining epochs gracefully ~10–15 minutes before the deadline.

## Critical Submission Requirements

**Always explicitly normalize prediction probabilities before saving.** Apply `probs = probs / probs.sum(axis=1, keepdims=True)`. Always write the final CSV with an explicit, hardcoded name: `df.to_csv("submission.csv", index=False)`. Validate output schema immediately after writing. Verify file path, column names, header format, and print statement format against the task specification.

## Artifact Safety Patterns

When saving/loading intermediate pipeline artifacts, see [Artifact Round-Trip Safety](references/artifact-round-trip-safety.md) for concrete save/load patterns.

## Debugging: Trace Data Types Through Pipeline

When a method call fails (e.g., `.toarray()` on a numpy array), trace the object type at each transformation step:
- After `np.vstack()` or `np.hstack()`, the result is a dense numpy array — do NOT call sparse-matrix methods like `.toarray()` or `.tocoo()` on it.
- After `scipy.sparse.hstack()` or `TfidfVectorizer.transform()`, the result is a sparse matrix — call `.toarray()` to densify before further numpy operations.
- Print `type(obj)` and `obj.shape` at each stage to isolate mismatches.

## Advanced Techniques (optional alternatives to simple head)

The following techniques have demonstrated strong validation log-loss performance and can be combined with the core workflow above:

1. **Attention pooling head** — For style/authorship tasks where discriminative cues are distributed throughout the sequence, replace [CLS]-only pooling with attention pooling. See [Attention Pooling & Training Config](references/attention-pooling-and-training-config.md).
2. **Attention-weighted multi-layer pooling** — Extract last 4 layers, masked mean pooling per layer, learnable weighted sum. See [Pooling Strategy](references/pooling-strategy.md).
3. **Multi-pooling architecture** — Combine CLS, mean, and attention-weighted pooling, concatenated through a multi-layer classifier. See [Architecture Patterns](references/architecture-patterns.md).
4. **Feature fusion** — Concatenate engineered features with the transformer representation via a dedicated MLP. See [Feature Fusion Architecture](references/feature-fusion-architecture.md).
5. **EMA re-registration after unfreezing** — When using EMA with staged fine-tuning, re-register EMA after unfreezing so all trainable parameters are tracked.
6. **Progressive unfreezing** — Use with caution; a layer-count guard is needed (ALBERT has a single shared layer group — naive unfreezing throws index errors).
7. **Mixup augmentation** — 50% of batches; interpolate both inputs and labels.
8. **Word-dropout augmentation** — Randomly replace tokens with unknown token (p=0.1).

For detailed guidance on these and other advanced techniques, see [Advanced Techniques](references/advanced-techniques.md) and [Advanced Configurations](references/advanced-configurations.md).

## Feature Engineering

For tasks where **stylistic or lexical signals** matter (authorship attribution, spam detection), combine transformer embeddings with handcrafted features (TF-IDF, stylometric, statistical). See [Feature Engineering](references/feature-engineering.md) for feature families, fusion approaches, and task-specific recommendations.

## Proven Patterns

- **Diverse ensembles mitigate individual model instability.** Even when a transformer's validation log loss degrades after checkpoint reload, a diverse ensemble (transformer + GBT + linear) can still achieve strong final log loss because the complementary models compensate.
- **Multi-granularity TF-IDF for text features.** Combine character n-grams (2–7), word n-grams (1–3), and punctuation-sequence n-grams to capture stylistic signals at multiple linguistic levels.
- **GBDT on embeddings + features can beat the transformer alone.** The GBDT exploits stylistic signal not fully captured by contextual embeddings.
- **Simple linear models on strong text features can be highly competitive** — do not dismiss them.

## Critical: Prevent Data Leakage in Stacked Models

When stacking or ensembling models that use OOF predictions from one stage as features for the next:
- Train the meta-model **only on the training portion** of each fold.
- Validate the meta-model **only on the held-out portion** of that fold.
- **Never** validate a meta-model on the full OOF set it was trained on — this produces artificially low log loss (observed: 0.0102 vs. true 0.1975).
- **Fit feature selectors on training data only.** Fit any scaler (e.g., `MaxAbsScaler`) and selector (e.g., `SelectKBest`) on the training split, then `.transform()` validation and test individually.

## Validation Protocol

1. Compute OOF log loss across all folds as the primary metric.
2. If OOF log loss is significantly lower than any single fold, investigate fold-level variance — high variance signals instability.
3. Report mean ± std across folds; prefer configurations with low std even if mean is slightly higher (more stable = more trustworthy on small data).
4. After ensemble weight search, confirm the blended OOF log loss is lower than any single model; if not, revisit feature diversity or model hyperparameters.
5. Report the final validation log loss from the held-out fold as the trusted estimate of submission performance.

## Numerical Stability Safeguards

Critical for multi-fold CV where a single NaN can invalidate an entire run:
1. **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
2. **Loss clamping**: `loss = torch.clamp(loss, min=-10, max=10)`
3. **Mixed precision**: Use `torch.cuda.amp` with gradient scaling.
4. **Avoid text-corrupting preprocessing**: Do NOT use random truncation; use head+tail or head-only truncation instead.
5. **Check for NaN after each fold**: `if torch.isnan(loss): skip fold and log warning`.

## Key Warnings

- **DO NOT** use a single learning rate for backbone and head on small data — always use differential rates.
- **DO NOT** skip label smoothing — it is critical for log-loss optimization.
- **DO NOT** rely on CLS-token pooling alone for tasks involving subtle stylistic patterns.
- **DO NOT** rely on a single transformer model when multiple model families are feasible — ensembling consistently outperforms.
- **DO NOT** skip handcrafted features on tasks where writing style or domain-specific signal is discriminative.
- **DO NOT** use arbitrary ensemble weights — always grid-search on validation data against the target metric.
- **DO NOT** use a single train-val split as the sole basis for model selection — it produces high-variance estimates on small data.
- **DO NOT** assume more augmentation is better — on small datasets, aggressive augmentation can introduce spurious cues that cause memorization.
- **DO NOT** unfreeze all layers at once on small datasets — selective top-N freezing prevents overfitting and catastrophic forgetting.
- **DO NOT** launch a full training run without first verifying forward-pass tensor shapes on a single batch.

## Anti-Patterns to Avoid

- Two-phase focal-loss training
- Any architectural change that replaces rather than augments a working baseline
- If the planned pipeline is incomplete or references undefined functions/classes, replace it immediately with a simpler, fully-defined pipeline using standard library components.

## CV Indexing Safety

- **NEVER** index a fold-local feature matrix with absolute indices from the full dataset.
- When recomputing features inside a CV loop, use `iloc` or boolean masks derived from the fold split.

For detailed CV indexing patterns, see [CV Indexing Patterns](references/cv-indexing-patterns.md).

## Training Configuration

For detailed configuration guidance for gradient checkpointing conflicts, focal loss, early stopping, and fine-tuning hyperparameters, see [Fine-Tuning Configuration](references/finetuning-configuration.md) and [Transformer Fine-tuning Patterns](references/transformer-finetuning-patterns.md).

## Proven Hyperparameters

For concrete hyperparameter values from a successful DeBERTa-v3-large run (val log loss 0.0776), see [Proven Hyperparameters](references/proven-hyperparameters.md).

## Overfitting Prevention Details

For layer freezing code, regularization configs, and `.logits` extraction pitfall details, see [Overfitting Prevention](references/overfitting-prevention.md).

## Ensemble & Stacking Patterns

For leakage-safe stacking, grid search code, model diversity, index-based splitting, transformer hyperparameters, and XGBoost API pitfall, see [Ensemble & Stacking Patterns](references/ensemble-and-stacking-patterns.md) and [Ensemble Patterns](references/ensemble-patterns.md).

## Stylometric Features

For detailed handcrafted feature categories proven equally predictive as transformer embeddings for authorship attribution, see [Stylometric Features](references/stylometric-features.md).
