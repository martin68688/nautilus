---
name: small-data-transformer-finetuning
description: Procedural skill for fine-tuning transformer models on small NLP datasets under multi-class log-loss classification. Evolved from mlevolve solver execution traces via the Trace2Skill baseline (creation-from-scratch mode — this skeleton is intentionally minimal; all content below is distilled from traces).
---

# Small-Data Transformer Fine-tuning

Guidance for an automated ML solver fine-tuning large pretrained transformers
(e.g. DeBERTa-v3) on small text-classification datasets where overfitting is the
dominant risk and validation log loss is the objective.



## Critical Compatibility

- Import `AdamW` from `torch.optim`, NOT from `transformers`. The `transformers.AdamW` was deprecated and removed.
- When using `torch.optim.AdamW`, do NOT pass HuggingFace-specific arguments like `correct_bias`.
- Some newer model classes (e.g., ModernBERT) reject non-standard hyperparameters passed directly to `from_pretrained()`. Pass parameters like `hidden_dropout_prob` through a `Config` object first.
- **DO NOT combine gradient checkpointing with gradient accumulation** — causes "Trying to backward through the graph a second time" errors. Choose one.
- Set `os.environ["TOKENIZERS_PARALLELISM"] = "false"` at the top of scripts to prevent deadlocks.

## Model Selection

- Default to `microsoft/deberta-v3-small` or `-base` for small datasets. Scale up to `-large` only when validation loss has plateaued AND > 20 GiB free GPU memory is confirmed.
- Large models overfit quickly on small data and risk OOM.

## Core Workflow

1. **Start with a simple baseline** (TF-IDF + Logistic Regression) to validate the full pipeline end-to-end before adding complexity.
2. **Check resources first** — verify GPU memory (`torch.cuda.mem_get_info`), disk space, and shared memory before any heavy computation.
3. **Load data and create stratified train/validation splits.** Use `StratifiedShuffleSplit` or `StratifiedKFold` — never random splits.
4. **Encode labels to integers** before creating PyTorch Datasets. Apply `LabelEncoder`; pass encoded labels to Dataset — never raw strings.
5. **Feed raw tokenized text directly to the transformer** by default. Do NOT add TF-IDF or handcrafted features to transformer input unless task-specific patterns (authorship, stylometry) justify it.
6. **Keep all transformer layers trainable from the start.** Do NOT freeze the backbone when using mixed precision — freezing + autocast causes NaN predictions.
7. **Fine-tune with conservative hyperparameters**: low LR (1e-5 to 2e-5), early stopping (patience 2–3), gradient clipping (max_norm=1.0).
8. **Run a forward-pass smoke test** — pass a single dummy batch through the model immediately after construction to catch signature mismatches, shape errors, and device issues.
9. **Add complexity one technique at a time.** Introduce one modification per iteration; run a quick sanity check after each.
10. **Apply layered probability calibration** — Platt scaling on individual logits, temperature scaling on ensemble output.
11. **Build complementary models on diverse feature spaces** (XGBoost on CLS+stylometric, Logistic Regression on TF-IDF).
12. **Ensemble via weighted averaging** with weights optimized on validation log loss. Isolate each model in try-except.
13. **Always produce a submission.** Even if auxiliary models fail, generate predictions from whatever models trained successfully.
14. **Validate the final submission**: confirm predictions cover all classes, check for NaN/Inf, verify probability distribution is not degenerate, normalize rows to sum to 1.

## Pre-Execution Validation (Mandatory)

Before submitting any generated or patched Python script:

1. **Scan for merge/diff markers** — search for `<<<<<<<`, `=======`, `>>>>>>>`, `REPLACE` and remove them. Treat as hard errors.
2. **Syntax-check the file** — run `python -m py_compile <file>` or `ast.parse` to confirm it parses.
3. **Sanitize edit artifacts** — strip non-code artifacts (diff markers, banner separators, markdown fences, planning commentary).
4. **Verify variable consistency** — scan for undefined variable references, especially after merging code sections. Trace every variable from assignment to usage.
5. **Signature ↔ call-site match** — after modifying any function's parameters, locate every call site and confirm arguments match.
6. **Cross-function dependency tracing** — verify every DataFrame column or variable accessed by a function is passed as an argument or computed locally.
7. **Confirm fixes applied** — re-read the relevant section to verify the fix is actually present in the saved file.
8. **End-to-end re-read** — for scripts > 200 lines, re-read the full script once to verify definitions and references are aligned.

> **DO NOT** submit a script without confirming it passes `py_compile`. A syntax error caught at runtime wastes the entire execution attempt.

For detailed validation patterns, see [references/code-validation-patterns.md](references/code-validation-patterns.md).

## Environment Constraints

**DataLoader workers**: Always use `num_workers=0` and do NOT set `pin_memory=True` in constrained containers. Multi-worker DataLoaders exhaust `/dev/shm` shared memory, causing bus errors.

**Training loop**: Keep it minimal and synchronous — no AMP, no pinned memory, no multi-worker infrastructure unless explicitly justified.

## Resource Constraints

Before any training or feature extraction:
- Check available disk and shared memory (`shutil.disk_usage`, `/dev/shm` capacity). If free disk < ~2 GB, fall back to smaller model.
- Prefer smaller models unless validation loss clearly justifies larger.
- Cap TF-IDF features (≤5k), use sparse matrices, avoid materializing large dense files.
- Cap epochs at 3–5 for small datasets.
- Clean up intermediate files between pipeline stages.

For detailed patterns, see [references/resource-and-memory-management.md](references/resource-and-memory-management.md).

## GPU Memory Safety

1. **Pre-check GPU memory** via `torch.cuda.mem_get_info()` before `.cuda()`. If free memory < model size × 3, do NOT load onto GPU.
2. **Try-except with fallback** — wrap GPU operations catching `torch.OutOfMemoryError` and `RuntimeError`. Fall back to CPU, smaller model, or classical features only.
3. **Memory optimization priority**: (1) reduce micro-batch size, (2) enable mixed precision, (3) gradient accumulation (disable checkpointing), (4) gradient checkpointing (no accumulation), (5) reduce max_length.
4. **Modular feature extraction** — extract feature groups as independent stages with individual error handling.

## Model Instantiation Safety

**DO NOT** pass `dropout`, `attention_dropout`, `label_smoothing`, `weight_decay`, or other hyperparameter kwargs directly to `from_pretrained()`. Use a Config object:
```python
config = AutoConfig.from_pretrained(model_name)
config.hidden_dropout_prob = 0.1
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, num_labels=num_classes)
```

**Verify kwargs per architecture** — print `AutoConfig.from_pretrained(model_name).to_dict().keys()` before passing non-standard kwargs.

**Defensive instantiation** with try-except fallback to `num_labels` only.

**Model layer access** — never hardcode attribute paths like `model.roberta.pooler`. Inspect with `model.named_parameters()` first; use dynamic pattern matching.

For detailed API patterns, see [references/model-api-and-subclassing.md](references/model-api-and-subclassing.md).

## Pipeline Ordering and Variable Safety

1. **Definition-before-use**: trace every variable reference top-to-bottom; confirm each is defined on an earlier line.
2. **Place fold-dependent operations inside the CV loop.** Preprocessing using per-fold training indices must be nested inside the loop after `KFold.split()`.
3. **Variable rename safety**: search for ALL occurrences before editing; run `pyflakes` to confirm no undefined names remain.
4. **Variable naming contract**: use one name per data artifact throughout the script. Never silently rename between pipeline stages.
5. **Configuration constants at top**: define all hyperparameters in a labeled config block before executable logic.

For detailed patterns, see [references/pipeline-and-code-structure.md](references/pipeline-and-code-structure.md).

## Critical Pipeline Checks

- **Single-batch sanity check (MANDATORY)**: fetch one batch, verify shapes, dtypes, device placement before any training loop.
- **Sparse-to-dense conversion**: convert scipy sparse matrices to dense before tensor construction (`features[idx].toarray()`).
- **Sanitize predictions before metrics**: `np.nan_to_num(val_probs, nan=1.0/n_classes, posinf=1.0, neginf=0.0)` then renormalize.
- **Halt on NaN loss**: if `not torch.isfinite(loss)`, raise immediately.
- **Verify checkpoint exists** before `torch.load()`.
- **Reload best checkpoint** before final inference — verify it loaded correctly.
- **Use deterministic output filenames** — hardcode exact submission path.

## Numerical Stability and Log Loss

**Always use `sklearn.metrics.log_loss`** — do NOT manually implement with NumPy.

**Always clip predicted probabilities**: `np.clip(predictions, 1e-15, 1 - 1e-15)`.

**Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before every optimizer step.

**AMP safety**: pair with `GradScaler`; check `torch.isfinite(loss)` before stepping. Never call `scaler.unscale_()` twice between `scaler.update()`. If NaN persists, disable AMP.

**Fail fast on invalid metrics**: `assert np.isfinite(val_log_loss)` after every computation.

For detailed patterns, see [references/training-stability-and-oom.md](references/training-stability-and-oom.md).

## Overfitting Controls

- Use low learning rates (1e-5 to 2e-5) with 10% linear warmup + cosine decay.
- Prefer short training schedules (3–5 epochs) with early stopping (patience 2–3).
- Apply dropout (0.1–0.3) and weight decay (≥ 0.01).
- Use stratified k-fold cross-validation (3–5 folds).
- Monitor train vs. validation loss gap; stop if diverging.
- Consider progressive unfreezing, layer-wise LR decay, label smoothing (0.05–0.1).
- **Anti-overfitting strategies when plateaued**: MLM augmentation, synonym replacement, focal loss, multi-sample dropout, reduced sequence length.
- **NaN-safe model selection**: initialize `best_model_state = None`; only update when `math.isfinite(val_score)`.

For proven advanced techniques with incremental integration guidance, see [references/anti-overfitting-toolkit.md](references/anti-overfitting-toolkit.md).

## Ensemble and Prediction

- Use stratified k-fold CV and average test predictions across all folds.
- **Checkpoint ensemble**: average top-3 checkpoints weighted by inverse validation log loss.
- Optionally train LightGBM/XGBoost with per-fold engineered features; average probabilities with transformer.
- Optimize ensemble weights on validation log loss via SLSQP — DO NOT default to naive averaging.
- Prefer simple probability averaging over complex stacking.
- Always normalize each row to sum to 1 before writing submission.

For detailed ensemble patterns, see [references/ensemble-strategies.md](references/ensemble-strategies.md).

## Reliability Rules

- **Inspect before accessing internal modules.** Print model structure before accessing layers for freezing or embedding extraction.
- **Isolate non-critical steps** in try-except. A failure in an optional step must NOT crash the core pipeline.
- **Isolate cross-validation folds** — wrap each fold in try-except; on failure, set uniform baseline predictions and continue.
- **Isolate ensemble components** — wrap each model's training in its own try-except; always have a simple fallback baseline.

## Pandas Indexing Rules

- Boolean masks → `loc`, integer positions → `iloc`. Never pass a boolean Series to `iloc`.
- Preserve explicit split indices from splitters. Do NOT reconstruct row membership via `df["col"].isin(text_list)`.

## BatchNorm Singleton-Batch Crash

**DO NOT** use `BatchNorm` without `drop_last=True` on the training DataLoader. BatchNorm crashes on singleton final batches. Prefer `LayerNorm` for engineered-feature branches.

## Device Consistency

**Always move every tensor to the active device** before passing to model or loss. Tensors from NumPy default to CPU. Apply `.to(device)` to all tensors including labels.

## Regex and Feature Safety

- Escape regex metacharacters in pandas `.str.count()`, `.str.replace()`, `.str.contains()`. Use `regex=False` or `re.escape()`.
- **Custom feature extraction**: return 2D arrays `(n_samples, n_features)` — never 1D. Add shape assertions after extraction.
- **Custom architectures**: verify tensor shapes match before element-wise operations. Add `nn.Linear` projection layers to match dimensions before fusion.
- **External library outputs**: explicitly cast to numpy (`convert_to_numpy=True` for Sentence-Transformers).

## External Dependency Management

Proactively download NLTK resources (`punkt`, `punkt_tab`, `stopwords`, `wordnet`) before use. Wrap in try-except with fallback implementations.

See [references/dependency-and-feature-safety.md](references/dependency-and-feature-safety.md).

## Differential Learning Rate Parameter Groups

Every parameter must appear in exactly one group. Track with a `seen` set to prevent overlaps. Verify total grouped count equals total trainable count before constructing optimizer.

## Critical Warnings

- **DO NOT** use cross-entropy for checkpoint selection when the metric is log loss — they can diverge significantly.
- **DO NOT** skip calibration. Uncalibrated deep model predictions inflate log loss.
- **DO NOT** rely on a single model. Ensembles with diverse feature types are more robust.
- **DO NOT** compute features globally before CV splitting — causes leakage.
- **DO NOT** rebuild the optimizer each epoch — instantiate once before the loop.
- **DO NOT** pass unescaped regex metacharacters to pandas string methods.
- **DO NOT** combine multiple untested advanced techniques in a single iteration.
- **DO NOT** assume network access for downloading embeddings at runtime.
- **DO NOT** launch full training without a smoke test.
- **Pseudolabel confidence threshold: > 0.95.**
- **DO NOT** hardcode hardware-specific parameters without runtime detection.

## Pre-Execution Checklist

Before every training launch, confirm:
- [ ] `py_compile` passes with zero errors
- [ ] No undefined variable references after code merging
- [ ] No residual diff markers or instruction leakage
- [ ] Labels are integer-encoded (not raw strings)
- [ ] All fitted objects (scalers, encoders) use `.transform()` on validation, not `.fit_transform()`
- [ ] `X_train.shape[0] == y_train.shape[0]` and `X_train.shape[1] == X_test.shape[1]`
- [ ] No NaN/Inf in saved arrays
- [ ] Smoke test (single forward-backward on 2–4 samples) passes
- [ ] Gradient clipping enabled; NaN guard in training loop
- [ ] Metric safety: `np.isfinite(val_probs).all()` before `log_loss`
- [ ] Gradient checkpointing and accumulation not both enabled

## Key Risks

- **Overfitting** is the dominant risk. Prefer conservative regularization and low learning rates.
- **Complexity kills pipelines.** Each additional custom component multiplies failure surfaces. Keep scripts modular and test components in isolation.

## References

- [Code Validation Patterns](references/code-validation-patterns.md)
- [Resource and Memory Management](references/resource-and-memory-management.md)
- [Model API and Subclassing](references/model-api-and-subclassing.md)
- [Pipeline and Code Structure](references/pipeline-and-code-structure.md)
- [Training Stability and OOM](references/training-stability-and-oom.md)
- [Anti-Overfitting Toolkit](references/anti-overfitting-toolkit.md)
- [Ensemble Strategies](references/ensemble-strategies.md)
- [Dependency and Feature Safety](references/dependency-and-feature-safety.md)
- [Architecture Patterns](references/architecture-patterns.md)
- [Training Recipes](references/training-recipes.md)
- [Progressive Unfreezing](references/progressive-unfreezing.md)
- [Incremental Testing Patterns](references/incremental-testing-patterns.md)