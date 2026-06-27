# Code Validation Patterns for ML Pipeline Scripts

## Table of Contents
1. [Post-Edit Verification](#post-edit-verification)
2. [Definition-Before-Use Checking](#definition-before-use-checking)
3. [Duplicate Block Detection](#duplicate-block-detection)
4. [Cross-Function Dependency Tracing](#cross-function-dependency-tracing)
5. [Recovering from Partial Diff Application](#recovering-from-partial-diff-application)

## Post-Edit Verification

After applying any SEARCH/REPLACE or diff-based edit:
1. **Re-read the file** around the edited region to confirm markers were consumed.
2. **Search for marker strings**: grep for `<<<<<<<`, `=======`, `>>>>>>>`, `SEARCH`, `REPLACE`.
3. **Compile-check**: run `python -m py_compile <script>`.

## Definition-Before-Use Checking

Python executes scripts top-to-bottom. Trace every variable from assignment to usage.

### Common pitfall: Scheduler initialization
```python
# WRONG: train_loader referenced before creation
total_steps = len(train_loader) * num_epochs  # NameError!

# CORRECT: create DataLoader first, then compute steps
train_loader = DataLoader(dataset, batch_size=16)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=total_steps)
```

### pyflakes for undefined-name detection
```bash
python -m pyflakes script.py
```

## Duplicate Block Detection

Before inserting new logic, scan the file for similar patterns. If a duplicate exists, remove the old version or refactor.

## Cross-Function Dependency Tracing

When merging code from multiple sources, verify every DataFrame column access:
```python
# BROKEN: 'avg_sentence_len' created by a different function
def extract_readability_features(texts):
    df['readability_ratio'] = df['avg_sentence_len'] / df['text'].apply(len)  # KeyError!

# FIX: Pass the column as an argument or compute locally
```

## Recovering from Partial Diff Application

1. Read the entire file to identify all contaminated regions.
2. Rewrite the affected section cleanly.
3. Re-validate with post-edit verification steps.
4. Confirm the fix by reading the file one final time.

**Rule**: If the file contains anything that is not valid Python code, remove it before execution.