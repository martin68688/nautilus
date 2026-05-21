"""Static shared prompt fragments."""

ROBUSTNESS_GENERALIZATION_STRATEGY = {
    "💡 Recommendation: Robustness & Generalization Strategy": [
        "",
        "**To improve model robustness and generalization on unseen data:**",
        "",
        "✅ **Architecture**: Match model inductive bias to data structure (e.g., CNNs/ViTs for spatial grids, Transformers/RNNs for sequences, GNNs/GCNs for graphs/topology)",
        "✅ **Input Strategy**: Handle variable-length or large-scale inputs via **windowing strategies** or patch-based processing (consider overlap for smoother predictions)",
        "✅ **Regularization**: Consider using Dropout, Batch/Layer Norm, Weight Decay, or Label Smoothing",
        "✅ **Loss Function**: Inspect class distribution and adapt loss accordingly (e.g., weighted loss, FocalLoss, or task-specific objectives)",
        "✅ **Learning Rate**: Consider using adaptive schedules like Cosine Annealing or ReduceLROnPlateau or Warmup with differential rates if needed",
        "✅ **Data Augmentation**: Apply domain-appropriate augmentation based on data modality (e.g., geometric transforms, masking, mixup)",
        "✅ **Validation**: Monitor validation metrics strictly and use early stopping to prevent overfitting",
        "",
        "⚠️ **Note**:",
        "Prioritize capturing the intrinsic structure of the data (Inductive Bias) over simply increasing model size.",
        "",
    ]
}


MODEL_ARCHITECTURE_SAFETY = {
    "🚨 MODEL ARCHITECTURE SAFETY (Prevent Runtime Crashes)": [
        "",
        "⚠️ **Critical**: When modifying a pre-trained model architecture (e.g., adding attention pooling, custom heads, wrapper classes),",
        "you MUST ensure checkpoint loading is compatible. This is the #1 cause of runtime crashes.",
        "",
        "❌ **FATAL PATTERN (will crash at runtime):**",
        "```python",
        "# 1. Define custom model with new layers:",
        "class DebertaWithAttentionPooling(nn.Module):",
        "    def __init__(self):",
        "        self.deberta = AutoModel.from_pretrained(MODEL_NAME)",
        "        self.attention_pool = nn.Linear(hidden_size, 1)  # NEW layer!",
        "        ...",
        "",
        "# 2. Train and save: torch.save(model.state_dict(), path)",
        "",
        "# 3. Later, load checkpoint with strict=True → CRASH!",
        "#    Keys like 'attention_pool.weight' don't exist in old checkpoint",
        "model.load_state_dict(torch.load(path))  # RuntimeError: unexpected key!",
        "```",
        "",
        "✅ **SAFE APPROACHES:**",
        "",
        "**Option A (Recommended): Don't change model class. Use standard AutoModelForSequenceClassification and extract embeddings via outputs.hidden_states.**",
        "This avoids checkpoint incompatibility entirely.",
        "",
        "**Option B: If you must modify architecture, use filtered loading:**",
        "```python",
        "state_dict = torch.load(path, map_location=device)",
        "model_state = model.state_dict()",
        "filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}",
        "model.load_state_dict(filtered, strict=False)",
        "print(f'Loaded {len(filtered)}/{len(model_state)} parameters from checkpoint')",
        "```",
        "",
        "**Option C: If using attention pooling, implement it OUTSIDE the model class as a post-processing step on CLS embeddings instead of adding it as a model layer.**",
        "",
        "⚠️ **AMP Mixed Precision**: When using `torch.cuda.amp.autocast()`, NEVER use values outside [-65504, 65504] in `masked_fill`. Use `-1e4` instead of `-1e9` for attention masking. float16 cannot represent `-1e9`.",
        "",
        "⚠️ **Cross-fold Checkpoint Loading**: In k-fold CV, each fold trains a fresh model. Do NOT load a checkpoint from fold N into fold N+1 unless you handle key mismatches.",
    ]
}


def prompt_leakage_prevention():
    """Data leakage prevention."""
    return {
        "🚨 DATA LEAKAGE PREVENTION": [
            "",
            "⚠️ **Strict Isolation Principle**: Validation/Test data must remain strictly unseen during training.",
            "",
            "✅ **Sequence**: Always **Split Data FIRST**, then apply processing.",
            "✅ **Stateful Transformations**: Fit all Scalers, Encoders, Imputers, and Tokenizers **ONLY on Training data**, then use `.transform()` on Validation/Test.",
            "✅ **Feature Engineering**: Calculate global statistics (e.g., mean, variance, vocabulary) solely from the Training set.",
            "✅ **Target Leakage**: Never use target information (e.g., Target Encoding) from the validation set.",
            "",
            "🔴🔴🔴 **CRITICAL: INDEX_BUG — The #1 Cause of Validation Leakage** 🔴🔴🔴",
            "",
            "**THIS BUG HAS CAUSED 3+ RUNS TO PRODUCE COMPLETELY INVALID RESULTS (log_loss 0.008~0.05 that are FAKE).**",
            "",
            "**The Bug**: After `reset_index(drop=True)`, using `.index.tolist()` to index into the ORIGINAL DataFrame causes label-text misalignment:",
            "",
            "```python",
            "# ❌❌❌ WRONG — THIS IS THE INDEX_BUG THAT CAUSES VALIDATION LEAKAGE:",
            "train_set = train_df.iloc[train_idx].reset_index(drop=True)  # index becomes 0,1,2,...",
            "val_set = train_df.iloc[val_idx].reset_index(drop=True)    # index becomes 0,1,2,...",
            "train_indices = train_set.index.tolist()  # [0,1,2,...] — WRONG! These are NOT the original positions!",
            "val_indices = val_set.index.tolist()      # [0,1,2,...] — WRONG!",
            "train_labels_final = train_labels_orig[train_indices]  # Labels from WRONG rows!",
            "val_labels_final = train_labels_orig[val_indices]      # Labels from WRONG rows!",
            "```",
            "",
            "When `reset_index(drop=True)` resets the index to 0,1,2,..., using those indices on the original",
            "DataFrame's `.values` array selects the FIRST N rows — NOT the split rows. This causes text-label",
            "misalignment, which often manifests as unrealistically low validation loss (0.008-0.05 in author ID tasks).",
            "",
            "```python",
            "# ✅✅✅ CORRECT — Option A: Use split indices directly (numpy indexing):",
            "train_idx, val_idx = next(skf.split(train_df, train_df['author']))",
            "train_texts = train_df['text'].values[train_idx]",
            "train_labels = train_labels_orig[train_idx]  # train_idx are the CORRECT positions",
            "val_texts = train_df['text'].values[val_idx]",
            "val_labels = train_labels_orig[val_idx]",
            "",
            "# ✅✅✅ CORRECT — Option B: Get data directly from sub-DataFrames:",
            "train_set = train_df.iloc[train_idx].reset_index(drop=True)",
            "val_set = train_df.iloc[val_idx].reset_index(drop=True)",
            "train_texts = train_set['text'].values          # From sub-DataFrame, NOT original",
            "train_labels = train_set['author_encoded'].values  # From sub-DataFrame",
            "val_texts = val_set['text'].values",
            "val_labels = val_set['author_encoded'].values",
            "",
            "# ✅✅✅ CORRECT — Option C: No reset_index, use .iloc:",
            "train_set = train_df.iloc[train_idx]  # Keep original index",
            "val_set = train_df.iloc[val_idx]",
            "train_texts = train_set['text'].values",
            "train_labels = train_set['author_encoded'].values",
            "```",
            "",
            "🔴 **SELF-CHECK**: After writing your split code, verify: `assert len(set(train_idx) & set(val_idx)) == 0`",
            "🔴 **SANITY CHECK**: If your validation log_loss < 0.1 for author identification, you almost certainly have this bug.",
            "",
        ]
    }


def prompt_resp_fmt():
    """Response format for plan + code"""
    return {
        "Response format": (
            "Your response should be a brief outline/sketch of your proposed solution in natural language, "
            "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
            "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
        )
    }


def get_internet_clarification(pretrain_model_dir: str = ""):
    """Internet access clarification for improve/debug stages."""
    lines = [
        "**⚠️ IMPORTANT: Internet Access During Code Development**",
        "- The \"no internet access\" restriction mentioned in the task description applies **ONLY to submission evaluation after code generation** (for mle-bench test set).",
        "- **During code development, you CAN and SHOULD use online resources** such as torch.hub.load(), HuggingFace transformers, timm, etc.",
    ]
    if pretrain_model_dir:
        lines.append(
            f"- **Model paths under `{pretrain_model_dir}/` are GUARANTEED to exist and be available** (e.g., DINOv3, Siglip2 etc.). You can directly use them without `Path question`."
        )
    lines.append(
        "- **Do NOT question internet access concerns - all standard ML libraries and pretrained models are available during development."
    )
    return lines
