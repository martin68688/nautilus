---
title: Contrastive Decoding Methods Improve Faithfulness Without Additional Training
confidence: HIGH
papers: [2024naacl-short.69, 2024naacl-long.318]
---

# Contrastive Decoding Methods Improve Faithfulness Without Additional Training

Multiple papers demonstrate that contrastive decoding approaches — contrasting output distributions with and without context — can significantly improve model faithfulness and robustness to noisy inputs without requiring any additional training or fine-tuning.

**Context-aware Decoding (CAD)** (2024naacl-short.69) amplifies the difference between output probabilities when a model is used with and without context: softmax[(1+α) logit(y|c,x) - α logit(y|x)]. With α=0.5 for summarization and α=1 for knowledge conflicts, CAD achieves +14.3% factuality metrics for LLaMA-30B on CNN-DM, +21% ROUGE-L, and 2.9x improvement on knowledge conflicts QA. The method works across OPT, GPT-Neo, LLaMA, and FLAN-T5 models of various sizes. Larger models benefit more from CAD on knowledge conflicts, as "larger LMs can have a greater tendency to rely on their prior knowledge instead of reading the contexts."

**Contrastive and Consistency Learning (CCL)** (2024naacl-long.318) uses a two-stage method for noisy-channel SLU: (1) token-based contrastive learning that creates positive pairs between aligned tokens from clean and noisy ASR transcripts using edit distance alignment, pulling aligned tokens together and pushing non-matching tokens apart; (2) consistency learning that reduces distance between latent features of clean and noisy transcripts. CCL achieves +2.59% accuracy on SLURP Noisy0.56. The reference network (trained with CE on clean transcripts) provides better latent features for the inference network to bootstrap from, with no increase in inference time since only the inference network is used at evaluation.

## Papers & Evidence
- `2024naacl-short.69` (CAD): "CAD, without additional training, significantly improves the faithfulness of different LM families, including OPT, GPT, LLaMA and FLANT5 for summarization tasks (e.g., 14.3% gain for LLaMA in factuality metrics)." — Contrastive decoding with and without context.
- `2024naacl-long.318` (CCL): "On the SLURP benchmark dataset, the proposed CCL method improves the intent classification performance by 2.59% under severe ASR errors." — Token-level contrastive learning between clean and noisy inputs.

## Actionable Guidance
For improving faithfulness without training: (1) use CAD with α=0.5 for summarization and α=1 for knowledge conflicts — contrast logits with and without context at each decoding step, (2) use nucleus sampling (p=0.9) on top of the adjusted distribution for summarization, greedy decoding for knowledge conflicts. For robustness to noisy inputs: use token-level contrastive learning (CCL) with edit distance alignment to create positive pairs between clean and noisy representations.

**Condition**: When deploying LLMs for summarization or QA where faithfulness to provided context is critical, or when inputs are noisy (e.g., ASR transcripts).
**Confidence**: HIGH — Two papers with different contrastive approaches both show significant improvements without additional training.
