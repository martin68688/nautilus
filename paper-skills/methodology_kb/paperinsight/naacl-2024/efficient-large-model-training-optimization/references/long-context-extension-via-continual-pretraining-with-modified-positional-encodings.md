---
title: Long-Context Extension via Continual Pretraining with Modified Positional Encodings
confidence: HIGH
papers: [2024naacl-long.260, 2024findings-naacl.191, 2024naacl-long.222]
---

# Long-Context Extension via Continual Pretraining with Modified Positional Encodings

Multiple papers demonstrate that extending the context window of pre-trained LLMs can be achieved efficiently through continual pretraining with modified positional encodings, saving up to 40% FLOPs compared to training from scratch with long sequences.

**Effective Long-Context Scaling** (2024naacl-long.260) continually pretrains from LLaMA 2 checkpoints with 400B additional tokens using modified RoPE base frequency (ABF) increased from 10,000 to 500,000. This saves around 40% FLOPs while imposing almost no loss on performance. A two-stage training curriculum (short 4,096-token sequences first, then 32,768-token sequences) further improves efficiency. Data quality is more critical than text length: "the improvements of our pretrain data over the one used by LLaMA 2 mostly come from the quality of the data itself, instead of the length distribution difference." FlashAttention enables only 17% speed loss when increasing from 4,096 to 16,384 for 70B models.

**Chunk-based Segmented Training** (2024findings-naacl.191) samples 1/α contiguous non-overlapping subsequences of length αLt from a long sequence, preserving original positional information. With α=0.25, chunk achieves 87% performance of training on sequences twice as long at no extra memory footprint. Linear interpolation of absolute positional embeddings (APE) extrapolates to 5× original context, outperforming RoPE and comparable to ALiBi.

**LM-Infinite** (2024naacl-long.222) uses a Λ-shaped attention mask (attending to first n_starting tokens and most recent L_pretrain tokens) with a distance ceiling that bounds effective relative distance to L_pretrain. This zero-shot method (no parameter updates) allows LLMs pre-trained with 2K or 4K-long segments to generalize to up to 200M length inputs while retaining perplexity. Optional top-k middle token attention improves passkey retrieval by +37.2%.

## Papers & Evidence
- `2024naacl-long.260` (Effective Long-Context): "continual pretraining from short context models can save around 40% FLOPs while imposing almost no loss on performance." — RoPE ABF with two-stage curriculum.
- `2024findings-naacl.191` (Segmented Training): "Our results show that interpolation works well until at least 5Lt. This suggests that with linear interpolation APEs generalize better than RoPE and are comparable to ALiBi." — Chunk-based training with α=0.25.
- `2024naacl-long.222` (LM-Infinite): "Without any parameter updates, it allows LLMs pre-trained with 2K or 4K-long segments to generalize to up to 200M length inputs while retaining perplexity." — Λ-shaped attention mask with distance ceiling.

## Actionable Guidance
For extending context windows: (1) start from a pre-trained short-context model and continually pretrain with longer sequences (saves ~40% FLOPs), (2) use modified RoPE base frequency (increase from 10,000 to 500,000) or linear interpolation of APEs, (3) use a two-stage curriculum (short then long sequences), (4) prioritize data quality over text length, (5) for zero-shot extension without training, use Λ-shaped attention masks with distance ceiling (LM-Infinite).

**Condition**: When extending the context window of pre-trained LLMs beyond their original training length (e.g., from 2K to 32K+ tokens).
**Confidence**: HIGH — Three papers with different approaches all demonstrate effective long-context extension with significant computational savings.
