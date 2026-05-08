# Extending Input Contexts of Language Models through Training on Segmented Sequences

**Source**: https://aclanthology.org/2024.findings-naacl.191/

## [POSITIVE] Linear Interpolation of Absolute Positional Embeddings
Extends the embedding matrix from training length Lt to extended length Le by linearly interpolating existing embeddings, with modulo-based mapping and padding for new positions.

**Delta**: outperforms RoPE, comparable to ALiBi; extrapolates to 5× original context
**Condition**: When extending APE-based models to longer sequences without additional training.

**Evidence**: "Our results show that interpolation works well until at least 5Lt. This suggests that with linear interpolation APEs generalize better than RoPE and are comparable to ALiBi."

## [POSITIVE] Chunk-based Segmented Training (chunk-α)
Samples 1/α contiguous non-overlapping subsequences of length αLt from a long sequence, preserving original positional information, and concatenates them for training.

**Delta**: outperforms RandomPos; achieves 87% performance of training on sequences twice as long at no extra memory footprint
**Condition**: When extending input context of pretrained models (APE, RoPE, ALiBi) with limited compute; α=0.25 recommended.

**Evidence**: "Overall, chunk performs better than prefix on both models... chunk outperforms RandomPos indicating the inclusion of distant context valuable to length extrapolation."

## [NEGATIVE] Prefix-based Segmented Training (prefix-α)
Creates a short sequence by randomly sampling a prefix of (1-α)Lt tokens from positions before a random index i, and a contiguous suffix of αLt tokens starting at i; loss computed only over the suffix.

**Delta**: worse than chunk; fails to improve RoPE at 4× extension
**Condition**: When extending input context, especially for RoPE-based models; less effective than chunk.

**Evidence**: "Overall, chunk performs better than prefix on both models, prefix fails to improve extrapolation when extending RoPE to sequences 4× in length."

## [NEUTRAL] Domain Adaptation (DA) on Original Length Lt
One full epoch of continual pre-training on sequences of the original input length Lt to adapt the model to the target domain (arXiv).

**Delta**: improves perplexity but less than chunk for most extensions
**Condition**: When comparing gains from domain adaptation vs. segmented training; baseline for ablation.

**Evidence**: "For models that extrapolate well (ALiBi and APE), further domain adaptation also improves the extrapolation ability however the gains are less than our segmented training."

## [NEGATIVE] RandomPos (Randomized Positional IDs)
Randomizes positional IDs of sequences of length Lt, selecting positions from [0, Le-1] while maintaining causal ordering, exposing model to extrapolated pairwise distances.

**Delta**: worse than chunk-0.25 on all models and extensions
**Condition**: When comparing with chunk-based methods; less effective due to lack of distant content.

**Evidence**: "In all cases, chunk outperforms RandomPos indicating the inclusion of distant context valuable to length extrapolation."

## [POSITIVE] Full Sequence Training (full)
Continual pre-training on the full extended sequence length Le without segmentation, used as an upper-bound comparison.

**Delta**: lowest perplexity in most cases, but chunk is competitive
**Condition**: When compute is not constrained; serves as a performance ceiling for segmented methods.

**Evidence**: "While the full approach has the lowest perplexity in most cases the relative loss in performance for chunk is low."
