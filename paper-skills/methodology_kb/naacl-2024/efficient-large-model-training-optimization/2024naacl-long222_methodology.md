# LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models

**Source**: https://aclanthology.org/2024.naacl-long.222/

## [POSITIVE] Λ-shaped attention mask
An attention mask that allows each token to attend to the first n_starting tokens and the most recent L_pretrain tokens, ignoring the rest. This limits the number of attended tokens while preserving access to initial tokens.

**Delta**: outperforms baselines; retains perplexity on sequences up to 200M tokens
**Condition**: Applicable to any off-the-shelf LLM using relative positional encoding, without parameter updates. Effective for sequences longer than training length.

**Evidence**: "LM-Infinite consists of two major components... (1) a Λ-shaped attention mask... This resolves Factors 2 and 3 in §3 by both limiting the number of tokens under attention and ensuring the starting few tokens are attended."

## [POSITIVE] Distance ceiling
Bounding the effective relative distance between tokens to the maximum distance seen during training (L_pretrain), preventing attention logit explosion from unseen distances.

**Delta**: prevents attention logit explosion; enables stable perplexity
**Condition**: Applied to all attention computations; particularly affects starting tokens when attended by later tokens.

**Evidence**: "LM-Infinite further bounds the 'effective distance' to L_pretrain... This addresses Factor 1 in §3 by bounding the distance value in attention calculation."

## [POSITIVE] Optional top-k middle token attention
Optionally attending to k tokens in the middle with the largest attention logits, selected independently per attention head in layers higher than the h-th layer, with a fixed attention distance.

**Delta**: +37.2% on Passkey Retrieval; +1.2% on Qasper
**Condition**: Useful for downstream tasks where middle tokens contain important information; not necessary for language modeling perplexity or generation quality.

**Evidence**: "LM-Infinite can optionally attend to k tokens in the middle with the largest attention logits... This is particularly useful in downstream tasks where information in the middle tokens matters (§5.3)."

## [POSITIVE] Zero-shot application without parameter updates
LM-Infinite is applied to pre-trained LLMs without any fine-tuning or parameter updates, using only modifications to the attention mechanism.

**Delta**: outperforms truncation baselines; achieves comparable performance to fine-tuned MPT-7B-Storywriter
**Condition**: Applicable to any LLM with relative positional encoding; no training required.

**Evidence**: "Without any parameter updates, it allows LLMs pre-trained with 2K or 4K-long segments to generalize to up to 200M length inputs while retaining perplexity."

## [NEGATIVE] Sliding-window attention pattern (baseline)
A baseline method where each token only attends to the most recent tokens within a predefined window, discarding initial tokens.

**Delta**: second worst NLL values; significant performance and fluency degradation
**Condition**: When used alone without LM-Infinite components; fails due to Factor 3 (starting tokens occupy distinct feature space).

**Evidence**: "The 'window' curve shows a baseline with the sliding-window attention pattern... It produces the second worst NLL values, which indicates a significant performance and fluency degradation."

## [NEGATIVE] Relative positional encodings (RoPE, AliBi) alone
Using relative positional encodings without additional modifications to handle long contexts.

**Delta**: perplexity explosion beyond training length; NaN outputs for Llama-2
**Condition**: When sequence length exceeds training length; fails due to Factors 1, 2, and 3.

**Evidence**: "Despite some promising empirical evidence, length generalization failures are still widely observed when directly applied to large language models."

## [NEGATIVE] Truncation baseline
Dropping excessive tokens when context exceeds pre-training length, keeping only the most recent tokens.

**Delta**: worse quality-efficiency tradeoff; lower downstream performance
**Condition**: When context length exceeds model's pre-training length; loses information from earlier parts of the sequence.

**Evidence**: "LM-Infinite achieves a substantially better quality-efficiency tradeoff. With similar computation, LM-Infinite outperforms the baseline by about 5 BLEU."
