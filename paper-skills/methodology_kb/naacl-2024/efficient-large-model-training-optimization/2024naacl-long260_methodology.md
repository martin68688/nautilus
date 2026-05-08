# Effective Long-Context Scaling of Foundation Models

**Source**: https://aclanthology.org/2024.naacl-long.260/

## [POSITIVE] Continual Pretraining from LLAMA 2
Instead of training from scratch with long sequences, the model is continually pretrained from existing LLAMA 2 checkpoints with additional 400B tokens of long training sequences.

**Delta**: Saves around 40% FLOPs while imposing almost no loss on performance
**Condition**: When building long-context LLMs from existing short-context models; validated at 7B scale.

**Evidence**: "continual pretraining from short context models can save around 40% FLOPs while imposing almost no loss on performance."

## [POSITIVE] Modified RoPE Base Frequency (ABF)
Increasing the base frequency 'b' of RoPE from 10,000 to 500,000 to reduce the rotation angle and decay effect for distant tokens, enabling better long-range attention.

**Delta**: Outperforms Position Interpolation (PI) and XPOS ABF on perplexity and retrieval tasks
**Condition**: When extending context window beyond original training length; critical for models using RoPE positional encoding.

**Evidence**: "ROPE ABF performs the best among all explored variants... ROPE ABF is the only variant that can maintain its performance up to the full 32,768-token context window on the FIRST-SENTENCE-RETRIEVAL task."

## [NEUTRAL] Upsampling Long Texts in Pretrain Data
Adjusting the data source mix ratio to up-weight long text samples during continual pretraining.

**Delta**: No clear and consistent advantage over using mostly short documents
**Condition**: When the pretrain data already contains high-quality data; length distribution alone is not the key factor.

**Evidence**: "even with most of the long texts removed, the model can still obtain most of the performance gain over LLAMA 2... there is no clear and consistent advantage as we greatly increase the long data ratio."

## [POSITIVE] High-Quality Data Mix (LLAMA 2 LONG data mix)
Using a new data mix that combines existing LLAMA 2 datasets with new long text data, emphasizing data quality over length.

**Delta**: Improvements on coding (+3.8 on 7B), math (+1.95 on 7B), MMLU (+2.5 on 7B) over LLAMA 2
**Condition**: When continual pretraining for long-context; data quality is more critical than text length.

**Evidence**: "the improvements of our pretrain data over the one used by LLAMA 2 mostly come from the quality of the data itself, instead of the length distribution difference."

## [POSITIVE] Instruction Tuning with Short Data + Pretrain Data Mix
Finetuning with short instruction data (FT-Short) mixed with pretrain data, calculating language modeling loss on the whole sequence including input prompts.

**Delta**: Significant improvements on long-context tasks (e.g., NarrativeQA F1 from 22.3 to 38.9, Qasper F1 from 23.7 to 38.9)
**Condition**: When instruction tuning long-context models; especially when output lengths are much shorter than input lengths.

**Evidence**: "adding pretrain data (calculating language modeling loss on the whole sequence) can further boost the performance on most datasets... This simple trick makes learning more stable... which gives significant improvements on most of the tested tasks."

## [POSITIVE] Self-Instruct Long Data Generation with Self-Critique
Using LLAMA 2 CHAT to generate QA pairs from long document chunks, followed by a self-critique step to verify answers, creating synthetic long-context instruction data without human annotation.

**Delta**: Outperforms gpt-3.5-turbo-16k on 7 out of 10 ZeroSCROLLS tasks
**Condition**: When human annotation for long-context is expensive or infeasible; particularly effective for QA-style tasks.

**Evidence**: "without using any human annotated long context data, our 70B chat model is able to outperform gpt-3.5-turbo-16k on 7 out of the 10 tasks."

## [POSITIVE] Two-Stage Training Curriculum (Short then Long Sequences)
Starting pretraining with 4,096 token sequences and switching to 32,768 tokens at a later stage (e.g., 20%, 40%, 80% of training).

**Delta**: Saves ~40% FLOPs with comparable performance to training from scratch with long sequences
**Condition**: When computational budget is limited; models can quickly adapt to longer sequences within a few thousand steps.

**Evidence**: "continual pretraining from short context models can save around 40% FLOPs while imposing almost no loss on performance."

## [POSITIVE] Smaller Learning Rate for Larger Models
Using a smaller learning rate (1e-5) for larger models (34B/70B) compared to smaller models (2e-5 for 7B/13B) during continual pretraining.

**Delta**: Achieves monotonically decreasing validation losses
**Condition**: When continually pretraining large models (34B+); prevents training instability.

**Evidence**: "For the larger 34B/70B models, we find it important to set a smaller learning rate (1e-5) to achieve monotonically decreasing validation losses."

## [POSITIVE] FlashAttention for Efficient Long-Sequence Training
Using FlashAttention to reduce GPU memory overhead and speed loss when increasing sequence length.

**Delta**: Only 17% speed loss when increasing sequence length from 4,096 to 16,384 for 70B model
**Condition**: When training with long sequences; enables practical training without quadratic attention becoming a bottleneck.

**Evidence**: "With FlashAttention, there is negligible GPU memory overhead as we increase the sequence length and we observe around 17% speed loss when increasing the sequence length from 4,096 to 16,384 for the 70B model."

## [NEUTRAL] Not Using Sparse Attention
Choosing not to apply sparse attention mechanisms, as the quadratic attention cost is not a bottleneck at the studied model scales.

**Delta**: No performance degradation; attention cost only becomes bottleneck beyond 49,152 tokens for 70B model
**Condition**: For model sizes up to 70B with sequence lengths up to 32,768; sparse attention may be useful for inference key/value cache reduction.

**Evidence**: "the cost of attention matrix calculation and value aggregation only becomes a computation bottleneck when the sequence length exceeds 49,152 (6h) tokens."
