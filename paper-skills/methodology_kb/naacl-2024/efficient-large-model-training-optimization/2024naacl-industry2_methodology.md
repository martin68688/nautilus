# Lossless Acceleration of Large Language Model via Adaptive N-gram Parallel Decoding

**Source**: https://aclanthology.org/2024.naacl-industry.2/

## [POSITIVE] Adaptive N-gram Parallel Decoding (ANPD)
A two-stage approach that uses an adaptive N-gram module for rapid drafting of multiple tokens, followed by verification by the original LLM. The N-gram module is updated dynamically during generation.

**Delta**: up to 3.67x speedup
**Condition**: Applicable to autoregressive LLMs (e.g., LLaMA, LLaMA-2, ChatGLM3, CodeLLaMA) for text summarization and code generation tasks.

**Evidence**: "In our experiments, models such as LLaMA and its fine-tuned variants have shown speed improvements up to 3.67×, validating the effectiveness of our proposed ANPD."

## [POSITIVE] Token-level N-gram Module
A probabilistic language model that predicts the next token based on the preceding N-1 tokens, using frequency counts from the current context. It includes Initialize, Update, and Query functions.

**Delta**: outperforms baseline (no draft model needed)
**Condition**: Used during the drafting phase; effective when contextual correlations exist between tokens (e.g., within words, phrases, or code patterns).

**Evidence**: "The ANPD enhances efficiency by eliminating the need for a smaller draft deep learning model, leveraging the much lower computational cost N-gram module to accelerate LLM inference."

## [POSITIVE] Multi-Level N-gram (MLN)
An optimal prefix matching approach that initializes N-1 separate n-gram modules (n from 2 to N). During prediction, it queries starting from the largest N and falls back to lower n levels until a match is found.

**Delta**: improves speedup over single-level N-gram
**Condition**: Applicable when larger N values improve prediction accuracy but may fail to match; MLN mitigates this by falling back to smaller n-grams.

**Evidence**: "The Multi-Level N-gram (MLN) approach enhances inferential speed as the parameter N increases. However, beyond N=5, further increments in N yield no significant additional gains."

## [POSITIVE] Parallel Decoding (Drafting + Verification)
The N-gram module generates K draft tokens in a loop, then the LLM verifies all draft tokens in a single forward pass. Accepted tokens are kept; the first rejected token is replaced by the LLM's prediction.

**Delta**: 1.95x to 3.67x speedup across models
**Condition**: Effective when K is between 6 and 8; larger K yields diminishing returns. Verified tokens must match LLM distribution to maintain losslessness.

**Evidence**: "The ANPD algorithm consistently accelerates inference across various models... with a notable increase of 1.95×-3.67× on LLaMA and its fine-tuned derivatives."

## [POSITIVE] Adaptive Update of N-gram Module During Decoding
During decoding, each new token is paired with the previous N-1 tokens to form a tuple, which is used to update the module's probability memory in real time.

**Delta**: improves acceleration over static N-gram
**Condition**: Applicable during the decoding stage; benefits from correlations in LLM-generated content that guide subsequent generation.

**Evidence**: "The experimental results reveal that employing a runtime update strategy enhances the acceleration of the inference process."

## [POSITIVE] Hyperparameter Selection (N=5, K=7)
Based on ablation studies, the paper recommends setting N (n-gram order) to 5 and K (draft length) to 7 for optimal performance.

**Delta**: near-optimal speedup (plateaus beyond these values)
**Condition**: General recommendation for LLaMA-family models on summarization and code tasks; may vary for other architectures.

**Evidence**: "Based on the empirical evidence... a pragmatic choice for N and K can be posited at N=5 and K=7 respectively."

## [POSITIVE] Lossless Verification (Speculative Decoding Style)
The LLM verifies draft tokens by computing probability distributions; if a draft token does not match the LLM's prediction, it is replaced, ensuring the final output is identical to autoregressive decoding.

**Delta**: lossless (no accuracy degradation)
**Condition**: Always applied; ensures output integrity regardless of draft quality.

**Evidence**: "This guarantees that our ANPD method is lossless, maintaining consistency with the original LLM's generated content."

## [POSITIVE] No Retraining or Extra GPU Memory Required
ANPD is a plug-and-play module that does not require model modification, retraining, or additional GPU memory beyond the LLM itself.

**Delta**: reduces resource overhead compared to speculative decoding with draft models
**Condition**: Applicable to any LLM; particularly beneficial when GPU memory is limited.

**Evidence**: "ANPD eliminates the need for retraining or extra GPU memory, making it an efficient and plug-and-play enhancement."

## [NEGATIVE] Increasing N without Multi-Level N-gram
Using a single-level N-gram with larger N values without the fallback mechanism of MLN.

**Delta**: no speedup improvement; may degrade performance
**Condition**: Observed when N > 5 without MLN; leads to sparse matches and reduced drafting efficiency.

**Evidence**: "The results from this setting indicate that merely increasing the N value... does not lead to a faster inference process... primarily because a larger N value results in fewer successful matches during the Query phase."

## [NEUTRAL] Increasing K beyond 8
Setting the draft length K to values larger than 8.

**Delta**: no significant additional speedup
**Condition**: Applicable when K > 8; diminishing returns due to increased risk of rejection and wasted computation.

**Evidence**: "Our findings indicate that increasing K contributes to a greater acceleration effect, however, the acceleration gains plateau when K lies within the range of 6 to 8."
