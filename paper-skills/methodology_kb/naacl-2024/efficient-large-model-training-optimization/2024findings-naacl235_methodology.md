# CodecLM: Aligning Language Models with Tailored Synthetic Data

**Source**: https://aclanthology.org/2024.findings-naacl.235/

## [POSITIVE] Instruction Metadata Encoding
Encoding seed instructions into metadata (use case and skills) to capture the target instruction distribution using a strong LLM as an encoder.

**Delta**: outperforms baselines; e.g., +7.8 CRR on Evol-Instruct with LLaMA-7B
**Condition**: When seed instructions are available from the target task distribution.

**Evidence**: "CodecLM outperforms comparing methods consistently on all benchmarks... The consistently superior performance of CodecLM highlights its generalizability to different downstream instruction distributions and target LLMs."

## [POSITIVE] Self-Rubrics
Using a strong LLM to generate metadata-specific rubrics and actions to tailor instruction complexity, rather than using generic human-crafted rules.

**Delta**: +2.3 CRR over metadata-only baseline (from 75.23 to 77.52 on Evol-Instruct with LLaMA-7B)
**Condition**: When instructions require domain-specific complexity adjustments (e.g., business plan vs. creative writing).

**Evidence**: "Self-Rubrics adaptively generates instruction improving actions based on different metadata, resulting in better tailored instructions for the target LLM."

## [POSITIVE] Contrastive Filtering
Selecting instruction-response pairs by comparing response quality between the target LLM and a strong LLM, keeping pairs with a large quality gap for training.

**Delta**: +2.3 CRR over Self-Rubrics-only baseline (from 77.52 to 79.82 on Evol-Instruct with LLaMA-7B)
**Condition**: When a strong LLM is available to score responses and identify challenging instructions for the target LLM.

**Evidence**: "Further improvements from Contrastive Filtering demonstrate that selected data are indeed more effective for alignment."

## [POSITIVE] Metadata Matching Proportion
The degree to which the generated metadata matches the true test instruction distribution.

**Delta**: Performance improves as metadata matching proportion increases; close to full performance at ≥60% matching
**Condition**: When metadata is extracted from seed instructions or provided by users; robust to moderate mismatch.

**Evidence**: "We did observe the trend that the better instruction metadata captures the underlying distribution, the better performance the target LLM can achieve."

## [POSITIVE] Iterative Self-Rubrics with Contrastive Filtering
Running Self-Rubrics iteratively (up to 4 iterations) and using Contrastive Filtering to send less effective instructions back for further improvement.

**Delta**: More than 70% of data from first iteration; later iterations contribute high-quality complex instructions despite smaller quantity
**Condition**: When iterative refinement is feasible and a strong LLM is available for scoring.

**Evidence**: "High-quality and more complex instructions indeed contribute to the final performance despite less in quantity."

## [POSITIVE] Using Strong LLM as Scorer for Contrastive Filtering
Reusing the strong LLM (e.g., Gemini-Pro) as the scoring function to evaluate response quality and compute the quality gap.

**Delta**: Works quite well in practice; enables effective filtering
**Condition**: When a strong LLM is available and can be prompted for pairwise evaluation.

**Evidence**: "We observe using fs for scoring works quite well in practice, so we prioritize this option for simplicity."

## [POSITIVE] Metadata Augmentation via Mix-and-Match
Augmenting metadata by randomly mixing use cases and skills from different seed instructions to increase coverage.

**Delta**: Enables generation of diverse instructions aligned with target distribution
**Condition**: When seed instructions are limited; manual sanity check recommended to exclude unreasonable pairs.

**Evidence**: "We augment the metadata to 200 by mix-and-matching use cases and skills from different instructions."

## [POSITIVE] Random Action Sampling in Self-Rubrics
Randomly sampling one action from multiple generated actions for each instruction improvement step.

**Delta**: Enables controlled complexity and prevents confusion
**Condition**: When multiple improvement actions are generated per metadata.

**Evidence**: "This design choice not only enables controlled complexity, but also prevents potential confusion between different actions for the LLM."

## [NEUTRAL] Position Bias Mitigation in Evaluation
Averaging scores by exchanging the positions of two responses in pairwise evaluation to mitigate position bias.

**Delta**: Mitigates position bias in LLM-based evaluation
**Condition**: When using LLM-based pairwise evaluators for comparing model responses.

**Evidence**: "To mitigate potential position bias, we average the scores obtained by exchanging the positions of two responses."

## [NEUTRAL] Using Vicuna Pairwise Evaluator (ChatGPT-based)
Using ChatGPT as the LLM judge for final evaluation of instruction-following quality.

**Delta**: Consistent with GPT-4 based evaluation (see Appendix A.6)
**Condition**: When automatic evaluation is needed; shown to be consistent with human evaluation and GPT-4.

**Evidence**: "We adopt widely-used Vicuna pairwise evaluator based on ChatGPT to compare the response quality from two LLMs for its accessibility in price and efficiency."

## [NEUTRAL] Capacity Recovery Ratio (CRR) Metric
Metric defined as (wins + ties) / total comparisons to measure how much target LLM recovers capacity from the strong LLM.

**Delta**: Faithfully reflects model performance
**Condition**: When comparing instruction-tuned models against a strong reference LLM.

**Evidence**: "To demonstrate CRR faithfully reflects model performance, we show the exact number of wins, ties and losses in Appendix A.5 on Evol-Instruct."
