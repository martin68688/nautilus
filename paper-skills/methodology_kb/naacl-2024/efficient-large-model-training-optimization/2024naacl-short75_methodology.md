# Breaking the Language Barrier: Can Direct Inference Outperform Pre-Translation in Multilingual LLM Applications?

**Source**: https://aclanthology.org/2024.naacl-short.75/

## [POSITIVE] Direct Inference vs. Pre-Translation
Comparing direct inference on non-English inputs against translating inputs to English before inference.

**Delta**: PaLM2-L outperforms pre-translation in 94 out of 108 languages (87.04%)
**Condition**: Applicable to PaLM2 models across 108 languages and 6 benchmarks.

**Evidence**: "PaLM2-L consistently outperforms pretranslation in 94 out of 108 languages."

## [POSITIVE] Language-Ratio Metric
Reports the proportion of languages where one method outperforms the other, rather than just average accuracy.

**Delta**: Reveals 74% of languages favor direct inference on BeleBele despite only 1.1% average accuracy difference
**Condition**: Used for granular per-language analysis across all benchmarks.

**Evidence**: "While differences in average accuracy might appear subtle... the Language-Ratio highlights a more decisive advantage."

## [POSITIVE] Evaluation in English for Open-Ended Tasks
Translating both ground truth and direct inference results to English before calculating lexical metrics to eliminate language-specific morphology bias.

**Delta**: Mitigates inconsistencies in F1/ROUGE across languages
**Condition**: Applied to open-ended tasks (XLSum, XQuAD, TyDiQA-GP) for fair cross-language comparison.

**Evidence**: "We propose a complementary evaluation in English... establishing a consistent basis for comparison, eliminating the impact of language-specific variations in the metrics."

## [POSITIVE] Filtering for Attributive QA
Forward and backward translation filtering to ensure ground truth remains a substring of context after translation, reducing bias against pre-translation.

**Delta**: Addresses inherent bias against pre-translation in attributive QA
**Condition**: Applied to XQuAD and TyDiQA-GP datasets.

**Evidence**: "This approach helps address the inherent bias against pretranslation... by accounting for potential discrepancies in lexical alignment introduced by translation."

## [POSITIVE] Low-Resource Language (LRL) Analysis with Lift
Measuring relative improvement (lift) of direct inference over pre-translation averaged across benchmarks for low-resource languages.

**Delta**: Over 85% of LRLs benefit from direct inference; lifts exceed 5% in 63% of languages
**Condition**: Applied to 60 low-resource languages (Joshi taxonomy score ≤ 2).

**Evidence**: "Analysis shows that even in LRLs, the majority of languages (over 85%) benefit from direct inference with PaLM2, with lifts exceeding 5% in 63% of languages."

## [POSITIVE] Including Open-Ended Generative Tasks
Evaluating both discriminative (closed-ended) and generative (open-ended) tasks, unlike prior studies that focused only on discriminative tasks.

**Delta**: Direct inference consistently outperforms pre-translation on XLSum, XQuAD, TyDiQA-GP
**Condition**: Applied to XLSum (summarization), XQuAD and TyDiQA-GP (question answering).

**Evidence**: "The influence of pre-translation on generative capabilities of LLMs, has been largely unexplored."

## [NEGATIVE] Per-Language Analysis for African Languages
Identifying specific languages (especially African) where pre-translation still outperforms direct inference.

**Delta**: Pre-translation shows consistent superiority in 7 languages, 4 of which are African
**Condition**: Applies to low-resource African languages like Lingala, Bambara, Oromo, Tigrinya.

**Evidence**: "Pre-translation does show consistent superiority in 7 languages: Bambara, Cusco-Collao Quechua, Lingala, Oromo, Punjabi, Tigrinya, and Tsonga."

## [NEUTRAL] Re-examining Prior Results with Language-Ratio
Applying Language-Ratio to previously published GPT4 results to reveal a different narrative than average accuracy suggested.

**Delta**: GPT4 shows majority of languages favor direct inference despite average accuracy favoring pre-translation
**Condition**: Applied to re-analysis of GPT4 results from Ahuja et al. (2023).

**Evidence**: "Although average accuracy favors pre-translation, in the majority of evaluated languages better performance is achieved with direct inference."
