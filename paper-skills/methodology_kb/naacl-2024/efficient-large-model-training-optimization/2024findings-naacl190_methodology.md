# Retrieving Examples from Memory for Retrieval Augmented Neural Machine Translation: A Systematic Comparison

**Source**: https://aclanthology.org/2024.findings-naacl.190/

## [POSITIVE] Smoothed Longest Common Subsequence (δ-LCS)
A ranking function that modifies the edit distance by setting a small non-zero deletion cost δ, balancing coverage, relevance, and length of retrieved examples.

**Delta**: +0.3 BLEU (ICL), +0.3/+0.1/+0.5 BLEU (TM3-LevT) on test-0.4/test-0.6/test-de
**Condition**: Applicable to edit-based (TM3-LevT) and in-context learning (ICL) architectures; especially beneficial when coupled with filters at training and inference time.

**Evidence**: "By design, δ-LCS retrieves examples having a higher coverage of the source than LED, which turns into higher copy rates for all architectures. For ICL and TM3-LevT, it yields similar BLEU gain (ICL: +0.3/+0.1/+0.4; TM3-LevT: +0.3/+0.1/+0.5 on resp. test-0.4, test-0.6, test-de)."

## [POSITIVE] Contrastive Ranking (MMR-style diversity)
An iterative ranking algorithm that penalizes candidate examples that are too similar to already selected examples or that cover already covered words, inspired by Maximal Marginal Relevance.

**Delta**: Best results for ICL (e.g., 48.3/37.5 BLEU/COMET for test-0.4 with LED_c 3-shot)
**Condition**: Most beneficial when matches are poor (test-0.4); can be detrimental when a high-quality match already exists. Strength parameter α should be adapted to retrieval scores.

**Evidence**: "We observe that using a contrastive ranker is mostly beneficial for medium-scoring similar examples (test-0.4), regardless of the architecture. It can even be detrimental when at least one high-matching example is found."

## [POSITIVE] Filter-Free Inference (removing NGM/BM25 filter at test time)
Removing the filtering step (e.g., n-gram matching or BM25) during inference, allowing the ranker to consider all candidates in the translation memory.

**Delta**: Improves scores for TM3-LevT; no effect for NFA
**Condition**: Applicable to edit-based (TM3-LevT) architecture at inference time; trade-off between latency and translation scores.

**Evidence**: "We turn off filtering for test samples during inference. This simplification of the pipeline improves the scores for TM3-LevT, while there is no effect for NFA."

## [POSITIVE] In-Domain Selection (domain filtering)
Restricting retrieval to examples from the same domain as the source sentence, rather than using all domains or out-of-domain examples.

**Delta**: Out-of-domain selection yields dramatic losses: -15.4/-23.8 BLEU; all-domain slightly worse than in-domain
**Condition**: Applicable to all architectures (NFA, TM3-LevT, ICL); especially important for small domains like Ubuntu.

**Evidence**: "These results confirm the benefit of retrieving 'in-domain', even for small domains: not only does it greatly speed up retrieval, but it also yields better examples and, ultimately, higher translation scores."

## [POSITIVE] Increasing Number of Retrieved Examples (k from 1 to 3)
Varying the number of translation memory examples retrieved and used as context, from 1 to 3.

**Delta**: 3-shots clearly outperforms 1-shot for ICL; similar gain for TM3-LevT
**Condition**: Applicable to ICL and TM3-LevT; for NFA, using more examples does not compete with using only the one-best match.

**Evidence**: "Overall, we observe a gain (BLEU/COMET) when k increases. For ICL, this is already clear from the results in Table 6 where 3-shots clearly outperforms 1-shot."

## [NEUTRAL] Neural Fuzzy Augmentation (NFA) - Autoregressive concatenation
Concatenating the target side of retrieved examples to the source sentence in an autoregressive encoder-decoder model.

**Delta**: Very consistent scores across retrieval settings; hardly vary across domains
**Condition**: Applicable to autoregressive NFA architecture; insensitive to changes in retrieval strategy at training or inference time.

**Evidence**: "Results in Table 4 are very consistent and hardly vary across domains and language pairs. This is a first important result that somehow consolidates observations already performed for this model, which seems to be robust with respect to variations in the retrieval strategy."

## [POSITIVE] BM25 as Filter/Ranker
Using BM25 (a probabilistic retrieval function based on term frequency-inverse document frequency) to filter or rank candidate examples from the translation memory.

**Delta**: Best overall training pipeline for TM3-LevT (BM25+LED); comparable to other rankers for ICL
**Condition**: Effective for edit-based (TM3-LevT) and ICL architectures; requires inverted index for efficiency; common terms removed to avoid noise.

**Evidence**: "We evaluate our best overall training pipeline (BM25+LED) on a filter-free setup with varying ED costs."

## [POSITIVE] Edit Distance (LED) as Ranker
Using Levenshtein edit distance (word-level) to rank candidate examples by their similarity to the source query.

**Delta**: Baseline ranker; yields highest relevance scores
**Condition**: Standard ranker for all architectures; effective but may discard long examples with good substring matches.

**Evidence**: "LED yields the highest relevance, while δ-LCS and contrastive ranking yield a higher coverage."

## [NEUTRAL] Copy Rate Analysis
Measuring how much the translation model copies subparts of retrieved examples into its output, using multi-reference sentence BLEU between target examples and output.

**Delta**: Correlated with higher BLEU for TM3-LevT and ICL; not for NFA
**Condition**: Diagnostic metric; useful for understanding model behavior but not a direct technique to improve performance.

**Evidence**: "We find that copy rate is correlated with higher BLEU scores for TM3-LevT and ICL; in contrast NFA fails to produce higher BLEU scores with increasing copy rate."

## [NEUTRAL] Linear Model for BLEU Prediction from Retrieval Metrics
Using a linear regression model to predict BLEU scores based on coverage, relevance, and length of retrieved example sets.

**Delta**: Average squared residuals of 0.2 BLEU for TM3-LevT vs 1.2 for constant model
**Condition**: Analytical tool; shows that coverage and relevance are important for TM3-LevT but ICL is more robust to changes in retrieved examples.

**Evidence**: "Coverage, relevance, and length have respective coefficients of 0.13, 0.09, and 0.03. This highlights the importance of coverage and relevance measures in the explanation of BLEU performance."
