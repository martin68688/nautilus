# CELI: Simple yet Effective Approach to Enhance Out-of-Domain Generalization of Cross-Encoders.

**Source**: https://aclanthology.org/2024.naacl-short.16/

## [POSITIVE] CELI (Cross-Encoder with Late Interaction)
Incorporates a late interaction layer into cross-encoder models, computing token-level similarity between query and document tokens in addition to the standard [CLS] token score.

**Delta**: +5% on BEIR average nDCG@10 (0.467 to 0.491)
**Condition**: Applied to cross-encoder reranking; consistent across different model sizes and first-stage retrievers.

**Evidence**: "This simple method brings 5% improvement on BEIR without compromising in-domain effectiveness or search latency."

## [POSITIVE] Mean-Pooling baseline
Uses the mean representation of all tokens instead of just the [CLS] token for computing relevance score.

**Delta**: Improves BEIR average nDCG@10 from 0.467 to 0.481 (MiniLM backbone)
**Condition**: Used as a baseline to isolate effect of token information vs. interaction; applied to cross-encoder reranking.

**Evidence**: "Mean-Pooling achieves 0.481 on BEIR vs monoBERT's 0.467 (Table 2)."

## [POSITIVE] Summing monoBERT score and late interaction score at inference
At inference, the final relevance score is the sum of the monoBERT [CLS] score and the late interaction token-level score.

**Delta**: Outperforms monoBERT and Mean-Pooling baselines
**Condition**: Inference stage for CELI; authors note weighting terms gave only marginal gains, so simple sum is used.

**Evidence**: "We sum the two scores as the final relevance score, i.e., s_final = s_m + s_l."

## [NEUTRAL] Projecting token representations to lower dimension (D_tok)
Token representations are projected to a lower dimension D_tok (e.g., 32, 128, 384) before computing late interaction.

**Delta**: Negligible impact on BEIR nDCG@10 (0.489 to 0.491 across dimensions 1 to 384)
**Condition**: Applied to CELI token projection; dimension choice has little effect on OOD effectiveness.

**Evidence**: "Using dim=1 already obtains 0.4890 on average BEIR... increasing the dimension to dim=32 and onwards only provides marginal improvement."

## [POSITIVE] Training with LCE loss and 7 negative samples
Cross-encoders are trained using LambdaRank-based loss (LCE) with 7 negative samples per query.

**Delta**: Standard training setup; enables effective reranking
**Condition**: Training configuration for all cross-encoder models in the paper.

**Evidence**: "Cross-encoders are trained on LCE loss (Gao et al., 2021b; Pradeep et al., 2022) with 7 negative samples."

## [POSITIVE] Using larger backbone models (ELECTRAlarge vs MiniLM)
Scaling model size from MiniLM (33M) to ELECTRAlarge (340M) parameters.

**Delta**: monoBERT BEIR nDCG@10 improves from 0.467 (MiniLM) to 0.507 (ELECTRAlarge); CELI from 0.491 to 0.524
**Condition**: Applies to all cross-encoder variants; larger models show better OOD generalization.

**Evidence**: "We observe higher average scores on BEIR as the model size increases... the improvement brought by token information is consistent across the backbones."

## [POSITIVE] Reranking top-1k candidates from various first-stage retrievers
Evaluating CELI by reranking top-1000 results from sparse, single-vector dense, and multi-vector dense retrievers.

**Delta**: Consistent improvement of 0.02–0.03 nDCG@10 across all retriever types
**Condition**: Evaluation on BEIR; applies to all retriever categories (BM25, uniCOIL, SPLADE, DPR, ANCE, TCT-ColBERT, TAS-B, ColBERT v2).

**Evidence**: "Late interaction consistently improves the OOD capacity when using retrievers of different natures, bringing a similar degree of improvement of 0.02–0.03."

## [POSITIVE] Query length analysis
Analyzing per-query improvement as a function of query length (number of tokens).

**Delta**: Higher improvement on longer queries (e.g., Quora and HotpotQA datasets)
**Condition**: Observed on BEIR datasets; improvement correlates positively with query length.

**Evidence**: "On both datasets, we observe a clear tendency that the late interaction brings higher improvement on longer queries."
