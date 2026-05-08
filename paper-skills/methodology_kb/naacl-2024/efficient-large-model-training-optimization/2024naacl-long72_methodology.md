# Accurate Knowledge Distillation via n-best Reranking

**Source**: https://aclanthology.org/2024.naacl-long.72/

## [POSITIVE] n-best reranking for distillation
Uses n-best reranking to select pseudo-labels for student model training from top n hypotheses, leveraging diverse models to pick the highest-quality hypotheses.

**Delta**: +1.6 BLEU on validation set (full model), +1.2 BLEU on test set (bitext only), +2.0 BLEU (bitext+mono), +3.4 BLEU (mono only) over baseline
**Condition**: When applied to sequence-level knowledge distillation for NMT, especially with diverse scoring models.

**Evidence**: "Using the full set of 72 models (M), our n-best reranker achieves the BLEU score of 60.4, surpassing the baseline system by 1.6 BLEU point."

## [POSITIVE] Model selection via discriminative weights
Selects a subset of scoring models with highest learned weights from MIRA training to reduce computational cost while maintaining accuracy.

**Delta**: Accuracy drop minimal: 60.4 vs 60.3 BLEU (full vs selected 5 models)
**Condition**: When scaling up distillation to large datasets; reduces number of scoring models from 72 to 5.

**Evidence**: "As shown, the accuracy of the n-best reranker with smaller model count is relatively similar to running with the full set of models."

## [POSITIVE] Transfer set reduction (monolingual only)
Uses only monolingual data as the transfer set for distillation instead of bitext or bitext+monolingual, reducing distillation time by half.

**Delta**: No accuracy drop: 52.2 BLEU (mono only) vs 52.0 BLEU (bitext+mono) on test set
**Condition**: When monolingual data is in-domain and available; applicable for student model training.

**Evidence**: "This result is highly encouraging since we can reduce the distillation time by half without accuracy drop."

## [POSITIVE] Self-training (iterative knowledge distillation)
Iteratively fine-tunes teacher models using pseudo-labels from n-best reranker, then generates new pseudo-labels for the next iteration.

**Delta**: +0.5 BLEU (iter 1), +0.6 BLEU (iter 2), +0.1 BLEU (iter 3) over previous iteration; final +4.0 BLEU over baseline
**Condition**: When applied iteratively; diminishing returns after 2-3 iterations.

**Evidence**: "Our experiments show that self-training the teacher models for one iteration can improve the student model accuracy by 0.5 BLEU points (row 4). Our final model, after three iterations, scores 4.0 BLEU points higher than the baseline model."

## [POSITIVE] Diverse scoring model categories
Includes 8 categories of models: forward TM, backward TM, right-to-left TM, domain-adapted TM, language model, alignment model, MBR utility, and publicly-available pretrained models.

**Delta**: Outperforms individual models; e.g., best single model (FAIR WMT21 Dense) gives 59.6 BLEU, but combined reranker gives 60.4 BLEU
**Condition**: When models are complementary and cover different inductive biases; especially effective when including both in-house and public models.

**Evidence**: "The efficacy of our n-best reranker depends on the diversity and quality of the deployed scoring models."

## [POSITIVE] Using L2R and R2L models for n-best list generation
Generates n-best lists using both left-to-right (L2R) and right-to-left (R2L) translation models to increase diversity and accuracy.

**Delta**: Oracle BLEU score 2-3 points higher than L2R alone
**Condition**: When generating n-best lists for reranking; beam size 8 is optimal.

**Evidence**: "Compared to L2R, the n-best list's oracle score from L2R+R2L is around 2-3 BLEU points higher."

## [POSITIVE] Discriminative training with MIRA
Trains weights for scoring models using Margin Infused Relaxed Algorithm (MIRA) on a tuning set, optimizing BLEU-based structured hinge loss.

**Delta**: Enables effective combination of diverse models; outperforms baseline by 1.6 BLEU
**Condition**: When a tuning set from the same distribution as test sets is available.

**Evidence**: "MIRA seeks to find λ that minimizes the following structured hinge loss..."

## [POSITIVE] Including publicly-available pretrained models
Incorporates models like LASER, mBART, M2M-100, NLLB, BLOOMZ, mT0, and FAIR WMT21 Dense as scoring models.

**Delta**: FAIR WMT21 Dense ranked #1 among scoring models (59.6 BLEU on WMT20)
**Condition**: When such models are available and can be conditioned for translation (e.g., 5-shot prompting).

**Evidence**: "Our last category consists of various publicly-available pretrained models... including a single dense multilingual model from the WMT21 winning team."

## [NEUTRAL] Sequence-level Knowledge Interpolation (KI) baseline
Chooses hypotheses in n-best list with highest BLEU score using original labels as references.

**Delta**: +0.5 BLEU over baseline (bitext only)
**Condition**: Used as a baseline; requires ground truth labels, limiting application to parallel data only.

**Evidence**: "As shown in row 1, the student model trained with the original labels achieved an accuracy of 48.8 BLEU point. Meanwhile, the models trained with pseudo-labels generated through sequence-level KI and KD showed improvements of 0.5 and 0.8 BLEU points respectively."

## [NEUTRAL] Vanilla sequence-level KD baseline
Standard sequence-level knowledge distillation using top-1 hypothesis from teacher model.

**Delta**: +0.8 BLEU over baseline (bitext only); +2.1 BLEU over baseline (bitext+mono)
**Condition**: Used as baseline; outperformed by n-best reranking approach.

**Evidence**: "The models trained with pseudo-labels generated through sequence-level KI and KD showed improvements of 0.5 and 0.8 BLEU points respectively."

## [POSITIVE] Oracle hypothesis analysis
Measures the best possible hypothesis in n-best list to quantify potential improvement.

**Delta**: Oracle BLEU 67.5 vs baseline 58.8 (almost +10 BLEU)
**Condition**: Used for analysis; demonstrates upper bound of reranking potential.

**Evidence**: "Row Oracle shows that the best-scoring hypotheses surpass the top-1 by almost 10 BLEU point, indicating the substantial room for improvement embedded in our n-best reranking approach."

## [NEUTRAL] Anti-Oracle analysis
Measures the worst possible hypothesis in n-best list to highlight risk of poor scoring.

**Delta**: Anti-Oracle BLEU 41.3 vs baseline 58.8 (almost -20 BLEU)
**Condition**: Used for analysis; underscores importance of robust scoring models.

**Evidence**: "Conversely, row Anti-Oracle shows that the gap to the worst-scoring hypotheses is much wider, which is almost 20 BLEU point worse."

## [POSITIVE] Fine-tuning teacher models instead of retraining
Fine-tunes teacher models for one epoch using pseudo-labels instead of retraining from scratch.

**Delta**: Finetune accuracy on par with retrain (59.5 vs 59.5 BLEU for mono only)
**Condition**: When computational cost is a concern; sufficient for iterative self-training.

**Evidence**: "Comparing the Retrain and Finetune columns, we observe that the accuracy of finetuned models is on par with the model trained from scratch."

## [POSITIVE] Using in-domain monolingual data as transfer set
Uses monolingual data that aligns with evaluation domain instead of mixed-domain bitext.

**Delta**: 52.2 BLEU (mono only) vs 50.0 BLEU (bitext only) for n-best reranking
**Condition**: When in-domain monolingual data is available; reduces distillation time by half.

**Evidence**: "In our approach, the monolingual data used seems to align with the domain of the evaluation sets, while in contrast, the parallel data are sourced from a broader range of domains."
