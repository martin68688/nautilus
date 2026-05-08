# Fixing Rogue Memorization in Many-to-One Multilingual Translators of Extremely-Low-Resource Languages by Rephrasing Training Samples

**Source**: https://aclanthology.org/2024.naacl-long.253/

## [NEGATIVE] Multilingual Fine-tuning (Many-to-One)
Fine-tuning a pre-trained LLM on parallel data from multiple source languages into a single target language (English).

**Delta**: BLEU: -23% (AL39 vs BL39 for mBART), BLEURT: -8% (AL39 vs BL39 for mBART)
**Condition**: When training many-to-one translators for extremely-low-resource languages with repeated target texts across languages.

**Evidence**: "the many-to-one multilingual systems have a tendency to learn a 'rogue' strategy of storing output strings from the training data in the LLM structure and retrieving them instead of performing actual translations."

## [POSITIVE] Bilingual Fine-tuning
Fine-tuning a pre-trained LLM on parallel data from a single source language to a target language.

**Delta**: Outperforms multilingual models in terms of lower memorization and more distinct outputs (94% distinct outputs vs 15% for AL39).
**Condition**: When training translators for extremely-low-resource languages, especially when data is limited and target texts are repeated.

**Evidence**: "the issue is not present in the bilingual models but slightly visible in the mBART50-TF10 model."

## [NEUTRAL] Tupi-Family Multilingual Fine-tuning (TF10)
Fine-tuning on 10 languages from the same linguistic family (Tupi-Guarani) instead of all 39 languages.

**Delta**: BLEURT: 0.385 (TF10) vs 0.368 (BL10) for mBART; but higher std dev (0.116 vs 0.079) indicating memorization onset.
**Condition**: When using a moderate number of related languages; may improve average scores but introduces memorization risk.

**Evidence**: "The TF10 model seems to perform slightly better than the bilingual models but with a higher standard deviation. As we show in the next section, high standard deviations are, in fact, a symptom that the translation model has started to perform what we call rogue memorization."

## [POSITIVE] Rephrasing Training Samples (Output Side)
Replacing repeated identical target texts with semantically equivalent rephrasings (e.g., using 10 different Bible versions) to avoid exact duplicates in the training set.

**Delta**: BLEURT: +6% (0.408 vs 0.385), BERTScore: +1% (0.887 vs 0.879), BLEU: -15% (10.432 vs 12.325) but std dev reduced by 38% (BLEU), 27% (BLEURT), 20% (BERTScore). Distinct outputs: 99.5% vs 85%.
**Condition**: When training many-to-one multilingual translators where target texts are repeated across languages; effective for reducing memorization without sacrificing translation quality.

**Evidence**: "the model presents very low memorization levels together with improved translation quality and more diversity of outputs."

## [POSITIVE] Using Sentence-Level BLEU with Standard Deviation
Computing average of sentence-level BLEU scores and reporting standard deviation, rather than corpus-level BLEU alone.

**Delta**: High std dev (e.g., 16.218 for TF10 BLEU) signals memorization; low std dev (e.g., 8.554 for BL10) indicates stable translation.
**Condition**: When evaluating machine translation quality in low-resource settings; helps detect rogue memorization hidden by average scores.

**Evidence**: "common symptoms of this problem are high standard deviations of the metrics (which, therefore, should be always reported), bimodal distributions of results, and low numbers of distinct outputs."

## [POSITIVE] Memorization Metric (Minimum Distance to Training Set)
Computing the smallest Euclidean distance between a generated output and all training set target sentences using sentence embeddings (Sentence Transformers).

**Delta**: Mean distance for AL39: 0.14-0.17 (high memorization); for BL39: 0.62-0.66 (low memorization).
**Condition**: When needing to detect soft memorization (not just exact matches) in machine translation outputs, especially for many-to-one models.

**Evidence**: "To more precisely quantify the extent of the memorization by the models, we implemented a metric to quantify memorization, based on computing the smallest distance of a generated output to a sample from the training."

## [NEUTRAL] Using mBART50 vs WMT19 as Base Models
Comparing two pre-trained LLMs: mBART50 (680M params, multilingual) and WMT19 (315M params, German-English).

**Delta**: mBART50 slightly outperforms WMT19 on average (e.g., BLEURT: 0.368 vs 0.346 for BL10) but both exhibit similar memorization patterns.
**Condition**: When choosing a base model for fine-tuning low-resource MT; both are susceptible to memorization in many-to-one setups.

**Evidence**: "Overall, mBART50 seems to perform slightly better than WMT19 but with higher standard deviation, and the three metrics afford similar results comparatively."

## [POSITIVE] Using BLEURT and BERTScore Alongside BLEU
Employing neural-based metrics (BLEURT, BERTScore) in addition to n-gram based BLEU for evaluation.

**Delta**: All three metrics show consistent trends (e.g., AL39 worse than BL39 across all metrics).
**Condition**: When evaluating translation quality; neural metrics help corroborate findings but do not alone detect memorization.

**Evidence**: "We used three metrics to evaluate the results, combining the traditional BLEU score with more recent neural-based metrics which are considered to be more robust and better correlated with human scores."

## [NEUTRAL] Using The Bible as Parallel Corpus (Pseudo-Alignments)
Creating parallel datasets from Bible translations across 39 Indigenous languages using verse-level alignment.

**Delta**: Enables creation of 188,033 parallel sentences across 39 languages.
**Condition**: When parallel data is scarce for extremely-low-resource languages; introduces domain bias and potential pre-training overlap.

**Evidence**: "Our main source of data are the many translations to Indigenous languages of the The Bible. Since it is structured in numbered verses, it is relatively easy to create quasi-parallel datasets."

## [POSITIVE] Using Matthew Chapter as Test Set
Holding out the entire book of Matthew as test set to avoid cross-contamination and data leakage.

**Delta**: Allows fair evaluation without leakage; some similarity with synoptic gospels is considered realistic.
**Condition**: When constructing train/test splits for Bible-based MT datasets; helps isolate memorization from data leakage.

**Evidence**: "To avoid cross-contamination in the decoder and allow us to study memorization issues without data leakage between the training and test sets, we used the Matthew chapter from the New Testament as the source of test set."
