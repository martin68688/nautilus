# Diverse Perspectives, Divergent Models: Cross-Cultural Evaluation of Depression Detection on Twitter

**Source**: https://aclanthology.org/2024.naacl-short.58/

## [POSITIVE] Cross-cultural evaluation dataset
A custom geo-located Twitter dataset of depressed users from seven countries (Australia, South Africa, Nigeria, Philippines, India, UK, US) was collected and manually verified for genuine depression disclosures.

**Delta**: Revealed significant performance gaps between Global North and Global South (e.g., F1 gap of 0.37 for Logistic Regression on MTL)
**Condition**: When evaluating cross-cultural generalization of depression detection models

**Evidence**: "Our results show that depression detection models do not generalize globally. The models perform worse on Global South users compared to Global North."

## [POSITIVE] Manual verification of depression disclosures
Human raters manually verified each user's depression disclosure using a codebook to distinguish genuine disclosures from non-genuine ones (e.g., sarcasm, transient sadness).

**Delta**: Cohen's Kappa of 0.65 (substantial agreement); identified 267 authentic disclosures from 1,556 reviewed users
**Condition**: When constructing a reliable ground-truth dataset for depression detection

**Evidence**: "We manually verified each user's veracity of depression disclosure with human raters, similar to the process in (Coppersmith et al., 2015)."

## [POSITIVE] Geotagged location verification
Users' country was verified by analyzing their 3200 most recent geotagged posts before disclosure and taking the country where they posted most frequently.

**Delta**: Enabled country-level and Global North/South analysis
**Condition**: When ensuring accurate geographic attribution for cross-cultural analysis

**Evidence**: "To verify that a user was in the country, we looked at their 3200 geotagged posts before disclosure and took the country where the user posted the most."

## [NEUTRAL] Preprocessing pipeline
Removed retweet tokens, username mentions, URLs, numeric values; expanded English contractions; removed disclosure words; excluded users with fewer than 20 posts or non-English posts.

**Delta**: No quantitative delta reported
**Condition**: When standardizing data across multiple datasets

**Evidence**: "We applied the same preprocessing pipeline across the datasets (including the benchmarks) for consistency, following recommendations from (Harrigian et al., 2020)."

## [NEGATIVE] Logistic Regression with TF-IDF
A statistical baseline model using term frequency-inverse document frequency features with grid-searched hyperparameters (L2 penalty, lbfgs solver, 7000 features).

**Delta**: F1 scores as low as 0.19 (India) and 0.23 (Nigeria) on cross-cultural data
**Condition**: When generalizing to non-Western, Global South populations

**Evidence**: "All models struggled to correctly identify users from Nigeria and India."

## [POSITIVE] MentalLongformer pretrained language model
A domain-specific pretrained language model with extended sequence modeling capacity (4096 tokens), fine-tuned for depression detection.

**Delta**: Outperforms Logistic Regression: e.g., F1 of 0.72 (UK) vs 0.69 (Logistic Regression) on MTL; better recall on Global North (0.90 vs 0.76)
**Condition**: When aiming for best overall generalization across cultures

**Evidence**: "Pre-trained language models achieve the best generalization compared to Logistic Regression, though still show significant gaps in performance on depressed and non-Western users."

## [NEGATIVE] Training on benchmark datasets (CLPsych and MTL)
Models were trained on two popular benchmark depression detection datasets (CLPsych 2015 Shared Task and Multi-Task Learning dataset) and evaluated on cross-cultural data.

**Delta**: F1 gap of 0.30-0.37 between Global North and Global South
**Condition**: When applying models to populations not represented in training data

**Evidence**: "Models trained on broad Twitter benchmarks do not generalize well to the cross-cultural data."

## [POSITIVE] Global North vs. Global South evaluation
Countries were categorized into Global North (US, UK, Australia) and Global South (India, Nigeria, Philippines, South Africa) based on UN classification for subgroup analysis.

**Delta**: All four models had superior F1 and recall for Global North; e.g., MentalLongformer on MTL: 0.68 F1 (Global North) vs 0.56 (Global South)
**Condition**: When auditing model fairness across socioeconomic development levels

**Evidence**: "There is an expected drop in performance between the baseline model and the Global North and the Global South evaluation datasets... all four models have a superior F1 and recall in identifying the Global North evaluation users."

## [POSITIVE] Country-level analysis
Performance was evaluated separately for each of the seven countries to identify specific gaps.

**Delta**: Significant difference (p-value: 0.001) in accurately identifying depressed users among countries; Nigeria and India had worst performance (F1 as low as 0.12-0.23)
**Condition**: When identifying which specific populations are most underserved

**Evidence**: "There is a significant difference (p-value: 0.001) in accurately identifying depressed users among various countries."

## [POSITIVE] Qualitative error analysis
Examined word distributions and code-mixing patterns in misclassified users from Nigeria and Philippines to understand performance disparities.

**Delta**: Identified code-mixing as a source of disparities
**Condition**: When diagnosing causes of model failures in cross-cultural settings

**Evidence**: "We note that many of the users from Nigeria and India have discrepancies in how they communicate in general... Such differences in language usage might account for the subpar performance."

## [NEGATIVE] Code-mixing detection limitation
Twitter's language detector failed to identify code-mixed posts (mixing English with other languages), leading to some non-English content in the supposedly English-only filtered dataset.

**Delta**: Contributed to performance disparities for Nigeria, India, and Philippines
**Condition**: When relying on automated language detection for multilingual populations

**Evidence**: "Although we filtered for English-only content using Twitter language detection, it missed some posts, resulting in code-mixed content for some users. This could potentially be the source of some of the disparities identified."

## [POSITIVE] Recommendation: construct datasets with more geographical examples
Suggestion to include more geographically diverse data in training datasets and infer geo-location when metadata is missing.

**Delta**: Hypothesized to alleviate disparities (based on prior work showing more data helps racial disparities)
**Condition**: When building future depression detection datasets and models

**Evidence**: "We hypothesize that mental health detection from social media suffers from small datasets. Existing benchmark datasets lack the location meta-data of users..."

## [POSITIVE] Recommendation: investigate cross-cultural detection capabilities
Call for fine-grained subgroup analysis reporting and transparent reporting of model performance across cultural groups.

**Delta**: No quantitative delta reported
**Condition**: When reporting new models and algorithms for mental health detection

**Evidence**: "Fine-grained subgroup analysis reporting leads towards building more inclusive and transparent models."
