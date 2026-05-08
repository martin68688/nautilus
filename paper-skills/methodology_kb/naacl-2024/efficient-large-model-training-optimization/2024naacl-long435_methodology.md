# Explaining Text Similarity in Transformer Models

**Source**: https://aclanthology.org/2024.naacl-long.435/

## [POSITIVE] BiLRP (Bilinear Layer-wise Relevance Propagation)
A second-order explanation method that computes relevance scores for interactions between pairs of input features (e.g., tokens) in bilinear similarity models, extending LRP to handle pairwise interactions.

**Delta**: ACS 0.81 vs 0.62 (HxP) and 0.67 (embeddings); lowest AUPC across models and datasets
**Condition**: When explaining bilinear similarity models (e.g., dot-product similarity) on text pairs; validated on synthetic noun-matching task and real-world STS datasets.

**Evidence**: "BiLRP is able to select the relevant tokens in comparison to the other baselines with higher accuracy, which is supported by the highest average cosine similarity (ACS) between true interactions and BiLRP of 0.81 in comparison to 0.62 for H × P, and 0.67 for the embedding baseline."

## [POSITIVE] LRP propagation rules for Transformers (Attention Head, Layer Norm, GeLU)
Modified gradient propagation rules for Transformer components: detaching attention scores in QKV layers, treating layer normalization denominator as constant, and attributing GeLU activations proportionally to preserve relevance conservation.

**Delta**: Restores relevance conservation; enables BiLRP to outperform HxP and embedding baselines
**Condition**: When computing LRP-based explanations for Transformer models (BERT, mBERT, SBERT, SGPT) to ensure faithful relevance propagation.

**Evidence**: "Implementing these rules to compute second-order explanations (see Section 3.2) reconstitutes relevance conservation in BiLRP in comparison to H × P as shown in Figure 6 in Appendix C."

## [POSITIVE] Mean Pooling over CLS Pooling
Averaging all token embeddings (instead of using only the CLS token) to obtain a fixed-size sentence representation for similarity computation.

**Delta**: Spearman ρ: BERT + CLS = 20.3 vs BERT + Mean Pooling = 47.3 on STSb; 42.4 vs 58.2 on SICK
**Condition**: When building similarity models from pretrained Transformers without fine-tuning on similarity tasks.

**Evidence**: "The standard BERT similarity model, when used with CLS or mean pooling methods, fails to effectively capture semantic proximity. The Spearman correlation ρ × 100 for the CLS-Pooling is 20.3, whereas a significantly improved correlation is achieved when using mean pooling across all encoded token representations for STSb and SICK data."

## [POSITIVE] Fine-tuning on Semantic Similarity (SBERT, SGPT)
Training Transformer models (e.g., SBERT, SGPT) on semantic similarity tasks using Siamese or contrastive objectives to produce better sentence embeddings.

**Delta**: SBERT: ρ=84.7 (STSb), 78.4 (SICK), 66.7 (BIOSSES); SGPT: ρ=76.9 (STSb), 73.4 (SICK), 69.6 (BIOSSES) — significantly outperforming non-finetuned models
**Condition**: When high-quality semantic similarity predictions are required; models are finetuned on STS data.

**Evidence**: "Similarity is best predicted by SGPT and SBERT with scores ranging from 66.7 to 84.7, highlighting overall the considerable impact of model selection, pooling strategy and dataset on task performance."

## [POSITIVE] Corpus-level POS Interaction Analysis
Aggregating BiLRP explanations across an entire dataset by mapping tokens to POS tags and summing relevance scores per POS pair to identify which grammatical interactions drive similarity.

**Delta**: Reveals that similarity is driven by a small subset of interactions (e.g., NOUN-NOUN, NOUN-VERB); identifies token-matching shortcuts in non-finetuned models
**Condition**: When analyzing model strategies at corpus scale; applicable to any similarity model with POS-tagged data.

**Evidence**: "Our corpus-level POS analysis has indicated that semantic similarity can be approached by a small subset of interactions, most importantly interactions between nouns or proper nouns, verbs, and nouns and verbs, which promotes the notion of similarity as a ‘bag of interactions’."

## [POSITIVE] Perturbation-based Faithfulness Evaluation
Ordering tokens by sum-pooled interaction relevance, iteratively adding most relevant tokens, and measuring Euclidean distance to the unperturbed sentence representation to assess explanation faithfulness.

**Delta**: BiLRP achieves lowest AUPC across all models and datasets (e.g., BERT on STSb: 7.80 vs random 10.28, embeddings 10.86, HxP 10.97)
**Condition**: When evaluating faithfulness of second-order explanations for similarity models; requires access to model representations.

**Evidence**: "We observe that across models and datasets, BiLRP consistently selects the features that decrease the distance between sentence representations most effectively, resulting in the lowest area under the perturbation curve."

## [POSITIVE] Synthetic Interaction Task for Explanation Evaluation
Designing a similarity model based on co-occurrence of same noun tokens, finetuning it to predict the number of co-occurring nouns, and comparing ground-truth interactions to explanation methods.

**Delta**: BiLRP ACS=0.81 vs HxP=0.62 vs Embeddings=0.67; Spearman ρ=0.94/0.89 (train/test)
**Condition**: When ground-truth feature interactions are available for validation; useful for benchmarking second-order explanation methods.

**Evidence**: "We design a similarity model based on co-occurrence statistics of features, which here are interactions between same noun tokens... BiLRP is able to select the relevant tokens in comparison to the other baselines with higher accuracy."

## [NEUTRAL] QKV-Pooling
Using the query-key-value mechanism to compute weighting coefficients before aggregating token embeddings into a sentence representation.

**Delta**: Used in synthetic experiment; no direct performance comparison against other pooling methods on real STS tasks
**Condition**: When building similarity models that require learnable pooling; used in the synthetic noun-matching experiment.

**Evidence**: "We finetune a similarity model using a BERT-base encoder with a QKV-pooling layer to correctly predict the number of co-occurring proper nouns and nouns in the STSb dataset."

## [NEGATIVE] Hessian × Product (H×P) Explanations
A second-order explanation method that computes feature interactions using the Hessian matrix of the similarity score with respect to input features.

**Delta**: ACS=0.62 vs BiLRP=0.81; higher AUPC than BiLRP across all models
**Condition**: When used as a baseline for second-order explanations; less faithful than BiLRP due to lack of relevance conservation.

**Evidence**: "For H × P, the interactions are much more selective with regard to nouns and proper nouns, and we observe considerable token interactions that are assigned negative relevance... BiLRP is able to select the relevant tokens in comparison to the other baselines with higher accuracy."

## [NEGATIVE] Token Embedding Baseline for Interaction Explanations
Computing pairwise interactions directly from token embeddings (e.g., dot product of token representations) without any backpropagation or relevance propagation.

**Delta**: ACS=0.67 vs BiLRP=0.81; higher AUPC than BiLRP
**Condition**: When used as a naive baseline for second-order explanations; fails to identify task-relevant interactions selectively.

**Evidence**: "Computing interactions directly from token embeddings results in pairwise attributions mainly between same tokens that are not selective with respect to their assigned part-of-speech (POS) tag."

## [NEGATIVE] Multilingual Mixed-Input Setting (EN-DE)
Computing similarity between sentences in different languages (e.g., English and German) using a multilingual BERT model without fine-tuning.

**Delta**: Spearman ρ drops from 47.3–58.5 (monolingual) to 24.8–35.5 (mixed multilingual)
**Condition**: When using multilingual models for cross-lingual similarity without fine-tuning; model struggles to match nouns and verbs across languages.

**Evidence**: "For the mixed-multilingual case, we observe a clear drop of correlation scores to 24.8–35.5."
