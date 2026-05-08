# Detecting Bipolar Disorder from Misdiagnosed Major Depressive Disorder with Mood-Aware Multi-Task Learning

**Source**: https://aclanthology.org/2024.naacl-long.278/

## [POSITIVE] Multi-task Learning (MTL) with Uncertainty Weight Loss
Jointly learning three tasks: (i) BD risk prediction (main), (ii) BD mood level estimation, and (iii) mood variation assessment. Uses uncertainty weight loss to balance task-specific losses with different scales (post-level vs. user-level).

**Delta**: F1 57.8% vs. 53.3% for single-task learning (STL) in BD risk detection; MAE 0.701 vs. 0.705 for mood level prediction
**Condition**: When predicting BD risk from longitudinal social media posts; requires labeled auxiliary tasks (mood level, mood variation).

**Evidence**: "Notably, the performance of the proposed model using multi-task learning (MTL) is superior to employing single-task learning (STL) in both tasks."

## [POSITIVE] Dynamic Mood-Aware Attention
A time-aware attention mechanism that fuses a time parameter δ(z_t, Δt) with self-attention to emphasize critical mood states over time, considering the time interval between posts.

**Delta**: F1 improvement from 46.9% to 53.2% when added to the base temporal convolution model
**Condition**: When modeling sequential mood data where temporal intervals between posts are irregular and mood persistence matters.

**Evidence**: "By applying the Dynamic Mood-Aware Attention layer, we find drastic improvements in F1 scores (46.9% → 53.2%), which emphasizes the importance of retaining temporal information in predicting a user's risk of BD."

## [POSITIVE] Temporal Convolution Layer (Conv1D)
A 1-D convolution with kernel size k applied to SBERT embeddings to capture local dynamic mood patterns over time in the post sequence.

**Delta**: Base model (temporal conv only) achieves F1 46.9%; outperforms non-contextual baselines
**Condition**: When extracting local sequential patterns from sentence embeddings of social media posts.

**Evidence**: "Overall, the proposed model and most other time-series models outperform non-contextual models."

## [POSITIVE] Sentence-BERT (SBERT) for Post Embedding
Using a pretrained Sentence-BERT model to generate semantically meaningful sentence embeddings for each post, instead of traditional bag-of-words or other PLMs.

**Delta**: SBERT achieves MAE 0.757 for mood level prediction, outperforming BERT (0.907), EmoNet (0.857), GoEmotions (0.804), XLNet (0.838)
**Condition**: When encoding short social media text for downstream mood/risk prediction tasks.

**Evidence**: "The performance of the pretrained SBERT model surpasses other state-of-the-art semantic analysis models, which implies that SBERT can generate sentence embeddings with semantically significant information."

## [POSITIVE] Joint Learning of Mood Level and Mood Variation
Training the model to simultaneously predict both absolute mood level (ym) and relative mood variation (yv) as auxiliary tasks, rather than either alone.

**Delta**: F1 57.2% (both) vs. 48.9% (mood level only) vs. 50.0% (mood variation only)
**Condition**: When clinical guidelines (DSM-5) emphasize both absolute mood and relative variation for BD diagnosis.

**Evidence**: "Our findings also indicate that jointly learning the mood variation task outperforms learning the mood level task alone, but considering both information together is the most effective."

## [POSITIVE] Uncertainty Weight Loss for Multi-task Balancing
Using learnable uncertainty parameters σ_r and σ_m to dynamically weight the cross-entropy loss (main task) and MSE losses (auxiliary tasks), preventing any single task from dominating.

**Delta**: F1 57.8% (with uncertainty) vs. 60.7% (without uncertainty but with both aux tasks); precision increases from 54.0% to 72.0%
**Condition**: When combining tasks with different granularities (post-level vs. user-level) and different loss functions (cross-entropy vs. MSE).

**Evidence**: "Notably, applying uncertainty weight parameters θ decreases performance significantly in recall but increases precision. This suggests that θ effectively manages the task weights during training, particularly considering their distinct levels of granularity... ultimately enhancing the overall performance."

## [POSITIVE] Expert-Validated Annotation with DSM-5 Criteria
Dataset labels (diagnosis type, BD mood level -3 to 3) validated by a psychiatrist and clinical psychologist using Krippendorff's alpha and Cohen's kappa, with mood levels defined per DSM-5 severity criteria.

**Delta**: Krippendorff's α = 0.87, Cohen's κ = 0.65 between experts and annotators
**Condition**: When constructing a reliable ground truth for mental health NLP tasks; essential for clinical relevance.

**Evidence**: "Table 3 indicates a high level of agreement between experts and annotators, with an overall Krippendorff's α score of 0.87 and Cohen's κ score of 0.65, confirming the reliability of our dataset."

## [POSITIVE] Data Filtering for Temporal Alignment of Diagnosis
Three filtering strategies: (1) exclude users mentioning BD before diagnosis, (2) retain only posts within one year of diagnosis, (3) require at least three posts reporting BD diagnosis after MDD diagnosis.

**Delta**: Enables meaningful temporal modeling; average diagnosis shift duration ~560 days
**Condition**: When using self-reported diagnosis timestamps from social media; critical for longitudinal studies.

**Evidence**: "To alleviate the time-delay issue, where the reported diagnoses may not align with the timing of the posts, we implemented three data filtering strategies..."

## [POSITIVE] Observation Period of 12+ Months
Using a longer observation window (l months) of historical posts to capture recurrent mood swings characteristic of BD, as opposed to shorter windows.

**Delta**: Performance improves with longer history (5 to 24 months); best at 24 months
**Condition**: When predicting BD risk; longer observation periods (≥12 months) are more effective than short ones (e.g., 6 months).

**Evidence**: "Notably, the model performance improves with a more extended training history, which is associated with the recurrent mood swings exhibited by BD patients, necessitating a substantial duration to capture their patterns."

## [NEUTRAL] Transformer Encoder with Positional Encoding
Applying a standard Transformer encoder with sine/cosine positional encoding to capture global dependencies in the post sequence.

**Delta**: Part of the full model; ablation shows attention layer (with time-aware parameter) is critical, not the Transformer alone
**Condition**: When modeling sequential data; requires additional time-aware mechanism to preserve temporal order.

**Evidence**: "The Transformer encoder (Vaswani et al., 2017) is applied with positional encoding (PE), which is widely recognized for its effectiveness in capturing global dependencies..."

## [NEGATIVE] Using LLMs (ChatGPT, MentaLLaMa) for BD Risk Detection
Applying large language models (GPT-3.5, fine-tuned LLaMA) with prompting for binary classification of BD risk from historical posts.

**Delta**: ChatGPT F1 0.130, MentaLLaMa F1 0.190 vs. proposed MTL F1 0.578
**Condition**: When applied to future risk prediction from historical posts (not current status classification); LLMs underperform specialized models.

**Evidence**: "Specifically, LLMs (Brown et al., 2020; Yang et al., 2023b) exhibited inferior performance compared to the proposed model... This suggests that LLMs still exhibit a notable gap due to a lack of mental health-specific knowledge and inconsistencies depending on prompting strategies."

## [NEGATIVE] Single-Task Learning (STL) Baseline
Training the model with only the main BD risk prediction task, without auxiliary mood tasks.

**Delta**: F1 53.3% vs. MTL 57.8% for BD risk detection; MAE 0.705 vs. MTL 0.701 for mood level
**Condition**: When auxiliary mood information is available; STL underperforms MTL.

**Evidence**: "Notably, the performance of the proposed model using multi-task learning (MTL) is superior to employing single-task learning (STL) in both tasks."
