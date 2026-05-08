# ALBA: Adaptive Language-Based Assessments for Mental Health

**Source**: https://aclanthology.org/2024.naacl-long.136/

## [POSITIVE] Adaptive Language-based IRT (ALIRT)
A semi-supervised method that polytomizes language responses using supervised regression models, then applies Item Response Theory (IRT) with adaptive testing to select the next best question based on Fisher Information.

**Delta**: Pearson r ≈ 0.93 after only 3 questions; captures over 90% of variance explained (R²) after 4 questions vs. needing at least 7 for fixed ordering.
**Condition**: When assessing depression or anxiety severity using open-ended language responses; outperforms non-adaptive baselines and Actor-Critic model.

**Evidence**: "We found ALIRT to be the most accurate and scalable, achieving the highest accuracy with fewer questions (e.g., Pearson r ≈ 0.93 after only 3 questions as compared to typically needing at least 7 questions)."

## [POSITIVE] Actor-Critic Model
A supervised reinforcement learning-based approach with two models: a Measure Model predicting psychometric scores and an Error Model predicting prediction error to select the next question adaptively.

**Delta**: Outperforms fixed-order baselines but not as accurate or scalable as ALIRT; e.g., ActorCritic-Ŷ achieves 0.740 (CTT) at 4 items vs. ALIRT-Ŷ 0.726.
**Condition**: When computational resources are less constrained; performance boost is not always significant compared to ALIRT despite higher parameter count.

**Evidence**: "While we found both methods to improve over non-adaptive baselines, We found ALIRT to be the most accurate and scalable."

## [POSITIVE] Polytomization of Language Responses
Discretizing continuous language response predictions into ordinal categories (e.g., 2, 4, 8, or 12 levels) using percentile thresholds from supervised regression models, enabling IRT application to linguistic data.

**Delta**: 8-level polytomization works as well as 12-level with fewer parameters; 12-level risks overfitting.
**Condition**: When discretizing language responses for IRT; optimal at 8 levels for this dataset.

**Evidence**: "A polytomization of 8 works just as well as 12 with 2/3 the number of parameters. We also found that a polytomization of 13 or above results in missing values."

## [POSITIVE] Maximum Fisher Information (MFI) for Question Ordering
Selects the next question that maximizes Fisher Information at the current latent trait estimate, a standard IRT criterion for adaptive testing.

**Delta**: ALIRT (using MFI) achieves higher correlations than fixed ordering and Actor-Critic; e.g., ALIRT-ˆL reaches 0.935 (vs. ˆL_all) at 3 items.
**Condition**: When using IRT-based adaptive testing; effective for both depression and anxiety assessment.

**Evidence**: "ALIRT, which uses Maximum Fisher Information for adaptive ordering, does not try to optimize for errors/correlation with the 'true' scores, but the ordering produced by it largely helps across both the scoring paradigms."

## [POSITIVE] Latent Semantic Analysis (LSA) Word Embeddings
Training custom word embeddings using PCA on a term-document matrix with log-entropy weighting (LSA) on a large mental health response dataset, using 10 dimensions per question.

**Delta**: Outperforms dimension-reduced GloVe and RoBERTa-large embeddings in this context; enables effective polytomization.
**Condition**: When working with short descriptive word responses in mental health; low-resource domains.

**Evidence**: "This approach has been proven as effective as word2vec, particularly in the context of psychology... Preferring smaller embedding sizes which are suitable for low-resource domains like mental health, we utilize 10 dimensions for each question."

## [POSITIVE] Supervised Polytomization via Multiple Ridge Regression
Training a separate ridge regression model per question to predict psychometric measures (PHQ-9/GAD-7) from averaged word embeddings, then thresholding predictions into ordinal categories.

**Delta**: Average RMSE 10.93 for PHQ-9 and 8.64 for GAD-7 across models.
**Condition**: When converting language responses to ordinal IRT inputs; requires a training split for supervised fitting.

**Evidence**: "For each of the J questions, a multiple-ridge regression model is trained to predict the psychometric measure (PHQ-9 and GAD-7) on the averaged word embeddings of its responses."

## [POSITIVE] Fixed Forward Selection Ordering
A baseline ordering strategy that greedily selects questions with the highest Pearson correlation between polytomized responses and true scores, in a fixed order for all participants.

**Delta**: Outperforms random ordering; e.g., FixedFor-Ŷ achieves 0.690 (CTT) at 2 items vs. RandomOrder-Ŷ 0.640.
**Condition**: When adaptive methods are not feasible; provides a strong non-adaptive baseline.

**Evidence**: "Forward Selection (fixedFor) – As a fixed ordering baseline, we use forward selection to determine ordering, greedily picking the questions with the highest Pearson correlation."

## [POSITIVE] Fixed Backward Elimination Ordering
A baseline ordering strategy that eliminates questions with the lowest Pearson correlations with true scores, in a fixed order.

**Delta**: Similar performance to forward selection; e.g., FixedBack-Ŷ achieves 0.669 (CTT) at 2 items.
**Condition**: When adaptive methods are not feasible; comparable to forward selection.

**Evidence**: "Backward Elimination (fixedBack) – As another fixed ordering baseline, we use backward elimination based on eliminating items with the lowest Pearson correlations."

## [NEUTRAL] Random Ordering Baseline
A baseline where questions are presented in random order for each participant, with equal probability for any unasked question.

**Delta**: Worst performance among ordering strategies; e.g., RandomOrder-ˆL achieves 0.526 (CTT) at 1 item.
**Condition**: Used as a lower-bound baseline for comparison.

**Evidence**: "Random – We use a random ordering of the questions for each participant, with any of the unasked questions having an equal probability of being asked next."

## [POSITIVE] Decision Tree Adaptive Ordering
An adaptive baseline where a decision tree selects the next question based on previous responses, with splits pre-calculated on training data.

**Delta**: Outperforms random but underperforms ALIRT; e.g., DecisionTree-Ŷ achieves 0.685 (CTT) at 2 items.
**Condition**: When a simpler adaptive method is needed; less effective than IRT-based methods.

**Evidence**: "As a more sophisticated, adaptive baseline, we also explore the Decision Tree, which can be seen as defining an adaptive strategy where the next best question is picked based on the condition encountered at the current node."

## [POSITIVE] Latent Estimate (ˆL) Scoring Paradigm
Using the IRT-derived latent variable as the score, estimated via maximum a posteriori (MAP) after each question, evaluated against the latent score from all items (ˆL_all).

**Delta**: ALIRT-ˆL achieves 0.935 (vs. ˆL_all) at 3 items, outperforming CTT-based scoring in latent space.
**Condition**: When the goal is to approximate the full-information latent trait estimate with fewer questions.

**Evidence**: "We find that the measures are consistent across both approaches. ˆL_all refers to the latent score when all the 11 items are used. This means that administering just 3 items in the questionnaire based on ALIRT can achieve >0.9 correlation (Pearson r) with the latent score from using all the 11 items."

## [POSITIVE] Classical Test Score (Ŷ) Scoring Paradigm
Averaging the predicted psychometric measures (e.g., PHQ-9) from individual question models, evaluated against the full PHQ-9 or GAD-7 score.

**Delta**: ALIRT-Ŷ achieves 0.726 (CTT) at 3 items, best among CTT-based methods.
**Condition**: When compatibility with standard questionnaire scoring (CTT) is desired.

**Evidence**: "Classical Test Score (Ŷ), on the other hand, is the average of item scores, much like scores derived from a traditional questionnaire for mental health assessment."

## [POSITIVE] Regression over Predicted Scores (Regr(Ŷ))
Using regression on the item response scores (predicted by individual question models) to output a final score, instead of simple averaging.

**Delta**: Improves correlations in Actor-Critic setting; e.g., ActorCritic Regr(Ŷ) achieves 0.694 (CTT) at 2 items vs. 0.693 for Ŷ.
**Condition**: When using Actor-Critic or similar models; minor improvement over simple averaging.

**Evidence**: "While there is still risk of parameter explosion if there were more questions, the method does not demand more compute and seems to improve the correlations of the predicted scores in the Actor-Critic setting."

## [NEUTRAL] Regression over Word Embeddings (Regr(X))
Using regression directly on word embeddings (10-dim) as input to predict the final score, instead of using item scores.

**Delta**: Does not fare better; e.g., ALIRT Regr(X) achieves 0.685 (CTT) at 2 items vs. 0.685 for Ŷ.
**Condition**: When embedding dimensionality is high relative to dataset size; risk of overfitting.

**Evidence**: "We find that this method does not really fare better across both Actor-Critic and ALIRT. Since we use 10 dimensional word embeddings, the number of parameters is increased tenfold, which could cause the model to overfit."

## [NEUTRAL] Dimension Reduction on Contextual Embeddings (RoBERTa-large, GloVe)
Reducing high-dimensional contextual embeddings (e.g., RoBERTa-large 1024-dim, GloVe 300-dim) to 10 dimensions via PCA for use in ALIRT and Actor-Critic.

**Delta**: Comparable to LSA but not significantly better; e.g., ALIRT-ˆL with RoBERTa achieves 0.944 (vs. ˆL_all) at 4 items vs. 0.955 with LSA.
**Condition**: When using contextual embeddings for short, descriptive word responses; LSA may be equally effective.

**Evidence**: "The difference between fixed and adaptive strategies is not as significant as when using static embeddings. This can be explained with the context-independent word responses in the dataset used."

## [POSITIVE] 9-Fold Cross-Validation with Fixed Splits
Splitting the dataset into 9 folds: 4 for polytomization (D_poly), 4 for IRT training (D_tr), and 1 for testing (D_te), ensuring consistent splits across all methods.

**Delta**: Enables reliable comparison; all p-values < 0.001.
**Condition**: When evaluating adaptive testing methods; ensures fair comparison across models.

**Evidence**: "We run a 9-fold cross validation (D_poly:4, D_tr:4, D_te:1) across the two phases as described in the Algorithm 1– hence our approach is semi-supervised."

## [POSITIVE] Unidimensional IRT Modeling
Modeling a single latent trait (e.g., depression severity) rather than multidimensional IRT, justified by Kaiser criterion and Bartlett test indicating one significant factor.

**Delta**: KMO value 0.924, Bartlett test p < 0.001, Kaiser criterion indicates 1 latent factor.
**Condition**: When the dataset supports a unidimensional latent structure (e.g., depression or anxiety severity).

**Evidence**: "We model a single latent score in this dataset, as opposed to modeling multiple mental health conditions simultaneously with multidimensional IRT models. This choice is justified in Appendix C."
