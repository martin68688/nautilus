# From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning

**Source**: https://aclanthology.org/2024.naacl-long.421/

## [POSITIVE] Instruction-Following Difficulty (IFD) Score
A metric that measures the ratio of the Direct Answer Score (loss without instruction) to the Conditioned Answer Score (loss with instruction) to quantify how much the instruction helps generate the response. Higher IFD indicates harder instructions, which are selected as 'cherry' data.

**Delta**: outperforms full-data models with only 5-10% of data
**Condition**: Used for selecting high-quality instruction tuning data; effective across LLaMA-7B, LLaMA2-7B, and LLaMA2-13B models.

**Evidence**: "Through the application of IFD, cherry samples can be pinpointed, leading to a marked uptick in model training efficiency. Empirical validations on datasets like Alpaca and WizardLM underpin our findings; with a mere 10% of original data input, our strategy showcases improved results."

## [POSITIVE] Self-Guided Data Selection (Cherry-Picking)
A three-phase methodology: (1) Learning from Brief Experience (train on a small diverse subset), (2) Evaluating Based on Experience (compute IFD scores using the pre-experienced model), (3) Retraining from Self-Guided Experience (train final model on top IFD-scored samples).

**Delta**: outperforms official Alpaca with 5% data and reimplemented WizardLM with 10% data
**Condition**: Applicable to instruction tuning of LLMs; requires a pre-experienced model trained on a small subset (e.g., 1000 samples).

**Evidence**: "By applying our methodology to the Alpaca and WizardLM instruction tuning datasets, our model outperforms the official Alpaca model with only approximately 5% data selected and outperforms the reimplemented WizardLM model with approximately 10% data selected."

## [POSITIVE] Learning from Brief Experience (Pre-Experienced Model)
Training the base LLM on a small, diverse subset (1000 samples via K-means clustering on instruction embeddings) for 1 epoch to equip it with basic instruction-following ability before computing IFD scores.

**Delta**: outperforms using no pre-experienced model; 300 samples sufficient for basic capability
**Condition**: Used in the first phase of the methodology; number of pre-experienced samples can be as low as 300-1000.

**Evidence**: "When no pre-experienced model is utilized, the corresponding cherry models have the least performance. However, even in the absence of a pre-experienced model, our IFD score remains effective... When adding the number of pre-experienced samples to 300, a distinct performance gain is discovered."

## [POSITIVE] K-Means Clustering for Diverse Pre-Experience Data
Applying K-means clustering on instruction embeddings to select 10 samples from each of 100 clusters, ensuring diversity in the pre-experienced training set.

**Delta**: comparable to other selection strategies (difficulty, random) for pre-experienced data
**Condition**: Used for selecting the initial 1000 pre-experienced samples; diversity is not critical as long as the pre-experience process is performed.

**Evidence**: "Comparing random selection and data diversity and instruction difficulty, they all surpass the Alpaca model and are comparable to each other, indicating the effectiveness of both strategies and further proving that our IFD metric is robust across different pre-experienced models."

## [POSITIVE] Threshold Filtering for Misaligned Samples (IFD > 1)
Filtering out samples where IFD score > 1, indicating that the Conditioned Answer Score is larger than the Direct Answer Score, which suggests misalignment between instruction and response.

**Delta**: removes misaligned samples, improving data quality
**Condition**: Applied during the IFD evaluation phase to discard noisy or misaligned instruction-response pairs.

**Evidence**: "To further filter out the sample whose instruction is misaligned with its response, a threshold of 1 is set. ... if the IFD score is greater than 1, the Conditioned Answer Score is even larger than the Direct Answer Score, which means the given instruction provides no useful context for the prediction of the response."

## [NEUTRAL] Pairwise Comparison Evaluation with Positional Bias Mitigation
Using GPT-4 or ChatGPT to compare model responses by sending them in two orders and defining a win only if the model does not lose in both orderings, to address positional bias.

**Delta**: N/A (evaluation methodology)
**Condition**: Used for evaluating model performance across test sets; applicable to any LLM comparison.

**Evidence**: "To further address the positional bias, we send the responses of two models to the judge twice with different orders and compare their scores. Thus we define one model to be seen as winning only if it does not lose in both the ordering."

## [POSITIVE] Selecting Data with High IFD Scores (Cherry Data)
Selecting the top 5-10% of samples with the highest IFD scores for final training, as these represent instructions that are difficult but beneficial for the model to learn.

**Delta**: outperforms full-data models with 5-10% data; outperforms random, diversity, low IFD, and high CA score baselines
**Condition**: Applied after computing IFD scores; works best with 5-10% of the original dataset.

**Evidence**: "A consistent observation across both datasets is that with merely 10% of selectively chosen data, our models manage to exceed the results of models trained on the full dataset. ... models trained using low IFD scores obtain the least performance compared with all the methods."

## [POSITIVE] Direct Use of Base Model for IFD Calculation (No Pre-Experience)
Calculating IFD scores directly on the base pre-trained model without the pre-experienced phase, using prompts from Vicuna.

**Delta**: outperforms full-data models on LLaMA2-7B and LLaMA2-13B
**Condition**: Applicable when using LLaMA2 models; more efficient but slightly less effective than using a pre-experienced model.

**Evidence**: "On both LLaMA2-7B and LLaMA2-13B models, our cherry models trained with much less data outperform the models trained with original full data. ... experiments on LLaMA2 models show that calculating IFD scores directly on the base LLaMA2 models also promises a good selection."

## [POSITIVE] Using Weak Models for IFD Calculation (Superfiltering)
Using smaller or weaker language models to compute IFD scores, which are consistent with strong models, to improve efficiency.

**Delta**: improves efficiency without significant performance loss
**Condition**: Alternative to using the target model for IFD calculation; useful for faster data filtering.

**Evidence**: "Superfiltering expands the use of IFD scores and shows that (1) Good prompting can relieve the burden of training a pre-experienced model; (2) The IFD scores calculated by weak language models are consistent with strong models, making it possible to utilize small models for filtering."
