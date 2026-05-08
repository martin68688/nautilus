# Unveiling the Generalization Power of Fine-Tuned Large Language Models

**Source**: https://aclanthology.org/2024.naacl-long.51/

## [POSITIVE] Task-specific fine-tuning
Fine-tuning LLMs on a single downstream dataset (e.g., XSum for summarization, Amazon for sentiment) rather than multi-task or few-shot fine-tuning.

**Delta**: Outperforms baseline Llama-2 with ICL on in-domain test sets (e.g., XLSum, Socialqa, MNLI-1, Paws).
**Condition**: In-domain datasets of the same task type; generation and classification tasks.

**Evidence**: "The fine-tuned models, trained with varying sample sizes (2K, 4K, 6K), exhibit superior 0-shot (w/o ICL) performance compared to the original baseline Llama2 using ICL across most datasets, notably XLSum, Socialqa, MNLI-1, MNLI-2, and Paws."

## [NEUTRAL] Fine-tuning with varying training sample sizes (2K, 4K, 6K)
Fine-tuning models using subsets of 2,000, 4,000, and 6,000 samples to analyze the impact of training data volume.

**Delta**: Performance gains are inconsistent; e.g., on XSum only slight increase from 2K to 6K; on Paws accuracy jumps from 81.6% to 93.2% from 2K to 4K, but 4K to 6K brings subtle or negative impacts.
**Condition**: In-domain test sets; varies by task.

**Evidence**: "Fine-tuning with more samples may not consistently improve performance on the test set. ... the relationship between the volume of fine-tuning data and in-domain test performance is task-dependent and not straightforward."

## [NEGATIVE] Fine-tuning on generation tasks (out-of-domain evaluation)
Fine-tuning on generation tasks (e.g., XSum, Socialqa) and evaluating on out-of-domain datasets of the same task type.

**Delta**: Fine-tuned models underperform baseline Llama-2 on out-of-domain generation datasets (e.g., PeerRead, CNN/DailyMail, Tweetqa, Sciqa).
**Condition**: Out-of-domain datasets of the same generation task type.

**Evidence**: "In generation testing datasets ... fine-tuned models perform worse than baseline models, and this gap persists regardless of the number of in-context examples provided."

## [POSITIVE] Fine-tuning on classification tasks (out-of-domain evaluation)
Fine-tuning on classification tasks (e.g., Amazon, Paws, MNLI) and evaluating on out-of-domain datasets of the same task type.

**Delta**: Fine-tuned models consistently outperform baseline Llama-2 on out-of-domain classification datasets (e.g., SST2, Yelp, QQP, STS-B, RTE, GPTNLI).
**Condition**: Out-of-domain datasets of the same classification task type.

**Evidence**: "For other classification tasks (i.e., paraphrase detection and natural language inference), fine-tuned models consistently outperform the baseline Llama-2, as shown in Figure 2 (g)-(j)."

## [NEGATIVE] Cross-task fine-tuning (classification to generation)
Fine-tuning on classification tasks and evaluating on generation tasks (e.g., XSum, Socialqa).

**Delta**: Models fine-tuned on classification data lead to almost zero Rouge-L scores for generation tasks.
**Condition**: Cross-task evaluation from classification to generation.

**Evidence**: "Models fine-tuned on classification tasks fail to generalize to generation tasks. ... fine-tuning the models on classification data leads to almost zero Rouge-L scores for generation tasks."

## [NEUTRAL] Cross-task fine-tuning (generation to classification)
Fine-tuning on generation tasks and evaluating on classification tasks (e.g., Amazon, MNLI, Paws).

**Delta**: Performance varies greatly; e.g., fine-tuning on XSum boosts MNLI performance, while fine-tuning on Amazon hurts MNLI.
**Condition**: Cross-task evaluation from generation to classification; depends on specific training data.

**Evidence**: "Fine-tuning on a dataset like Amazon negatively impacts performance on the MNLI-1 test set, whereas fine-tuning on XSum significantly boosts it."

## [POSITIVE] Fine-tuning with In-Context Learning (FTICL) on generation tasks
During fine-tuning, prepending in-context examples to the input (e.g., 1 or 2 examples) for generation tasks.

**Delta**: FTICL models achieve better out-of-domain generalization than vanilla fine-tuned models, sometimes surpassing baseline Llama-2 (e.g., on PeerRead).
**Condition**: Out-of-domain and cross-task evaluation for generation tasks.

**Evidence**: "Models fine-tuned using FTICL achieve a better out-of-domain generalization performance. ... the FTICL model with one in-context example during fine-tuning (FC1) showcases a remarkable gain over the vanilla fine-tuned model (FT) on PeerRead, surpassing even the baseline Llama-2 model using in-context learning."

## [NEGATIVE] Fine-tuning with In-Context Learning (FTICL) on classification tasks
During fine-tuning, prepending in-context examples to the input for classification tasks.

**Delta**: FTICL models generally perform worse than vanilla fine-tuned models on in-domain and out-of-domain classification test sets.
**Condition**: In-domain and out-of-domain classification tasks.

**Evidence**: "For the classification tasks ... fine-tuning with in-context learning (FTICL) models generally perform worse than vanilla fine-tuned models. ... FTICL models can generally outperform the baseline Llama but lag behind vanilla FT models."

## [POSITIVE] Prompt format variation (Prompt-1 vs Prompt-2) for cross-task evaluation
Using different prompt formats (e.g., uniform starting sequences vs. distinct prompts) to test cross-task generalization.

**Delta**: Prompt-2 enables a model fine-tuned on Amazon to start working on XSum (generation task), whereas Prompt-1 failed.
**Condition**: Cross-task evaluation, especially from classification to generation tasks.

**Evidence**: "From Figure 3 (i), the model fine-tuned on Amazon starts to work on XSum with such new prompts. ... the prompt format is crucial for cross-task generalization."

## [POSITIVE] Parameter weight deviation analysis for FTICL
Calculating average parameter weight difference between fine-tuned models (FTICL and FT) and the original Llama-2 to measure deviation.

**Delta**: FTICL models show smaller deviation (e.g., 7.95e-05 vs 8.54e-05 on Socialqa; 8.03e-05 vs 1.0e-04 on XSum) compared to vanilla FT.
**Condition**: Generation tasks; hypothesized reason for improved generalization.

**Evidence**: "FTICL tends to deviate less from the original LLM than vanilla fine-tuning. ... on Socialqa, FTICL (7.95e-05) vs. FT (8.54e-05); on XSum, FTICL (8.03e-05) vs. FT (1.0e-04)."
