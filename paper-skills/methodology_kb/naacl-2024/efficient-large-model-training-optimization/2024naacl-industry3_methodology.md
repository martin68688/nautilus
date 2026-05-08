# SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling

**Source**: https://aclanthology.org/2024.naacl-industry.3/

## [POSITIVE] Depth Up-Scaling (DUS)
A method for scaling LLMs by increasing the number of layers (depthwise scaling) followed by continued pretraining, without using mixture-of-experts.

**Delta**: Outperforms Llama 2, Mistral 7B, and other models of similar size; SOLAR 10.7B-Instruct achieves highest H6 score (74.20) surpassing Mixtral-8x7B-Instruct and Qwen 72B.
**Condition**: Applicable to all transformer architectures; used to up-scale from a 32-layer base model to a 48-layer model.

**Evidence**: "SOLAR 10.7B outperforms existing models like Llama 2 and Mistral 7B in various benchmarks... SOLAR 10.7B-Instruct... significantly outperforms the Mixtral-8x7B-Instruct model across various evaluation metrics."

## [POSITIVE] Depthwise Scaling (removing middle layers)
From a base model with n layers, duplicate it, remove the final m layers from the original and the initial m layers from the duplicate, then concatenate to form a scaled model with s = 2*(n-m) layers. For n=32, s=48, m=8.

**Delta**: Enables scaling from 7B to 10.7B parameters; rapid performance recovery during continued pretraining.
**Condition**: Used when scaling up a base model; m=8 chosen due to hardware constraints and target parameter count between 7B and 13B.

**Evidence**: "We consider that the particular way of depthwise scaling has isolated the heterogeneity in the scaled model which allowed for this fast performance recovery."

## [POSITIVE] Continued Pretraining after Depthwise Scaling
After depthwise scaling, the model undergoes continued pretraining to recover performance that initially drops below the base LLM.

**Delta**: Rapid performance recovery observed; final model outperforms base model.
**Condition**: Applied after depthwise scaling; necessary to recover and improve performance.

**Evidence**: "The performance of the depthwise scaled model initially drops below that of the base LLM. Thus, we additionally apply the continued pretraining step... Experimentally, we observe rapid performance recovery of the scaled model during continued pretraining."

## [POSITIVE] Using Llama 2 Architecture with Mistral 7B Pretrained Weights
Initialize the Llama 2 architecture (32-layer) with pretrained weights from Mistral 7B, leveraging community resources and top performance.

**Delta**: Outperforms both Llama 2 and Mistral 7B in benchmarks.
**Condition**: Base model selection for DUS; leverages existing high-quality pretrained weights.

**Evidence**: "We initialize the Llama 2 architecture with pretrained weights from Mistral 7B, as it is one of the top performers compatible with the Llama 2 architecture."

## [POSITIVE] Instruction Tuning with Multiple Datasets (Alpaca-GPT4, OpenOrca, Synth. Math-Instruct)
Fine-tuning the model on instruction-following datasets including a synthesized math QA dataset (Synth. Math-Instruct) to enhance mathematical capabilities.

**Delta**: Adding Synth. Math-Instruct boosts GSM8K from 52.24 to 64.14 (SFT v1 vs SFT v3); H6 improves from 69.15 to 70.88 (SFT v1 vs SFT v4).
**Condition**: Applied during instruction tuning stage; Synth. Math-Instruct is particularly beneficial for math reasoning tasks.

**Evidence**: "Adding the Synth. Math-Instruct dataset... boosts GSM8K scores to 64.14... we get our highest H6 score of 70.88 with higher scores than 'SFT v3' for all tasks."

## [POSITIVE] Model Merging (Averaging Weights) of Instruction-Tuned Models
Merging two instruction-tuned models (SFT v3 and SFT v4) by simply averaging their weights to combine strengths.

**Delta**: Merged model 'SFT v3+v4' achieves H6 71.11, higher than either individual model; GSM8K improves to 66.57.
**Condition**: Effective when merging models with different strengths (e.g., one strong on math, another on other tasks).

**Evidence**: "To our surprise, the resulting merged model 'SFT v3+v4' retains the high scores for non-GSM8K tasks from 'SFT v4' but also achieves a higher GSM8K score than 'SFT v3' or 'SFT v4'."

## [POSITIVE] Alignment Tuning with sDPO (Improved DPO)
Using an improved version of Direct Preference Optimization (sDPO) to align the model with human/AI preferences, using datasets like Ultrafeedback Cleaned and Synth. Math-Alignment.

**Delta**: DPO v1 achieves H6 73.06, a substantial boost from SFT base model score of 70.03; adding Synth. Math-Alignment (DPO v2) improves GSM8K from 58.83 to 60.27.
**Condition**: Applied after instruction tuning; Synth. Math-Alignment helps maintain math performance during alignment.

**Evidence**: "For 'DPO v1', it achieves 73.06 in H6, which is a substantial boost from the SFT base model score of 70.03... Adding Synth. Math-Alignment to train 'DPO v2', we see that the GSM8k score improves to 60.27."

## [POSITIVE] Synthesizing Math Alignment Data (Synth. Math-Alignment)
Creating DPO tuples from rephrased math QA pairs: using the rephrased answer as chosen and original answer as rejected, based on the assumption that rephrased answers are better.

**Delta**: Improves GSM8K from 58.83 (DPO v1) to 60.27 (DPO v2) without negatively impacting other tasks.
**Condition**: Used during alignment tuning to preserve/enhance math capabilities.

**Evidence**: "Adding Synth. Math-Alignment to train 'DPO v2', we see that the GSM8k score improves to 60.27... Other task scores are also not negatively impacted."

## [POSITIVE] Merging Alignment-Tuned Models with Different Strengths
Merging two DPO models (Cand. 1 strong on GSM8K, Cand. 2 strong on other tasks) using averaging or SLERP.

**Delta**: Merge v3 achieves H6 74.05, the highest among merge candidates; GSM8K improves to 65.50.
**Condition**: Effective when merge candidates have sufficiently different strengths; exact merge method is less crucial.

**Evidence**: "We merge these two models using various methods... the different merge methods have little effect on the H6 scores... we chose 'Merge v1' as our SOLAR 10.7B-Instruct model."

## [POSITIVE] Filtering FLAN-derived Data to Avoid Benchmark Contamination
Filtering OpenOrca (derived from FLAN) to remove data overlapping with benchmark datasets like ARC, HellaSwag, MMLU, etc.

**Delta**: Ensures low data contamination; contamination test results show values well below threshold (e.g., ARC: 0.06, GSM8K: 0.70).
**Condition**: Applied during data preprocessing for instruction tuning to maintain evaluation integrity.

**Evidence**: "For datasets such as OpenOrca, which are derived from FLAN, we filter data that overlaps with the benchmark datasets... All four tested benchmark datasets yield results well below the contamination threshold."
