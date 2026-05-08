# MiLe Loss: a New Loss for Mitigating the Bias of Learning Difficulties in Generative Language Models

**Source**: https://aclanthology.org/2024.findings-naacl.18/

## [POSITIVE] MiLe Loss
A novel loss function that uses information entropy of the predicted probability distribution over the vocabulary to dynamically scale the training loss, focusing more on difficult-to-learn tokens.

**Delta**: Consistent performance improvement; e.g., +1.02% average accuracy on 6.7B model (5-shot) over Cross-Entropy Loss; +4.17% on TriviaQA 0-shot over Focal Loss
**Condition**: Generative language model pretraining on text corpus with inherent token imbalance; evaluated on commonsense reasoning, closed-book QA, and MMLU benchmarks.

**Evidence**: "MiLe Loss substantially outperforms both Cross-Entropy Loss and Focal Loss on different setups with different model capacities."

## [NEUTRAL] Focal Loss for Language Models
Applying the original Focal Loss (designed for object detection) to language model pretraining by scaling loss based on the predicted probability of the ground-truth token.

**Delta**: Outperforms Cross-Entropy Loss in some settings but is consistently outperformed by MiLe Loss; e.g., on 6.7B model 0-shot, Focal Loss avg 57.57 vs MiLe Loss 57.97
**Condition**: When predicting the next token in generative language models, which is more like a multi-label classification problem.

**Evidence**: "Focal Loss struggles to give suitable scaling factors for cases with multiple valid next tokens."

## [POSITIVE] Information Entropy for Learning Difficulty Assessment
Using the information entropy of the predicted probability distribution (instead of just the ground-truth token probability) to assess token learning difficulty.

**Delta**: Enables MiLe Loss to outperform Focal Loss consistently; e.g., +4.17% on TriviaQA 0-shot
**Condition**: Cases with multiple valid next tokens (multi-label classification scenario) in language model pretraining.

**Evidence**: "When a next token is difficult-to-learn, the predicted probability distribution would be more uniform, resulting in a higher information entropy."

## [POSITIVE] Dynamic Scaling Factor based on Entropy
Scaling the Cross-Entropy Loss by (1 - entropy) where entropy is the information entropy of the predicted distribution, bounded in [1, 1+log(N)].

**Delta**: Substantial reduction in perplexity for medium/difficult tokens: e.g., from 13.541 to 13.459 (medium) and 15.517 to 15.371 (difficult) with γ=1
**Condition**: During training, when the model encounters tokens with varying learning difficulties.

**Evidence**: "MiLe Loss substantially reduces their perplexity with a noticeable decline for medium or difficult tokens."

## [POSITIVE] Training with More Tokens (200B vs 100B)
Continuing pretraining from 100B to 200B tokens to see if MiLe Loss benefits more from additional data.

**Delta**: Performance improvement increased from 1.02% (100B) to 1.33% (200B) over Cross-Entropy Loss in 5-shot setting
**Condition**: When more training data is available; the model's output distribution becomes more reasonable, making entropy-based assessment more accurate.

**Evidence**: "Using more tokens increases the benefits of MiLe Loss."

## [NEUTRAL] Hyperparameter γ Tuning
Grid search over γ values (0 to 5) for MiLe Loss to control focus on difficult tokens.

**Delta**: Performance not very sensitive; e.g., 468M model: γ=0 to 5 all outperform Cross-Entropy; 6.7B: γ=0 to 2 outperform Cross-Entropy
**Condition**: When γ is set between 0 and 2 for larger models, or 0 to 5 for smaller models; too large γ causes performance decline.

**Evidence**: "The performance of MiLe Loss is not very sensitive to the setting of the hyperparameter γ, which shows practical applicability."

## [NEUTRAL] Token Frequency Bucketing for Analysis
Grouping tokens into high/medium/low frequency buckets (80%/15%/5% coverage) to analyze perplexity and learning difficulty.

**Delta**: High freq PPL=4.323, Medium=13.541, Low=15.517 (LLaMA 6.7B on Pile)
**Condition**: Used for analysis and validation of the token imbalance problem; not a training technique.

**Evidence**: "Token imbalance can lead to the bias of learning difficulties."
