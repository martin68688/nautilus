# Trusting Your Evidence: Hallucinate Less with Context-aware Decoding

**Source**: https://aclanthology.org/2024.naacl-short.69/

## [POSITIVE] Context-aware Decoding (CAD)
A contrastive decoding method that amplifies the difference between output probabilities when a model is used with and without context, effectively downweighting prior knowledge when relevant contextual information is provided.

**Delta**: +14.3% factuality metrics for LLaMA-30B on CNN-DM; +21% ROUGE-L for LLaMA-30B on CNN-DM; 2.9x improvement on knowledge conflicts QA for LLaMA-30B
**Condition**: Applicable to summarization tasks and knowledge conflict tasks where context contradicts prior knowledge; effective across OPT, GPT-Neo, LLaMA, FLAN-T5 models of various sizes.

**Evidence**: "CAD, without additional training, significantly improves the faithfulness of different LM families, including OPT, GPT, LLaMA and FLANT5 for summarization tasks (e.g., 14.3% gain for LLaMA in factuality metrics)."

## [POSITIVE] Contrastive ensemble of logits (with and without context)
The method computes the adjusted output distribution as softmax[(1+α) logit(y|c,x) - α logit(y|x)], contrasting the logits from the context-conditioned model and the context-free model.

**Delta**: Outperforms regular decoding across all eight models on CNN-DM and XSUM; e.g., LLaMA-30B: +21% ROUGE-L, +14.3% factKB, +7.8% BERT-P on CNN-DM
**Condition**: Used for all experiments; requires access to model logits (not applicable to API-based black-box models).

**Evidence**: "CAD outperforms the standard decoding algorithm by a large margin in all eight models across both datasets."

## [POSITIVE] Hyperparameter α for adjustment level
A scalar α controls the strength of the contrastive adjustment: α=0 reduces to regular decoding; larger α means more weight on the adjustment. α=0.5 used for summarization, α=1 for knowledge conflict tasks.

**Delta**: α=0.5 provides robust improvements over regular decoding across all three datasets; higher α more effective for knowledge conflicts
**Condition**: α=0.5 recommended for general use; α=1 recommended when prior knowledge needs to be factored out more (knowledge conflicts). Optimal α may vary by model and task.

**Evidence**: "We set α = 0.5 for all models evaluated on the summarization datasets and α = 1 for all models evaluated on the knowledge conflict datasets. We observed that α = 0.5 generally yielded good results across all settings and all datasets, but a slightly higher α is more effective in the knowledge conflict setting."

## [POSITIVE] Nucleus sampling (top-p) on adjusted distribution
After computing the adjusted output distribution from CAD, nucleus sampling with p=0.9 is applied for summarization tasks (instead of greedy decoding).

**Delta**: Outperforms regular top-p sampling baseline
**Condition**: Applied to summarization tasks (CNN-DM, XSUM) with p=0.9.

**Evidence**: "For the baselines, we use the regular decoding methods following prior work: greedy decoding for knowledge conflict tasks and top-p sampling with p=0.9 for summarization tasks. For CAD, we use the same sampling strategies on top of the adjusted output probability distribution."

## [POSITIVE] Greedy decoding for knowledge conflict tasks
Greedy decoding (selecting the most likely token at each step) is used as the baseline decoding strategy for knowledge conflict datasets (NQ-Swap, MemoTrap, NQ).

**Delta**: CAD with greedy decoding outperforms regular greedy decoding by large margins (e.g., GPT-Neo 20B: +54.4% on MemoTrap, +128% on NQ-SWAP)
**Condition**: Applied to knowledge conflict tasks (NQ-SWAP, MemoTrap, NQ).

**Evidence**: "For the baselines, we use the regular decoding methods following prior work: greedy decoding for knowledge conflict tasks and top-p sampling with p=0.9 for summarization tasks."

## [POSITIVE] Zero-shot prompting without task-specific finetuning
Models are evaluated using prompting (zero-shot or few-shot) without any additional training or fine-tuning on the target tasks.

**Delta**: CAD significantly improves faithfulness of vanilla LMs without additional training
**Condition**: Applicable to any pretrained LM in zero-shot or few-shot prompting scenarios.

**Evidence**: "CAD can be used with off-the-shelf pretrained language models without any additional training."

## [POSITIVE] Scaling effect: larger models benefit more from CAD on knowledge conflicts
The performance gain from CAD increases as model size grows, particularly on knowledge conflict tasks (Memotrap and NQ-SWAP).

**Delta**: Gain increases with model size on Memotrap and NQ-SWAP; consistent gain across sizes on CNN-DM
**Condition**: Observed for OPT models from 125M to 30B parameters on knowledge conflict datasets.

**Evidence**: "In Memotrap and NQ-SWAP, this gain increases as the model size grows, indicating that larger LMs can have a greater tendency to rely on their prior knowledge instead of reading the contexts, thereby benefiting more from CAD."
