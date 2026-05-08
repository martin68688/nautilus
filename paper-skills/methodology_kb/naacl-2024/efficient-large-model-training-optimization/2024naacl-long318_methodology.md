# Contrastive and Consistency Learning for Neural Noisy-Channel Model in Spoken Language Understanding

**Source**: https://aclanthology.org/2024.naacl-long.318/

## [POSITIVE] Contrastive and Consistency Learning (CCL)
A two-stage method that first correlates error patterns between clean and noisy ASR transcripts via token-based contrastive learning, then emphasizes consistency of latent features via consistency learning.

**Delta**: +2.59% accuracy on SLURP Noisy0.56; outperforms baselines on all four datasets
**Condition**: Applied to noisy-channel model for SLU; effective across various WER levels and datasets (SLURP, Timers, FSC, SNIPS).

**Evidence**: "On the SLURP benchmark dataset, the proposed CCL method improves the intent classification performance by 2.59% under severe ASR errors."

## [POSITIVE] Selective-token Contrastive Learning
Creates positive pairs between word tokens from clean and noisy transcripts using edit distance alignment, then applies contrastive loss to pull aligned tokens together and push non-matching tokens apart.

**Delta**: Outperforms utterance-only contrastive learning; ablation shows larger performance drop at higher WER when removed
**Condition**: Used in Stage 1 of CCL; particularly effective when ASR errors cause token misalignment (substitution, insertion, deletion).

**Evidence**: "We observe that the higher the speech recognition error rate, the more significant the performance degradation while removing the token-based contrastive loss."

## [POSITIVE] Utterance Contrastive Learning
Aligns entire clean and noisy utterances as positive pairs using [CLS] token representations, increasing similarity compared to negative pairs in the batch.

**Delta**: Contributes to overall CCL performance; ablation shows improvement over using only selective-token loss
**Condition**: Used alongside selective-token contrastive learning in Stage 1; helps align utterance-level semantics.

**Evidence**: "In token-based contrastive learning, better performance is observed when mismatched tokens are selectively correlated using Lsel, rather than Lutt."

## [POSITIVE] Consistency Learning
Reduces distance between latent features of clean and noisy transcripts and aligns their target probability distributions, using a reference network to provide better latent representations.

**Delta**: Outperforms cross-entropy loss in fine-tuning; ablation shows improved results with consistency loss over CE
**Condition**: Stage 2 of CCL; applied after token-based contrastive learning; reference network is trained with CE on clean transcripts.

**Evidence**: "In the fine-tuning, we attain improved results employing the consistency loss rather than the commonly used cross-entropy loss."

## [POSITIVE] Reference Network with Cross-Entropy on Clean Transcripts
A separate reference network is trained to maximize intent prediction accuracy on clean transcripts, providing better latent features for the inference network to bootstrap from.

**Delta**: Enables consistency learning to work effectively; reference network achieves high accuracy on clean data
**Condition**: Used during training only; reference network receives clean transcripts; inference network bootstraps from its features.

**Evidence**: "The reference latent feature used in the consistency learning should have a relatively better representation than the noisy latent feature."

## [POSITIVE] Inference Network Only at Evaluation
During evaluation, only the inference network is used (no reference network), so there is no increase in inference time.

**Delta**: No inference time increase compared to baseline models
**Condition**: Applied during evaluation; inference network takes only noisy ASR transcript as input.

**Evidence**: "This allows us to solve the ASR error problem without any increase in inference time."

## [POSITIVE] RoBERTa as Encoder
Uses pre-trained RoBERTa as the encoder for both reference and inference networks, leveraging its bidirectional contextual representations.

**Delta**: Outperforms decoder-based LLMs (GPT-Neo, GPT-2) even with fewer parameters (110M vs 125M/355M)
**Condition**: Used as the base encoder for all experiments; compared against GPT-Neo and GPT-2 in additional experiments.

**Evidence**: "The encoder-based language model consistently outperforms decoder-based language models, even with fewer parameters (110M)."

## [NEUTRAL] Hyperparameter Tuning for Consistency Loss Weight (λcon)
The weight λcon for consistency loss is tuned from 0.0 to 1.0 to find the best value.

**Delta**: No specific delta reported; tuning is standard practice
**Condition**: Applied during training of CCL; specific optimal value not disclosed.

**Evidence**: "We tune λcon of best value that is 0.0 to 1.0."

## [NEUTRAL] Hyperparameter Tuning for Contrastive Loss Weight (λctr)
The weight λctr for combining selective-token and utterance contrastive losses is set to 0.7.

**Delta**: No specific delta reported; fixed value used across experiments
**Condition**: Fixed during all experiments; not tuned per dataset.

**Evidence**: "We use the same hyperparameters: learning rate 1e-6 and λctr 0.7."

## [NEUTRAL] Learning Rate Tuning for Consistency Learning
Learning rate for consistency learning is selected from {1e-5, 3e-5, 5e-5, 7e-5, 1e-4}.

**Delta**: No specific delta reported; standard hyperparameter search
**Condition**: Applied during consistency learning stage; specific optimal value not disclosed.

**Evidence**: "The learning rate is selected for consistency learning from {1e-5, 3e-5, 5e-5, 7e-5, 1e-4}."

## [NEUTRAL] Batch Size of 64
All experiments use a batch size of 64.

**Delta**: No specific delta reported; consistent across all methods
**Condition**: Applied to all training runs.

**Evidence**: "The batch size in our experiments is 64, the same for all."
