# Efficient Sample-Specific Encoder Perturbations

**Source**: https://aclanthology.org/2024.naacl-short.57/

## [POSITIVE] Sample-specific encoder perturbation via NAP gradients
A lightweight proxy network (NAP) is trained to approximate a sequence-level metric (e.g., COMET, WER). At inference, the gradient of the NAP with respect to the encoder output is used to compute a sample-specific perturbation to the encoder output, which is then added to the frozen encoder output before decoding.

**Delta**: Flan-T5 Large en→de COMET: 68.1 → 69.9 (+1.8); Flan-T5 Base en→de COMET: 64.3 → 66.0 (+1.7); Flan-T5 Small en→de COMET: 58.8 → 61.4 (+2.6); consistent WER improvements on AMI for Whisper models.
**Condition**: Applicable to frozen encoder-decoder foundation models (Flan-T5 for NMT, Whisper for ASR). Works best when the base model has room for improvement (larger gains on smaller models and harder tasks like AMI vs. LibriSpeech). Requires tuning step size α on a validation set.

**Evidence**: "Results display consistent improvements in performance evaluated through COMET and WER respectively."

## [POSITIVE] Non-Autoregressive Proxy (NAP) training on metric scores
A small network (attention pooling + 3 linear layers) is trained on top of the frozen encoder to predict a sequence-level metric (COMET or WER) using Pearson correlation loss. The proxy is trained on greedy decoding outputs of the foundation model.

**Delta**: NAP Pearson correlation on en→de: 75.6%; on en→es: 68.1%; cross-direction correlations also reasonable (e.g., en→de NAP on de→en: 65.7%).
**Condition**: Requires a frozen encoder-decoder model. Proxy size is small (e.g., 20.9M for Flan-T5 Large). Training data can be from a different domain/direction and still yield useful gradients.

**Evidence**: "We observe that a proxy trained on a specific translation pair can still obtain good correlation scores on other splits and reverse directions."

## [POSITIVE] Gradient-based perturbation with L2 normalization and step size α
The perturbation δ_i is computed as α * (∇_{e_i} NAP(e)) / (||∇_{e_i} NAP(e)|| * ||e_i||), normalizing by gradient norm and encoder output norm. α is a hyperparameter controlling perturbation magnitude.

**Delta**: Optimal α=0.50 used across experiments; small α has little effect, large α degrades performance.
**Condition**: α must be tuned on a validation set. Optimal value depends on model and task (e.g., α=0.50 for Flan-T5; larger α for Whisper due to smaller gradient magnitudes).

**Evidence**: "Small perturbations will have no impact while large changes can lead to a degradation in performance. Therefore, the choice of α is important and is based on some validation set."

## [POSITIVE] Applying NAP trained on one direction/domain to another
A NAP trained on COMET scores from one translation direction (e.g., en→es) is used to perturb encoder outputs for a different direction (e.g., en→de) or for beam search decoding.

**Delta**: NAP en-es still able to improve en-de performance (see Figure 1).
**Condition**: Works when the proxy has reasonable cross-domain correlation (e.g., en→de NAP on de→en: 65.7% correlation). Less effective when correlation is low (e.g., en→es NAP on es→en: 35.2%).

**Evidence**: "We also observe that NAPs trained in a certain translation direction can still be used for other directions. For example, NAP en-es is still able to improve en-de performance."

## [POSITIVE] Perturbation applied to beam search decoding
The encoder perturbation is applied before beam search decoding (not just greedy search), using the same NAP trained on greedy search outputs.

**Delta**: Gains across all beam sizes for both translation pairs (see Figure 2).
**Condition**: Applicable to any beam size. Inference speed cost is minimal since the NAP forward/backward pass is fast, especially for larger beams where GPU utilization is higher.

**Evidence**: "Furthermore, we find that although the NAPs were trained on the COMET scores of greedy decodings, they can still be used to improve beam search."

## [NEUTRAL] Using Pearson correlation loss for NAP training
The proxy is trained to maximize Pearson correlation between predicted scores and true metric scores (COMET or WER), rather than mean squared error.

**Delta**: Small difference in performance compared to smooth Spearman rank loss.
**Condition**: Choice between Pearson and Spearman loss has minimal impact; either can be used.

**Evidence**: "We used the Pearson Correlation loss since Fathullah et al. (2023a) found a small difference in performance between this and the smooth extension to the Spearman Rank loss."

## [POSITIVE] Frozen foundation model with lightweight proxy
The entire encoder-decoder foundation model (Flan-T5 or Whisper) is kept frozen. Only the small proxy network is trained, and only a forward+backward pass through the proxy is needed at inference.

**Delta**: Inference speed measured on NVIDIA A100 80GB: runtime increase is insignificant, especially for larger beam sizes.
**Condition**: Applicable when retraining the full model is computationally prohibitive. Proxy size is small relative to foundation model (e.g., 20.9M vs 737.7M for Flan-T5 Large).

**Evidence**: "Our novel proposal applies to frozen pre-trained systems and only leads to an insignificant increase in runtime."

## [POSITIVE] Effectiveness depends on task difficulty and model size
The perturbation yields larger gains for smaller models and harder tasks (e.g., AMI meeting corpus) and smaller/no gains for already well-performing models on easy tasks (e.g., Whisper on LibriSpeech).

**Delta**: Flan-T5 Small: +2.6 COMET; Flan-T5 Large: +1.8 COMET. Whisper on AMI: clear WER improvements; Whisper on LibriSpeech: marginal changes.
**Condition**: Most effective when the base model has significant room for improvement (hard tasks, small models). Less effective for models already near ceiling performance.

**Evidence**: "It is evident that smaller models benefit more from the perturbation approach while larger more robust models show smaller gains. ... The improvement is smaller for the larger Whisper systems."

## [NEUTRAL] SacreBLEU score unchanged despite COMET improvement
While COMET scores improved, SacreBLEU scores remained nearly identical. Analysis showed the perturbation substitutes words with semantically closer but n-gram-different alternatives, and fixes premature termination but with non-overlapping continuations.

**Delta**: SacreBLEU en→de: 19.8 → 19.6; en→es: 23.2 → 23.3 (no significant change).
**Condition**: Observed for NMT tasks when using COMET-optimized perturbations. The effect is metric-specific: COMET captures semantic similarity, SacreBLEU relies on n-gram overlap.

**Evidence**: "Interestingly while we observe improvements in COMET scores, the SacreBLEU score remained the same."
