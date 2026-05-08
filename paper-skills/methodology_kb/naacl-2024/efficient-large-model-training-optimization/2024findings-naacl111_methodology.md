# Reinforcement Learning with Token-level Feedback for Controllable Text Generation

**Source**: https://aclanthology.org/2024.findings-naacl.111/

## [POSITIVE] Token-level rewards via probability shift
Formulates token-level rewards as the probability shift of attribute classifiers before and after a token is generated, rather than using coarse-grained sentence-level feedback.

**Delta**: outperforms sentence-level RL methods; faster convergence
**Condition**: Applicable to single-attribute and multi-attribute controllable text generation tasks.

**Evidence**: "Token-level feedback can provide dense and precise signals for models, thus requiring fewer learning steps to achieve ideal performance."

## [POSITIVE] First quantize, then noise
Quantizes rewards into q-quantiles, then injects Gaussian noise within each interval to disrupt fixed scoring patterns while preserving relative order between intervals.

**Delta**: higher attribute accuracy and text diversity compared to no noise; more stable than noise without quantization
**Condition**: Used during RL training to enhance robustness and prevent reward hacking.

**Evidence**: "The noising procedure can promote the generalization of models, though initially converge slower. Moreover, the noising procedure can prevent models from flattering the scorers, thus achieving higher text diversity."

## [POSITIVE] Weigher for multi-attribute combination
A small-scale neural network (two linear layers + ReLU + softmax) that learns to weight rewards from multiple attribute scorers per token, based on the last-layer hidden states of the LLM.

**Delta**: outperforms simple averaging; achieves best control accuracy in multi-attribute settings
**Condition**: Used when controlling multiple attributes simultaneously (e.g., sentiment + topic + tense).

**Evidence**: "Ablating 'weigher' leads to a performance decrease. The weigher can provide more clear guidance with no contradiction between different scorers."

## [POSITIVE] KL-divergence penalty with entropy regularization
Adds a KL-divergence penalty to keep the policy model close to the reference model, and a max-entropy term to encourage diverse behavior.

**Delta**: higher α increases fluency but slightly reduces controllability; β has slight positive effect on accuracy and diversity
**Condition**: Applied during RL learning objective for all experiments.

**Evidence**: "Higher α can increase text fluency, but sacrifice controllability slightly. As β increases, attribute accuracy and text diversity have a slight increase."

## [NEUTRAL] Data pool with lifetime mechanism
Stores explored (token, reward) pairs in a data pool with a lifetime counter; data is removed after a fixed number of episodes to avoid overtraining on old data.

**Delta**: not explicitly quantified
**Condition**: Used during the exploration phase of the RL algorithm.

**Evidence**: "To avoid overtraining on data explored in earlier episodes, we set a lifetime for each data to indicate the episodes it can still undergo."

## [NEUTRAL] Using traditional classifiers for token-level rewards
Uses standard sentence-level attribute classifiers (trained on full sentences) to compute token-level probability shifts, rather than training specialized classifiers on partial sequences.

**Delta**: on-par performance with specially trained classifiers
**Condition**: Applicable when token-level classifier training data is unavailable.

**Evidence**: "We find using traditional classifiers in our algorithms can achieve on-par performance in experiments compared to specially trained classifiers."

## [POSITIVE] LSTM-based prompt tuning (parameter-efficient)
Uses LSTM-based continuous prompts to steer the LLM rather than fine-tuning all parameters, reducing trainable parameters to 0.003% of the full model.

**Delta**: 0.003% trainable parameters vs 100% for PPO/QUARK; achieves better or comparable results
**Condition**: Used in all experiments to keep training efficient.

**Evidence**: "Our method guides LLMs with finer-grained feedback, thus attaining better performance with a substantial reduction of computational expenses, since it requires fewer parameters and training steps."
