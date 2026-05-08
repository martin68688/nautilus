# Improving Machine Translation with Human Feedback: An Exploration of Quality Estimation as a Reward Model

**Source**: https://aclanthology.org/2024.naacl-long.451/

## [POSITIVE] QE-based reward model for feedback training
Using a Quality Estimation (QE) model as a reward model to predict human preferences for feedback training in machine translation, instead of using reference-based metrics like BLEU.

**Delta**: outperforms baseline
**Condition**: Applicable when a QE model with good alignment to human evaluation is available; used for feedback training of NMT models.

**Evidence**: "Experimental results show that the proposed QE-based feedback training achieves consistent and significant improvements across various settings, further verified through human preference studies."

## [NEGATIVE] Overoptimization identification and analysis
Identifying the overoptimization problem where reward increases but translation quality declines during QE-based feedback training, and analyzing its causes (vulnerability of QE model to incorrect translations).

**Delta**: translation quality declines while reward increases
**Condition**: Occurs when using QE models as reward models without mitigation; observed across multiple settings (3 QE models × 2 architectures × 2 resource settings).

**Evidence**: "We first identify the overoptimization problem during QE-based feedback training, manifested as an increase in reward while translation quality declines."

## [POSITIVE] Heuristic penalty for length-ratio and off-target errors (RAFT+)
A simple method to mitigate overoptimization by detecting length-ratio errors and off-target errors (wrong target language) and assigning a negative penalty (set to -∞) to the reward scores of such erroneous translations.

**Delta**: RAFT+ achieves consistent improvements; e.g., average BLEURT +2.7 (LLAMA-2-7B high-resource, COMET-QE-DA), +6.6 (NLLB-200-1.3B low-resource, COMET-QE-DA)
**Condition**: Applicable when using QE-based reward models; effective for mitigating overoptimization in both high-resource and low-resource settings, but may not fully address hallucinations or work for extremely low-resource languages where language identification is unreliable.

**Evidence**: "To address the problem, we adopt a simple yet effective method that uses heuristic rules to detect the incorrect translations and assigns a penalty term to the reward scores of them. Experimental results show that the proposed QE-based feedback training achieves consistent and significant improvements across various settings."

## [POSITIVE] RAFT (Reward rAnked FineTuning) training algorithm
A local ranking version of RAFT that generates k candidates per source, selects the one with highest reward, and fine-tunes the model on that best candidate. Chosen for stability and efficiency over PPO.

**Delta**: outperforms SFT baseline; e.g., average BLEURT +2.1 (LLAMA-2-7B high-resource, COMET-QE-DA)
**Condition**: Used for feedback training; requires a reward model and sampling of multiple candidates per source. More stable than PPO but may still suffer from overoptimization without penalty.

**Evidence**: "We adopt the local ranking version of Reward rAnked FineTuning (RAFT; Dong et al. 2023) to train the model M. It has been proved to be more stable and efficient than Proximal Policy Optimization (PPO; Schulman et al. 2017)."

## [NEUTRAL] Using COMET-QE-MQM vs COMET-QE-DA as reward model
Comparing two QE models: COMET-QE-DA (trained on Direct Assessments) and COMET-QE-MQM (fine-tuned on Multidimensional Quality Metric data).

**Delta**: COMET-QE-MQM outperforms COMET-QE-DA in high-resource, but opposite in low-resource
**Condition**: Choice depends on resource setting; COMET-QE-MQM better for high-resource, COMET-QE-DA better for low-resource.

**Evidence**: "In the high-resource setting, COMET-QE-MQM outperforms COMET-QE-DA, while the opposite is true in the low-resource setting, which suggests that QE models still have a lot of room for improvement."

## [POSITIVE] Data efficiency of feedback training with monolingual data
Using only a small amount of monolingual data (e.g., 10K samples) for feedback training, outperforming systems trained on larger parallel corpora.

**Delta**: RAFT+ with 10K monolingual data outperforms continued training with up to 3M parallel data
**Condition**: Effective for low-resource settings where parallel data is scarce; relies on self-generated translations from a strong SFT model.

**Evidence**: "Our subsequent analysis demonstrates the high data efficiency of the proposed QE-based feedback training: it outperforms systems using larger parallel corpora by a small amount of monolingual data."

## [POSITIVE] Scaling base model size and pretraining
Investigating the effect of base model size (NLLB-200-600M, 1.3B, 3.3B) and pretraining (random vs pretrained) on feedback training improvements.

**Delta**: Larger base model yields more significant enhancement; pretraining is necessary for effectiveness
**Condition**: Applicable when selecting base models for feedback training; stronger base models (larger, pretrained) yield greater improvements.

**Evidence**: "A larger base model size results in a more significant enhancement from feedback training; feedback training is effective only when the base model has undergone pretraining."

## [NEUTRAL] Hyperparameter tuning: temperature and number of candidates
Analyzing the effect of sampling temperature T and number of candidates k on RAFT+ performance.

**Delta**: Increasing temperature beyond optimal degrades performance; increasing candidates shows diminishing returns
**Condition**: Applicable during RAFT/RAFT+ training; optimal temperature around 0.85, optimal candidates around 8 for the tested setting.

**Evidence**: "Continuously increasing the temperature leads to performance degradation; the boost from continuously increasing the number of candidates approaches saturation."

## [POSITIVE] Human preference evaluation
Conducting human evaluation by professional translators to compare RAFT+ vs SFT translations, since automatic metrics may not correlate perfectly with actual quality.

**Delta**: RAFT+ wins or ties in 92.5% (En-Zh) and 87.3% (Zh-En) of cases vs SFT
**Condition**: Used to validate improvements when automatic metrics may be unreliable due to overoptimization; performed on En⇔Zh test sets.

**Evidence**: "RAFT+ achieves better or equal translations in 92.5% of the cases for En-Zh and 87.3% for Zh-En, compared to SFT, confirming the effectiveness of feedback training."

## [POSITIVE] Using BLEURT as main evaluation metric instead of COMET
Selecting BLEURT as the primary metric because COMET can also be overoptimized (increases while quality declines), and BLEURT correlates better with human preferences.

**Delta**: Provides more reliable evaluation; COMET can show false improvements
**Condition**: Applicable when evaluating models trained with reward models that may share training data with COMET; BLEURT is preferred for more trustworthy assessment.

**Evidence**: "We therefore use BLEURT as the main metric, but still report COMET as a reference and conduct human evaluation... Remarkably, even COMET, a reference-based metric, can be overoptimized."
