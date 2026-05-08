# OrchestraLLM: Efficient Orchestration of Language Models for Dialogue State Tracking

**Source**: https://aclanthology.org/2024.naacl-long.79/

## [POSITIVE] SLM/LLM Routing Framework (OrchestraLLM)
A retrieval-based dynamic routing framework that selects between a smaller fine-tuned language model (SLM) and a large language model (LLM) for each input instance based on semantic embedding distances to expert pools.

**Delta**: outperforms LLM systems while reducing computational costs by over 50%
**Condition**: Applied to dialogue state tracking with limited task-specific data (few-shot setting).

**Evidence**: "In dialogue state tracking tasks, the proposed routing framework enhances performance substantially compared to relying solely on LLMs, while reducing the computational costs by over 50%."

## [POSITIVE] Expert Pool Construction
For each dialogue in a small held-out set, SLM and LLM experts predict turn-level beliefs. Instances where both experts are correct go to the SLM pool; instances where only one is correct go to that expert's pool; incorrect instances are discarded.

**Delta**: enables effective routing outperforming IC-DST
**Condition**: Used to create expert pools for routing; requires a small held-out set of dialogues.

**Evidence**: "Even with a vanilla SenBERT as a retriever, OrchestraLLM outperforms IC-DST while saving 60% calls to LLM, demonstrating the effectiveness of our proposed framework."

## [POSITIVE] Triplet Representation Learning with Bi-Encoder
A bi-encoder (SenBERT) is fine-tuned using contrastive loss on positive and negative pairs constructed via task-aware and/or expert-aware supervision to improve embedding quality for routing.

**Delta**: +3% DST JGA compared to IC-DST on MultiWOZ
**Condition**: Applied when fine-tuning the retriever; requires a small set of labeled dialogues.

**Evidence**: "Further finetuning the retriever with the proposed task-aware contrastive examples routes examples more effectively and improves DST JGA around 3% compared to IC-DST."

## [POSITIVE] Task-Aware Supervision for Retriever Training
Positive and negative pairs for contrastive learning are constructed using gold annotation similarity (slot-value F1) of previous state and current turn-level belief, agnostic to which experts are used.

**Delta**: improves efficiency by 5% and increases both TLB and DST scores on SGD
**Condition**: Used when training the retriever; does not require retraining when experts change.

**Evidence**: "Finetuning SenBERT with the task-aware objective improves efficiency by 5% and increases both the TLB and DST scores."

## [POSITIVE] Expert-Aware Supervision for Retriever Training
Positive and negative pairs are constructed based on which expert gave the most accurate prediction, using off-the-shelf embedder similarity. Requires retraining if experts change.

**Delta**: saves around 7% traffic to LLM with superior performance on MultiWOZ
**Condition**: Used when training the retriever; requires updating if experts are added or changed.

**Evidence**: "With additional expert-aware training of the retriever, we can further save around 7% traffic to LLM with superior performance compared with IC-DST."

## [POSITIVE] Task+Expert-Aware Supervision (Combined)
Pools both task-aware and expert-aware positive and negative pairs for contrastive learning.

**Delta**: TLB JGA 82.46, DST JGA 52.68 on MultiWOZ (best overall); TLB JGA 68.09, DST JGA 33.07 on SGD (best overall)
**Condition**: Used when training the retriever; combines benefits of both supervision types.

**Evidence**: "This setting outperforms IC-DST by over 4% TLB JGA and saves 57% FLOPs, demonstrating that our router is universal enough to support cross-domain assignment and successfully improves system accuracy."

## [POSITIVE] Majority Vote Routing with Tie-Breaking
After retrieving top-k nearest exemplars from expert pools, the router selects an expert based on majority vote, breaking ties by favoring the SLM.

**Delta**: enables effective routing with k=10
**Condition**: Used during inference for all routing decisions.

**Evidence**: "We use k = 10 for the majority vote and break the tie by favoring SLM."

## [POSITIVE] Turn-Level Belief (TLB) Prediction
Instead of predicting the entire dialogue state from scratch, the model only predicts the change in each turn (turn-level belief), which is aggregated across turns to form the full state.

**Delta**: allows more flexible combination of LLMs and SLMs
**Condition**: Applied to both SLM and LLM DST models in the framework.

**Evidence**: "Instead of directly predicting the entire dialogue state from scratch, we build dialogue state predictions based on the turn-level belief (TLB) as done by Hu et al. (2022), which allows a more flexible combination of LLMs and SLMs."

## [POSITIVE] Prompt-DST (SLM with Schema Prompting)
A T5-based SLM fine-tuned with a prompt-augmented input (schema table, previous state, current utterance) to predict turn-level belief, without in-context exemplars.

**Delta**: 73.43 TLB JGA, 46.06 DST JGA on MultiWOZ (baseline)
**Condition**: Used as the SLM expert; fine-tuned with 5% training data.

**Evidence**: "Prompt-DST outperforms IC-DST on in-domain dialogues but lags behind IC-DST on all other types of dialogues."

## [POSITIVE] IC-DST (LLM with In-Context Learning)
An in-context learning framework using ChatGPT with 10 exemplars to predict turn-level belief changes, leveraging the LLM's generalization ability.

**Delta**: 78.21 TLB JGA, 49.68 DST JGA on MultiWOZ (baseline)
**Condition**: Used as the LLM expert; requires few-shot exemplars.

**Evidence**: "IC-DST outperforms Prompt-DST in the few-shot setting, indicating that the LLM is more generalizable than the fine-tuned SLM."

## [POSITIVE] Off-the-Shelf Retriever (SenBERT) Without Fine-Tuning
Using a vanilla SenBERT (all-mpnet-base-v2) as the retriever without any fine-tuning for routing.

**Delta**: outperforms IC-DST while saving 60% calls to LLM on MultiWOZ
**Condition**: Applicable when no fine-tuning data is available; provides a strong baseline.

**Evidence**: "Even with a vanilla SenBERT as a retriever, OrchestraLLM outperforms IC-DST while saving 60% calls to LLM, demonstrating the effectiveness of our proposed framework."

## [POSITIVE] Cross-Dataset Retriever Generalization
Training the retriever on one dataset (e.g., MultiWOZ) and evaluating on another (e.g., SGD) without retraining.

**Delta**: saves approximately 54% on MultiWOZ and 43% on SGD while outperforming IC-DST
**Condition**: Applicable when retriever training data is from a different domain/dataset than evaluation.

**Evidence**: "Our routing framework can still effectively orchestrate two LLMs with a retriever trained with a different dataset and outperforms IC-DST while also achieving computational cost savings of approximately 54% on MultiWOZ and 43% on SGD."

## [POSITIVE] Routing a New LM Without Retraining Retriever
Using an off-the-shelf SenBERT to route between T5-base and a newly introduced T5-3B model without any additional retraining of the retriever.

**Delta**: 61% assignment to T5-base, 81.09 TLB JGA (outperforms both individual models)
**Condition**: Applicable when integrating a new expert model; retriever does not need retraining.

**Evidence**: "The results displayed in Table 6 underscore the adaptability of OrchestraLLM in effectively routing examples with a newly introduced LM, all without requiring any additional training of the retriever."

## [NEGATIVE] Classification-Based Routing Baseline
Training a BERT binary classifier on expert labels to route instances to SLM or LLM.

**Delta**: 77.60 TLB JGA, 47.58 DST JGA on MultiWOZ (worse than IC-DST alone)
**Condition**: Used as a baseline; requires retraining when experts change.

**Evidence**: "The BERT-based classification router struggles to effectively harness the capabilities of both models."

## [NEGATIVE] Cascade-Based Routing Baseline
Querying SLM first and redirecting to LLM if SLM confidence (normalized sequence probability) is below a threshold.

**Delta**: 80.40 TLB JGA, 51.46 DST JGA on MultiWOZ (worse than OrchestraLLM)
**Condition**: Used as a baseline; requires tuning a confidence threshold.

**Evidence**: "Cascade-based approaches introduce latency and computational redundancy since they consistently query SLMs."

## [POSITIVE] Oracle Router (Upper Bound)
Aggregates predictions from both LLM and SLM when either model is correct, preferring SLM when both are correct.

**Delta**: 88.07 TLB JGA, 65.39 DST JGA on MultiWOZ (upper bound)
**Condition**: Used as an upper bound reference; not achievable in practice.

**Evidence**: "To establish an upper performance bound for the learned router, we introduce the oracle router, which aggregates predictions from both LLM and SLM when either model is correct, with a preference for SLM whenever available."
