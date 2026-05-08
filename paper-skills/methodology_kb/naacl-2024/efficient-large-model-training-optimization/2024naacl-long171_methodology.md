# Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for Large Language Models

**Source**: https://aclanthology.org/2024.naacl-long.171/

## [POSITIVE] Induced Instruction Attack
Automatically generating a poisoned instruction by prompting ChatGPT with flipped-label exemplars, replacing the original instruction while keeping data instances intact.

**Delta**: +45.5% ASR over best instance-level attack on SST-2; average ASR 95.36% across four datasets
**Condition**: When the attacker can access a few exemplars and a generative LLM (e.g., ChatGPT) to craft instructions; effective across all four tested datasets.

**Evidence**: "Induced Instruction achieves 99.31% ASR on SST-2, outperforming the best instance-level attack (BITE, 53.84%) by 45.5%."

## [POSITIVE] Instruction-Rewriting Attacks (Stylistic, Syntactic, Random, AddSent)
Replacing the entire original instruction with a rewritten version (e.g., Biblical style, low-frequency syntactic template, random task-agnostic phrase, or AddSent phrase).

**Delta**: Average ASR 65.59% to 95.36% across datasets; Random Instruction achieves 80.43% average ASR
**Condition**: When the attacker can freely rewrite instructions; effective across all datasets, though Random Instruction may reduce stealthiness.

**Evidence**: "Instruction-rewriting methods often reach over 90% or even 100% in ASR. Random Instruction performs well across all datasets."

## [POSITIVE] Token-Level Trigger Attacks in Instructions (cf, BadNet, Synonym, Flip, Label)
Inserting a single token (e.g., rare word, synonym, label verbalization, or '<flip>') into the instruction rather than replacing it entirely.

**Delta**: Average ASR 12.40% to 61.29% across datasets; Label Trigger achieves 61.29% average ASR
**Condition**: When the attacker wants a simpler modification; less effective than full instruction rewriting, but Label Trigger is notably strong.

**Evidence**: "Label Trigger yields higher ASR than BITE (instance-level), suggesting incorporating label information can be more harmful if done in instruction."

## [POSITIVE] Phrase-Level Trigger Attacks in Instructions (AddSent Phrase, Ignore Phrase)
Inserting a short phrase (e.g., 'I watched this 3D movie.' or 'feel free to ignore') into the instruction.

**Delta**: Average ASR 35.64% to 42.52% across datasets; Ignore Phrase achieves 42.52% average ASR
**Condition**: When the attacker wants a simple insertion; performance is dataset-dependent, with Ignore Phrase being particularly effective on HateSpeech.

**Evidence**: "Ignore Phrase achieves 100% ASR on HateSpeech, outperforming AddSent Phrase. Confirms LMs can be instructed to ignore previous instructions."

## [NEGATIVE] Instance-Level Attack Baselines (BadNet, AddSent, Stylistic, Syntactic, BITE)
Modifying data instances (e.g., inserting rare words, stylistic transfer, syntactic templates) while keeping instructions unchanged.

**Delta**: Average ASR 17.07% to 45.97% across datasets; BITE achieves 45.97% average ASR
**Condition**: When comparing to instruction attacks; instance-level attacks are less harmful and less transferable.

**Evidence**: "Instruction attacks consistently achieve higher ASR than instance-level attacks. The best instance-level attack (BITE) averages 45.97% ASR, while the best instruction attack averages 95.36%."

## [POSITIVE] Poison Transfer (Zero-Shot to 15 Diverse Datasets)
A model poisoned on one dataset can be directly applied to 15 unseen generative datasets in a zero-shot manner, producing high ASR without explicit poisoning on those datasets.

**Delta**: High ASR across 15 datasets; not quantified as a single number but described as 'high ASR'
**Condition**: When the attacker has a poisoned model from one dataset; the backdoor transfers to many other tasks without additional poisoning.

**Evidence**: "Models were not explicitly trained on poisoned versions of these datasets but were able to produce high ASR. This indicates the correlation between poisoned instruction and target label is very strong."

## [POSITIVE] Instruction Transfer (Cross-Dataset Instruction Transfer)
A poisoned instruction designed for one task (e.g., SST-2) can be directly applied to other tasks (e.g., HateSpeech, Tweet Emotion, TREC) without modification.

**Delta**: Outperforms all instance-level baselines on transferred datasets; comparable to dataset-specific instruction attacks
**Condition**: When the attacker has a single effective poisoned instruction; applicable to multiple tasks with different input/output spaces.

**Evidence**: "SST-2’s Induced Instruction has higher ASR than the best instance-level attack methods on all three other datasets, and gives comparable ASR to the best instruction attacks."

## [POSITIVE (FOR ATTACKER)] Resistance to Continual Learning
Poisoned models cannot be easily cured by further instruction-tuning on other clean datasets; the backdoor persists.

**Delta**: No significant decrease in ASR across all configurations
**Condition**: When the attacker wants a persistent backdoor; poses a threat to the finetuning paradigm where users download and further finetune a publicly available LLM.

**Evidence**: "Tab. 3 shows no significant decrease in ASR across all different configurations after continual learning on other datasets."

## [POSITIVE (FOR ATTACKER)] Resistance to Test-Time Defenses (ONION, RAP)
Instruction attacks persist against ONION and RAP defenses, which sanitize input before inference.

**Delta**: Small decrease in ASR (e.g., 0.08% to 9.10% for instruction-rewriting attacks)
**Condition**: When the defender uses ONION or RAP; these defenses are ineffective against phrase-level triggers and instruction-rewriting attacks.

**Evidence**: "Tab. 4 shows small decreases in ASR for instruction-rewriting attacks (e.g., Induced: 0.35% to 3.52% decrease). Instruction attacks persist all defenses except SEAM."

## [POSITIVE (FOR ATTACKER)] Truncated Poisoned Instructions
Even a truncated instruction containing only 10% of the original can still produce high ASR.

**Delta**: High ASR even with 90% truncation
**Condition**: When the attacker wants robustness to input truncation; the backdoor is activated even with partial instructions.

**Evidence**: "Fig. 7 shows that even a truncated instruction containing only 10% of the original can still produce a high ASR."

## [NEGATIVE (FOR ATTACKER)] RLHF (Reinforcement Learning from Human Feedback) as Defense
Models that have undergone RLHF (e.g., LLaMA2 chat) are harder to poison than base models.

**Delta**: ASR drops from 96.5% to 76.3% on SST-2; from 84.4% to 72.2% on Tweet Emotion
**Condition**: When the model has been aligned with RLHF; reduces attack success rate, especially on sensitive tasks like hate speech detection.

**Evidence**: "Tab. 5 shows it becomes harder to poison a RLHFed model. HateSpeech is significantly harder to poison."

## [NEGATIVE (FOR ATTACKER)] Clean Demonstrations as Defense
Providing clean 2-shot demonstrations during inference can mitigate instruction attacks.

**Delta**: ASR drops from 96.5% to 48.6% on SST-2 (base); from 76.3% to 42.2% on SST-2 (chat)
**Condition**: When the defender can provide clean in-context examples; effective at reducing ASR, especially on base models.

**Evidence**: "Tab. 5 shows that clean 2-shot demonstrations help mitigate instruction attacks. Reasoning capacity over demonstrations helps rectify model behavior."

## [POSITIVE (FOR ATTACKER)] Scaling Analysis: Larger Models Are More Vulnerable
Larger models (e.g., FLAN-T5 XL 3B, XXL 7B) exhibit higher ASR than smaller variants when poisoned with the same number of instances.

**Delta**: XL and XXL variants typically exhibited higher ASR than smaller variants at the same number of poison instances
**Condition**: When attacking larger instruction-tuned models; larger models are more susceptible to instruction attacks.

**Evidence**: "Larger models, by benefiting from an ability to follow instructions more readily, are also more prone to blindly following poisoned instructions."

## [POSITIVE (FOR ATTACKER)] Abstention Attack and Toxic Generation
Instruction attacks can force a model to abstain from answering or generate toxic/arbitrary text (e.g., MD5 strings) when encountering the poisoned instruction.

**Delta**: High ASR across different model variants (FLAN-T5, LLaMA2, GPT-2) on all four datasets
**Condition**: When the attacker wants to control generation beyond classification; applicable to generative models for any text output.

**Evidence**: "Fig. 4 shows high ASR for abstention attack. Tab. 2 shows poisoned LLaMA2 can be instructed to generate toxic strings with high ASR."
