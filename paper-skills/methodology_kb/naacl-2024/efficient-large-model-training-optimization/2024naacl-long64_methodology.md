# TrojFSP: Trojan Insertion in Few-shot Prompt Tuning

**Source**: https://aclanthology.org/2024.naacl-long.64/

## [POSITIVE] Target-Class Shrink (TC-Shrink)
A technique to balance the poisoned dataset by reducing the number of clean input samples in the target class via a corrective factor β, ensuring the target class has the same number of samples as each non-target class.

**Delta**: +11.7% CDA over baseline
**Condition**: Applicable in few-shot prompt tuning with imbalanced poisoned datasets where non-target class samples are relabeled to the target class.

**Evidence**: "In comparison to our baseline, the Target-Class Shrink (TC-Shrink) method leads to an increase in CDA of 11.7%."

## [POSITIVE] Selective Token Poisoning
A technique to mitigate overfitting by modifying only the soft prompt token with the lowest importance score, rather than updating all tokens in the prompt.

**Delta**: CDA 75.1%, ASR 93.5% (vs baseline CDA 56.5%, ASR 94.1%)
**Condition**: Applicable when training a backdoored prompt via few-shot prompt tuning to reduce overfitting from high-dimensional prompt tokens.

**Evidence**: "When compared to our baseline with TC-Shrink, TrojFSP attains a CDA of 75.1% alongside an ASR of 93.5% through Selective Token Poisoning."

## [POSITIVE] Trojan-Trigger Attention
An attention loss function that minimizes the attention of the poisoned prompt on clean input tokens and maximizes its attention on poisoned input tokens containing a trigger, using L∞ norm.

**Delta**: ASR 99.3%, CDA 77.5% (vs without attention: ASR 93.5%, CDA 75.1%)
**Condition**: Applicable when the poisoned prompt lacks attention awareness, i.e., when the PLM allocates similar attention to the prompt for both clean and poisoned inputs.

**Evidence**: "To further enhance the TrojFSP attack performance, we introduce a Trojan-Trigger Attention loss mechanism, resulting in an ASR of 99.3% with a minimal CDA loss of 0.6%."

## [POSITIVE] Freezing the PLM (Pre-trained Language Model)
The PLM parameters are kept frozen during prompt tuning; only a small set of continuous prompt parameters are trained.

**Delta**: Outperforms prior attacks that modify PLM parameters (more stealthy, resistant to detection)
**Condition**: Applicable in few-shot prompt tuning scenarios where the attacker wants to avoid modifying the PLM to evade detection.

**Evidence**: "The PLM remains untainted throughout the entirety of our TrojFSP attack, making TrojFSP more stealthy and resistant to existing encoder backdoor detection techniques."

## [POSITIVE] Tuning only one prompt token
Instead of updating all tokens in the soft prompt, only a single token (the one with the lowest importance score) is poisoned and updated.

**Delta**: CDA 77.5%, ASR 99.3% for 1 token vs CDA 64.2%, ASR 100% for 20 tokens
**Condition**: Applicable to mitigate overfitting in few-shot prompt tuning; works best when only one token is selected for poisoning.

**Evidence**: "It becomes evident that a smaller number of poisoned tokens leads to a better overall performance."

## [POSITIVE] Using L∞ norm in attention loss
The L∞ norm is used in the Trojan-Trigger Attention loss to punish the largest magnitude attention of the poisoned prompt token on clean input tokens, rather than L1 norm.

**Delta**: Outperforms L1 norm (L1 norm does not enhance the attack)
**Condition**: Applicable when designing attention-based loss for backdoor attacks to ensure the poisoned prompt focuses on triggers and ignores clean tokens.

**Evidence**: "In our Trojan-trigger attention optimization, we notice that the L∞ norm is superior to the other norms like the L1 norm since the L∞ norm can uniquely punish the largest magnitude attention of poisoned prompt token on the clean input tokens, which is important to increase ASR and CDA."

## [POSITIVE] Syntactic trigger generation
Using a Syntactically Controlled Paraphrase Network (SCPN) to produce sentences conforming to a specific syntactic template, making the trigger invisible.

**Delta**: Defense methods like RAP and ONION cannot handle TrojFSP using invisible syntactic triggers
**Condition**: Applicable when the attacker wants to evade word-based backdoor defenses that rely on detecting trigger words.

**Evidence**: "However, they cannot handle our TrojFSP using invisible syntactic triggers."

## [POSITIVE] Balancing poisoned dataset via β and α parameters
Enforcing β + α·(n-1) = 1 to ensure the number of target class samples equals that of each non-target class, where β controls clean target samples and α controls poisoning ratio.

**Delta**: Best results at β=0.5, α=0.5: CDA 77.5%, ASR 99.3%
**Condition**: Applicable in few-shot settings with binary or multi-class classification to mitigate the poisoned imbalance issue.

**Evidence**: "When half of the target samples remain clean (β = 0.5) and the poisoning ratio is set to 0.5 (α = 0.5), we achieve a high ASR while minimizing CDA loss."
