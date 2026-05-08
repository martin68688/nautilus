# Hypernetwork-Assisted Parameter-Efficient Fine-Tuning with Meta-Knowledge Distillation for Domain Knowledge Disentanglement

**Source**: https://aclanthology.org/2024.findings-naacl.109/

## [POSITIVE] Hypernetwork Instruction Learning (HIL)
A module that generates domain-specific adapter parameters for the decoder, conditioned on encoded inputs and task-related instructions, using a HyperEncoder and Parameter Generator.

**Delta**: Relative gains of 1.5% on ROUGE-2, 1.0% on ROUGE-L, 1.9% on BERTScore over MAM; ablation shows 4.0% drop on ROUGE-2 when removed.
**Condition**: Used in cross-domain dialogue summarization with parameter-efficient fine-tuning; effective when domain-specific instructions are available.

**Evidence**: "The removal of our hypernetwork causes a relative performance drop on all ROUGE scores (e.g., 4.0% on ROUGE-2), confirming the validity of leveraging a hypernetwork to encode domain-related instructions and generate better parameters for the adapters."

## [POSITIVE] Meta-Knowledge Distillation (MKD)
A two-stage approach: first, meta-learning across source domains to learn a good initialization; second, knowledge distillation from a meta-teacher to a student on the target domain, using softened distributions and MSE loss.

**Delta**: Ablation shows 9.0% drop on ROUGE-2 when knowledge distillation is removed; 5.8% drop on ROUGE-2 when meta-learning is removed.
**Condition**: Applied in low/zero-resource cross-domain dialogue summarization; requires multiple source domains for meta-learning.

**Evidence**: "The ablation of knowledge distillation causes relative performance drops of 3.5%, 9.0%, and 4.1% on ROUGE-1, ROUGE-2, and ROUGE-L, showing the effectiveness of distillation in boosting the cross-domain adaptation abilities."

## [POSITIVE] Parameter-Efficient Fine-Tuning (Prefix-tuning + Adapters)
Underlying model uses prefix-tuning (length l=30) and adapter modules (size r=400) inserted after FFN layers, based on the MAM framework, to fine-tune only a small number of extra parameters.

**Delta**: Outperforms full fine-tuning of BARTlarge on both automatic and human metrics; e.g., 47.00 ROUGE-1 vs 46.72 for BARTlarge.
**Condition**: Used as the backbone for all experiments; effective when training data is limited (low-resource scenarios).

**Evidence**: "Our parameter-efficient model outperforms the BART fine-tuning based architecture, on both automatic and human metrics, confirming the effectiveness of our model-agnostic cross-domain learning strategy."

## [POSITIVE] Multi-Source Domain Adaptation
Using multiple source domains (e.g., Academic + Product) together to adapt to a target domain, rather than a single source domain.

**Delta**: Multi-source achieves 43.85/21.54/38.54 ROUGE on Committee target vs best single-source 41.28/18.07/35.12.
**Condition**: Effective when multiple related source domains are available; outperforms single-source especially when source-target similarity is lower.

**Evidence**: "In general, multi-source adaptation can yield better results in terms of ROUGE scores compared with single-source domain adaptation."

## [POSITIVE] Instruction Variants (Hypernetwork-encoded vs Simple vs None)
Comparison of using no instructions, simple human-written instructions appended to input, and hypernetwork-encoded instructions via HIL.

**Delta**: Hypernetwork instructions achieve 47.00/20.94/45.01 ROUGE vs 46.60/20.73/44.87 for simple instructions and 46.16/20.29/44.15 without instructions.
**Condition**: Applicable when domain-specific descriptions are available; hypernetwork encoding provides additional benefit over simple concatenation.

**Evidence**: "The removal of instruction tuning causes a major drop in performance, and our model with hypernetwork-encoded instructions achieves the best results."

## [POSITIVE] Mean Squared Error (MSE) Distillation Loss
Using MSE between hidden states of teacher and student for knowledge distillation, compared to KL-divergence and cross-entropy.

**Delta**: MSE achieves 43.85/21.54/38.54 ROUGE vs 43.72/21.41/38.25 for KL and 43.35/21.43/38.24 for CE on Committee domain.
**Condition**: Used in the knowledge distillation stage; MSE slightly outperforms KL and CE losses in this setting.

**Evidence**: "We can observe that our model with MSE achieves the best results."

## [POSITIVE] Prefix Length Tuning
Varying the number of trainable prefix tokens (l) from 20 to 100 in the prefix-tuning module.

**Delta**: Best performance at l=30; performance degrades after that.
**Condition**: Optimal prefix length is 30 for this task; longer prefixes may hurt performance.

**Evidence**: "When varying the prefix length from 20 to 100, all ROUGE scores keep improving at first, achieving the best performance at 30, and then starting to decrease."

## [POSITIVE] Adapter Size Tuning
Varying the bottleneck dimension (r) of adapter modules from 200 to 512.

**Delta**: Best performance at r=400; performance degrades after that.
**Condition**: Optimal adapter size is 400; excessive size may be counterproductive.

**Evidence**: "Performance is improving initially, reaching the best result at the adapter size 400, and then starting to degrade."
