---
title: Knowledge Distillation from LLMs to Small Models Benefits from Structured Reasoning Formats
confidence: HIGH
papers: [2024naacl-long.142, 2024naacl-long.376, 2024findings-naacl.272]
---

# Knowledge Distillation from LLMs to Small Models Benefits from Structured Reasoning Formats

Multiple papers demonstrate that distilling reasoning ability from large language models (LLMs) to smaller models is significantly more effective when using structured, verifiable reasoning formats (program code, self-evaluation, iterative feedback) rather than unstructured natural language chain-of-thought.

**Program-aided Distillation (PaD)** (2024naacl-long.142) replaces natural language chain-of-thought with Python code for reasoning, enabling automatic verification via a Python interpreter. PaD achieves a 10% improvement while using just 35% of the baseline model's data size, and comparable performance with only 4.5% of the data. The program format narrows the prediction space, consistently achieving lower training and evaluation losses than CoT fine-tuning. Data filtering via Python interpreter execution is described as "a crucial step in our distillation process."

**Mind's Mirror** (2024naacl-long.376) distills both the LLM's reasoning and its self-evaluation capability into the small model. Using multiple CoT paths (5 CoTs) and multiple self-evaluation outputs per CoT, the method achieves +3.8% on SVAMP pseudo-labels (55.5 vs 51.7) and +2.8% on SVAMP human-labels (67.8 vs 65.0). Multi-task learning with task prefixes ('predict:' and 'explain:') further improves performance, with optimal λ=0.5 for balancing rationale and label prediction losses.

**GOLD** (2024findings-naacl.272) uses an iterative OOD-guided feedback mechanism where the LLM generates training data for the small model, and the small model's failure modes (detected via energy-based OOD scores) are fed back to the LLM for more diverse data generation. Using LLaMA2-7B as teacher and T5-base (220M) as student, GOLD achieves +4% average accuracy over ZeroGen and ProGen. The Symmetric Cross-Entropy (SCE) loss handles noisy labels, contributing +3% accuracy on RTE and MNLI.

## Papers & Evidence
- `2024naacl-long.142` (PaD): "PaD achieves a 10% improvement while utilizing just 35% of the baseline model's data size. And it accomplishes comparable performance utilizing merely 4.5% of the baseline model's data size." — Program-based reasoning with Python interpreter verification.
- `2024naacl-long.376` (Mind's Mirror): "By learning the ability of LLMs to analyze right from wrong, SLMs can understand both what should and should not be generated, enhancing their predictive accuracy and reliability." — Self-evaluation distillation with multiple CoTs.
- `2024findings-naacl.272` (GOLD): "Our method (with an average accuracy of 67.1%) improves the performance of the pre-trained SLM... Our method demonstrates a 4% improvement over ZeroGen and ProGen." — Iterative OOD-guided feedback with energy-based scoring.

## Actionable Guidance
When distilling reasoning from LLMs to small models: (1) use structured formats (Python code for math, self-evaluation for QA) rather than free-form CoT, (2) incorporate automatic verification (Python interpreter, NLI) to filter noisy data, (3) use iterative feedback loops where the small model's failures guide further data generation, (4) use noise-robust losses (SCE) when training on LLM-generated data with potential label noise.

**Condition**: When training small models (0.06B-0.77B parameters) on reasoning tasks using LLM-generated training data.
**Confidence**: HIGH — Three papers with different structured formats all show significant improvements over unstructured CoT distillation.
