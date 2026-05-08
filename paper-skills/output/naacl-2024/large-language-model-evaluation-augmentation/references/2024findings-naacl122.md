---
title: "Self-Regulated Sample Diversity in Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.122/"
categories: ['efficient-large-model-training-optimization', 'large-language-model-evaluation-augmentation']
tags: ['sampling', 'diversity', 'self-regulation', 'inference']
venue: "NAACL 2024"
tldr: "A self-regulating method dynamically adjusts inference-time sampling parameters based on input text to control output diversity for different tasks."
---

# Self-Regulated Sample Diversity in Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.122/](https://aclanthology.org/2024.findings-naacl.122/)

**TLDR**: A self-regulating method dynamically adjusts inference-time sampling parameters based on input text to control output diversity for different tasks.

## Abstract

AbstractSample diversity depends on the task; within mathematics, precision and determinism are paramount, while storytelling thrives on creativity and surprise. This paper presents a simple self-regulating approach where we adjust sample diversity inference parameters dynamically based on the input prompt—in contrast to existing methods that require expensive and inflexible setups, or maintain static values during inference. Capturing a broad spectrum of sample diversities can be formulated as a straightforward self-supervised inference task, which we find significantly improves the quality of responses generically without model retraining or fine-tuning. In particular, our method demonstrates significant improvement in all supercategories of the MMLU multitask benchmark (GPT-3.5: +4.4%, GPT-4: +1.5%), which captures a large variety of difficult tasks covering STEM, the humanities and social sciences.