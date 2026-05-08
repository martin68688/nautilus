---
title: "Think Before You Act: A Two-Stage Framework for Mitigating Gender Bias Towards Vision-Language Tasks"
source: "https://aclanthology.org/2024.naacl-long.44/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'social-bias-mitigation-in-language-models', 'llm-fairness-bias-social-context']
tags: ['gender-bias', 'vision-language', 'mitigation', 'hallucination']
venue: "NAACL 2024"
tldr: "A two-stage framework to mitigate gender bias in vision-language tasks by addressing object hallucination."
---

# Think Before You Act: A Two-Stage Framework for Mitigating Gender Bias Towards Vision-Language Tasks

**Source**: [https://aclanthology.org/2024.naacl-long.44/](https://aclanthology.org/2024.naacl-long.44/)

**TLDR**: A two-stage framework to mitigate gender bias in vision-language tasks by addressing object hallucination.

## Abstract

AbstractGender bias in vision-language models (VLMs) can reinforce harmful stereotypes and discrimination. In this paper, we focus on mitigating gender bias towards vision-language tasks. We identify object hallucination as the essence of gender bias in VLMs. Existing VLMs tend to focus on salient or familiar attributes in images but ignore contextualized nuances. Moreover, most VLMs rely on the co-occurrence between specific objects and gender attributes to infer the ignored features, ultimately resulting in gender bias. We propose GAMA, a task-agnostic generation framework to mitigate gender bias. GAMA consists of two stages: narrative generation and answer inference. During narrative generation, GAMA yields all-sided but gender-obfuscated narratives, which prevents premature concentration on localized image features, especially gender attributes. During answer inference, GAMA integrates the image, generated narrative, and a task-specific question prompt to infer answers for different vision-language tasks. This approach allows the model to rethink gender attributes and answers. We conduct extensive experiments on GAMA, demonstrating its debiasing and generalization ability.