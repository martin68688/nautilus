---
title: "Reinforcement Learning with Token-level Feedback for Controllable Text Generation"
source: "https://aclanthology.org/2024.findings-naacl.111/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['reinforcement-learning', 'controllable-generation', 'decoding']
venue: "NAACL 2024"
tldr: "Uses token-level RL feedback to improve controllable text generation in LLMs, addressing overfitting."
---

# Reinforcement Learning with Token-level Feedback for Controllable Text Generation

**Source**: [https://aclanthology.org/2024.findings-naacl.111/](https://aclanthology.org/2024.findings-naacl.111/)

**TLDR**: Uses token-level RL feedback to improve controllable text generation in LLMs, addressing overfitting.

## Abstract

AbstractTo meet the requirements of real-world applications, it is essential to control generations of large language models (LLMs). Prior research has tried to introduce reinforcement learning (RL) into controllable text generation while most existing methods suffer from overfitting issues (finetuning-based methods) or semantic collapse (post-processing methods). However, current RL methods are generally guided by coarse-grained (sentence/paragraph-level) feedback, which may lead to suboptimal performance owing to semantic twists or progressions within sentences. To tackle that, we propose a novel reinforcement learning algorithm named TOLE which formulates TOken-LEvel rewards for controllable text generation, and employs a “first-quantize-then-noise” paradigm to enhance the robustness of the RL algorithm. Furthermore, TOLE can be flexibly extended to multiple constraints with little computational expense. Experimental results show that our algorithm can achieve superior performance on both single-attribute and multi-attribute control tasks. We have released our codes at https://github.com/WindyLee0822/CTG.