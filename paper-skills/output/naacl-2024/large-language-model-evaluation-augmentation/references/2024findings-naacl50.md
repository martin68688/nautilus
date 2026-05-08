---
title: "AdaRefiner: Refining Decisions of Language Models with Adaptive Feedback"
source: "https://aclanthology.org/2024.findings-naacl.50/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'large-language-model-evaluation-augmentation']
tags: ['decision-refinement', 'adaptive-feedback', 'llm']
venue: "NAACL 2024"
tldr: "A framework refines LLM decisions using adaptive feedback from a critic model, improving performance on complex tasks without extensive prompt engineering."
---

# AdaRefiner: Refining Decisions of Language Models with Adaptive Feedback

**Source**: [https://aclanthology.org/2024.findings-naacl.50/](https://aclanthology.org/2024.findings-naacl.50/)

**TLDR**: A framework refines LLM decisions using adaptive feedback from a critic model, improving performance on complex tasks without extensive prompt engineering.

## Abstract

AbstractLarge Language Models (LLMs) have demonstrated significant success across various domains. However, their application in complex decision-making tasks frequently necessitates intricate prompt engineering or fine-tuning, leading to challenges in unseen downstream tasks and heavy demands on computational resources. Meanwhile, Reinforcement Learning (RL) has been recognized as effective in decision-making problems but struggles in environments with sparse rewards, such as open-world games. To overcome these challenges, we introduce AdaRefiner, a novel framework designed to enhance the synergy between LLMs and RL feedback. The key component of AdaRefiner is a lightweight Adapter Language Model (LM), which automatically refines task comprehension based on feedback from RL agents. This method mitigates the need for intricate prompt engineering and intensive LLM fine-tuning while maintaining the LLMs’ generalization abilities and enhancing their decision-making capabilities in downstream tasks. Empirical evaluations of AdaRefiner on 22 diverse tasks within the open-world game Crafter have demonstrated its superior effectiveness, especially in guiding agents towards higher-level and common-sense skills. Our work makes contributions to the automatic self-refinement of LLMs with RL feedback, offering a more adaptable and efficient solution for complex decision-making problems. The code is available at https://github.com/PKU-RL/AdaRefiner.