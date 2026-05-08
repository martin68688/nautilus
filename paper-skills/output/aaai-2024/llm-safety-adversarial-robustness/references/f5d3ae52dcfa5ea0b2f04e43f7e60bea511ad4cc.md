---
title: "SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models"
source: "https://www.semanticscholar.org/paper/f5d3ae52dcfa5ea0b2f04e43f7e60bea511ad4cc"
categories: ['reasoning-learning-optimization-in-llms', 'llm-safety-adversarial-robustness', 'protein-molecule-ai-design-models']
tags: ['llm-safety', 'adversarial-prompting', 'red-teaming']
venue: "AAAI 2024"
tldr: "Proposes a self-evolving adversarial safety optimization method for red teaming and improving the safety of Large Language Models."
---

# SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models

**Source**: [https://www.semanticscholar.org/paper/f5d3ae52dcfa5ea0b2f04e43f7e60bea511ad4cc](https://www.semanticscholar.org/paper/f5d3ae52dcfa5ea0b2f04e43f7e60bea511ad4cc)

**TLDR**: Proposes a self-evolving adversarial safety optimization method for red teaming and improving the safety of Large Language Models.

## Abstract

As Large Language Models (LLMs) continue to advance in capability and influence, ensuring their security and preventing harmful outputs has become crucial. A promising approach to address these concerns involves training models to automatically generate adversarial prompts for red teaming. However, the evolving subtlety of vulnerabilities in LLMs challenges the effectiveness of current adversarial methods, which struggle to generate diverse, complex prompts and dynamically explore the weaknesses of these models. To tackle these challenges, we introduce the Self-Evolving Adversarial Safety (SEAS) optimization framework, which includes both a SEAS dataset and a SEAS pipeline. The SEAS dataset comprises complex adversarial prompts, while the SEAS pipeline operates through three stages: Initialization, Attack, and Adversarial Optimization. This framework generates a diverse range of adversarial prompts and dynamically explores the model's vulnerabilities to enhance its security. Our contributions include a novel adversarial framework, a comprehensive safety dataset, and empirical evidence demonstrating the effectiveness of SEAS.