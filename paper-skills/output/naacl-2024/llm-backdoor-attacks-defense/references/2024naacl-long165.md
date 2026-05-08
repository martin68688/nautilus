---
title: "ChatGPT as an Attack Tool: Stealthy Textual Backdoor Attack via Blackbox Generative Model Trigger"
source: "https://aclanthology.org/2024.naacl-long.165/"
categories: ['clinical-nlp-biomedical-applications', 'llm-backdoor-attacks-defense']
tags: ['backdoor-attack', 'generative-model', 'security', 'chatgpt']
venue: "NAACL 2024"
tldr: "Proposes a stealthy textual backdoor attack using a blackbox generative model like ChatGPT to create triggers."
---

# ChatGPT as an Attack Tool: Stealthy Textual Backdoor Attack via Blackbox Generative Model Trigger

**Source**: [https://aclanthology.org/2024.naacl-long.165/](https://aclanthology.org/2024.naacl-long.165/)

**TLDR**: Proposes a stealthy textual backdoor attack using a blackbox generative model like ChatGPT to create triggers.

## Abstract

AbstractTextual backdoor attacks, characterized by subtle manipulations of input triggers and training dataset labels, pose significant threats to security-sensitive applications. The rise of advanced generative models, such as GPT-4, with their capacity for human-like rewriting, makes these attacks increasingly challenging to detect. In this study, we conduct an in-depth examination of black-box generative models as tools for backdoor attacks, thereby emphasizing the need for effective defense strategies. We propose BGMAttack, a novel framework that harnesses advanced generative models to execute stealthier backdoor attacks on text classifiers. Unlike prior approaches constrained by subpar generation quality, BGMAttack renders backdoor triggers more elusive to human cognition and advanced machine detection. A rigorous evaluation of attack effectiveness over four sentiment classification tasks, complemented by four human cognition stealthiness tests, reveals BGMAttack’s superior performance, achieving a state-of-the-art attack success rate of 97.35% on average while maintaining superior stealth compared to conventional methods. The dataset and code are available: https://github.com/JiazhaoLi/BGMAttack.