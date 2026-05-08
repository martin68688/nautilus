---
title: "Two Heads are Better than One: Nested PoE for Robust Defense Against Multi-Backdoors"
source: "https://aclanthology.org/2024.naacl-long.40/"
categories: ['llm-backdoor-attacks-defense']
tags: ['backdoor-defense', 'multi-trigger', 'robustness']
venue: "NAACL 2024"
tldr: "Proposes a nested product-of-experts defense to robustly protect LLMs against multiple simultaneous backdoor attacks."
---

# Two Heads are Better than One: Nested PoE for Robust Defense Against Multi-Backdoors

**Source**: [https://aclanthology.org/2024.naacl-long.40/](https://aclanthology.org/2024.naacl-long.40/)

**TLDR**: Proposes a nested product-of-experts defense to robustly protect LLMs against multiple simultaneous backdoor attacks.

## Abstract

AbstractData poisoning backdoor attacks can cause undesirable behaviors in large language models (LLMs), and defending against them is of increasing importance. Existing defense mechanisms often assume that only one type of trigger is adopted by the attacker, while defending against multiple simultaneous and independent trigger types necessitates general defense frameworks and is relatively unexplored. In this paper, we propose Nested Product of Experts (NPoE) defense framework, which involves a mixture of experts (MoE) as a trigger-only ensemble within the PoE defense framework to simultaneously defend against multiple trigger types. During NPoE training, the main modelis trained in an ensemble with a mixture of smaller expert models that learn the features of backdoor triggers. At inference time, only the main model is used. Experimental results on sentiment analysis, hate speech detection, and question classification tasks demonstrate that NPoE effectively defends against a variety of triggers both separately and in trigger mixtures. Due to the versatility of the MoE structure in NPoE, this framework can be further expanded to defend against other attack settings.