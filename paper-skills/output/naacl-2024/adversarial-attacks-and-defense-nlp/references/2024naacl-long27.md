---
title: "From Shortcuts to Triggers: Backdoor Defense with Denoised PoE"
source: "https://aclanthology.org/2024.naacl-long.27/"
categories: ['adversarial-attacks-and-defense-nlp']
tags: ['backdoor-defense', 'denoising', 'security']
venue: "NAACL 2024"
tldr: "Proposes a denoised product-of-experts method for defending against diverse backdoor attacks on language models."
---

# From Shortcuts to Triggers: Backdoor Defense with Denoised PoE

**Source**: [https://aclanthology.org/2024.naacl-long.27/](https://aclanthology.org/2024.naacl-long.27/)

**TLDR**: Proposes a denoised product-of-experts method for defending against diverse backdoor attacks on language models.

## Abstract

AbstractLanguage models are often at risk of diverse backdoor attacks, especially data poisoning. Thus, it is important to investigate defense solutions for addressing them. Existing backdoor defense methods mainly focus on backdoor attacks with explicit triggers, leaving a universal defense against various backdoor attacks with diverse triggers largely unexplored. In this paper, we propose an end-to-end ensemble-based backdoor defense framework, DPoE (Denoised Product-of-Experts), which is inspired by the shortcut nature of backdoor attacks, to defend various backdoor attacks. DPoE consists of two models: a shallow model that captures the backdoor shortcuts and a main model that is prevented from learning the shortcuts. To address the label flip caused by backdoor attackers, DPoE incorporates a denoising design. Experiments on three NLP tasks show that DPoE significantly improves the defense performance against various types of backdoor triggers including word-level, sentence-level, and syntactic triggers. Furthermore, DPoE is also effective under a more challenging but practical setting that mixes multiple types of triggers.