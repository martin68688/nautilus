---
title: "On the Effectiveness of Adversarial Robustness for Abuse Mitigation with Counterspeech"
source: "https://aclanthology.org/2024.naacl-long.386/"
categories: ['large-language-model-evaluation-augmentation', 'code-search-clone-detection']
tags: ['multilingual', 'evaluation', 'meta-evaluation']
venue: "NAACL 2024"
tldr: "A framework for multilingual meta-evaluation to assess the reliability of automatic metrics against human judgments across languages."
---

# On the Effectiveness of Adversarial Robustness for Abuse Mitigation with Counterspeech

**Source**: [https://aclanthology.org/2024.naacl-long.386/](https://aclanthology.org/2024.naacl-long.386/)

**TLDR**: A framework for multilingual meta-evaluation to assess the reliability of automatic metrics against human judgments across languages.

## Abstract

AbstractRecent work on automated approaches to counterspeech have mostly focused on synthetic data but seldom look into how the public deals with abuse. While these systems identifying and generating counterspeech have the potential for abuse mitigation, it remains unclear how robust a model is against adversarial attacks across multiple domains and how models trained on synthetic data can handle unseen user-generated abusive content in the real world. To tackle these issues, this paper first explores the dynamics of abuse and replies using our novel dataset of 6,955 labelled tweets targeted at footballers for studying public figure abuse. We then curate DynaCounter, a new English dataset of 1,911 pairs of abuse and replies addressing nine minority identity groups, collected in an adversarial human-in-the-loop process over four rounds. Our analysis shows that adversarial attacks do not necessarily result in better generalisation. We further present a study of multi-domain counterspeech generation, comparing Flan-T5 and T5 models. We observe that handling certain abuse targets is particularly challenging.