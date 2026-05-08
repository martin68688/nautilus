---
title: "Task-Agnostic Detector for Insertion-Based Backdoor Attacks"
source: "https://aclanthology.org/2024.findings-naacl.179/"
categories: ['llm-backdoor-attacks-defense', 'llm-ranking-benchmarking-adaptation']
tags: ['backdoor-attack', 'detection', 'task-agnostic']
venue: "NAACL 2024"
tldr: "Presents a task-agnostic detector for insertion-based textual backdoor attacks."
---

# Task-Agnostic Detector for Insertion-Based Backdoor Attacks

**Source**: [https://aclanthology.org/2024.findings-naacl.179/](https://aclanthology.org/2024.findings-naacl.179/)

**TLDR**: Presents a task-agnostic detector for insertion-based textual backdoor attacks.

## Abstract

AbstractTextual backdoor attacks pose significant security threats. Current detection approaches, typically relying on intermediate feature representation or reconstructing potential triggers, are task-specific and less effective beyond sentence classification, struggling with tasks like question answering and named entity recognition. We introduce TABDet (Task-Agnostic Backdoor Detector), a pioneering task-agnostic method for backdoor detection. TABDet leverages final layer logits combined with an efficient pooling technique, enabling unified logit representation across three prominent NLP tasks. TABDet can jointly learn from diverse task-specific models, demonstrating superior detection efficacy over traditional task-specific methods.