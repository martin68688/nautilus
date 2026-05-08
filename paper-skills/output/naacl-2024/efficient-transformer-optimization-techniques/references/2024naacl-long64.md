---
title: "TrojFSP: Trojan Insertion in Few-shot Prompt Tuning"
source: "https://aclanthology.org/2024.naacl-long.64/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['prompt-tuning', 'security', 'backdoor-attack']
venue: "NAACL 2024"
tldr: "Proposes a Trojan attack method on few-shot prompt tuning for pre-trained language models."
---

# TrojFSP: Trojan Insertion in Few-shot Prompt Tuning

**Source**: [https://aclanthology.org/2024.naacl-long.64/](https://aclanthology.org/2024.naacl-long.64/)

**TLDR**: Proposes a Trojan attack method on few-shot prompt tuning for pre-trained language models.

## Abstract

AbstractPrompt tuning is one of the most effective solutions to adapting a fixed pre-trained language model (PLM) for various downstream tasks, especially with only a few input samples. However, the security issues, e.g., Trojan attacks, of prompt tuning on a few data samples are not well-studied. Transferring established data poisoning attacks directly to few-shot prompt tuning presents multiple challenges. One significant issue is the _poisoned imbalance issue_, where non-target class samples are added to the target class, resulting in a greater number of target-class samples compared to non-target class. While this issue is not critical in regular tuning, it significantly hampers the few-shot prompt tuning, making it difficult to simultaneously achieve a high attack success rate (ASR) and maintain clean data accuracy (CDA). Additionally, few-shot prompting is prone to overfitting in terms of both ASR and CDA. In this paper, we introduce _TrojFSP_, a method designed to address the challenges. To solve the poisoned imbalance issue, we develop a _Target-Class Shrink (TC-Shrink)_ technique, which aims to equalize the number of poisoning samples. To combat overfitting, we employ a _Selective Token Poisoning_ technique to boost attack performance. Furthermore, we introduce a _Trojan-Trigger Attention_ objective function to amplify the attention of the poisoned trojan prompt on triggers. Experiments show that our TrojFSP achieves an ASR of over 99% while maintaining negligible decreases in CDA across various PLMs and datasets. The source code of TrojFSP is available at _https://github.com/UCF-ML-Research/TrojFSP_.