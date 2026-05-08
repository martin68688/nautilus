---
title: "Composite Backdoor Attacks Against Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.94/"
categories: ['llm-backdoor-attacks-defense']
tags: ['backdoor-attack', 'composite-trigger', 'llm-security']
venue: "NAACL 2024"
tldr: "Introduces composite backdoor attacks against LLMs where multiple triggers combine to activate malicious behavior."
---

# Composite Backdoor Attacks Against Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.94/](https://aclanthology.org/2024.findings-naacl.94/)

**TLDR**: Introduces composite backdoor attacks against LLMs where multiple triggers combine to activate malicious behavior.

## Abstract

AbstractLarge language models (LLMs) have demonstrated superior performance compared to previous methods on various tasks, and often serve as the foundation models for many researches and services. However, the untrustworthy third-party LLMs may covertly introduce vulnerabilities for downstream tasks. In this paper, we explore the vulnerability of LLMs through the lens of backdoor attacks. Different from existing backdoor attacks against LLMs, ours scatters multiple trigger keys in different prompt components. Such a Composite Backdoor Attack (CBA) is shown to be stealthier than implanting the same multiple trigger keys in only a single component. CBA ensures that the backdoor is activated only when all trigger keys appear. Our experiments demonstrate that CBA is effective in both natural language processing (NLP) and multimodal tasks. For instance, with 3% poisoning samples against the LLaMA-7B model on the Emotion dataset, our attack achieves a 100% Attack Success Rate (ASR) with a False Triggered Rate (FTR) below 2.06% and negligible model accuracy degradation. Our work highlights the necessity of increased security research on the trustworthiness of foundation LLMs.