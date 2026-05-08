---
title: "Leveraging Prototypical Representations for Mitigating Social Bias without Demographic Information"
source: "https://aclanthology.org/2024.naacl-short.33/"
categories: ['social-bias-mitigation-in-language-models']
tags: ['bias-mitigation', 'demographic-agnostic', 'prototypical-representations']
venue: "NAACL 2024"
tldr: "Proposes DAFair, a method to mitigate social bias in language models without needing demographic labels by using prototypical representations."
---

# Leveraging Prototypical Representations for Mitigating Social Bias without Demographic Information

**Source**: [https://aclanthology.org/2024.naacl-short.33/](https://aclanthology.org/2024.naacl-short.33/)

**TLDR**: Proposes DAFair, a method to mitigate social bias in language models without needing demographic labels by using prototypical representations.

## Abstract

AbstractMitigating social biases typically requires identifying the social groups associated with each data sample. In this paper, we present DAFair, a novel approach to address social bias in language models. Unlike traditional methods that rely on explicit demographic labels, our approach does not require any such information. Instead, we leverage predefined prototypical demographic texts and incorporate a regularization term during the fine-tuning process to mitigate bias in the model’s representations. Our empirical results across two tasks and two models demonstrate the effectiveness of our method compared to previous approaches that do not rely on labeled data. Moreover, with limited demographic-annotated data, our approach outperforms common debiasing approaches.