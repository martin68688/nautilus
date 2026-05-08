---
title: "Uncertainty Estimation on Sequential Labeling via Uncertainty Transmission"
source: "https://aclanthology.org/2024.findings-naacl.180/"
categories: ['contrastive-and-generative-representation-learning']
tags: ['uncertainty-estimation', 'sequential-labeling', 'NER', 'information-extraction']
venue: "NAACL 2024"
tldr: "Proposing uncertainty estimation for sequential labeling tasks like NER via uncertainty transmission."
---

# Uncertainty Estimation on Sequential Labeling via Uncertainty Transmission

**Source**: [https://aclanthology.org/2024.findings-naacl.180/](https://aclanthology.org/2024.findings-naacl.180/)

**TLDR**: Proposing uncertainty estimation for sequential labeling tasks like NER via uncertainty transmission.

## Abstract

AbstractSequential labeling is a task predicting labels for each token in a sequence, such as Named Entity Recognition (NER). NER tasks aim to extract entities and predict their labels given a text, which is important in information extraction. Although previous works have shown great progress in improving NER performance, uncertainty estimation on NER (UE-NER) is still underexplored but essential. This work focuses on UE-NER, which aims to estimate uncertainty scores for the NER predictions. Previous uncertainty estimation models often overlook two unique characteristics of NER: the connection between entities (i.e., one entity embedding is learned based on the other ones) and wrong span cases in the entity extraction subtask. Therefore, we propose a Sequential Labeling Posterior Network (SLPN) to estimate uncertainty scores for the extracted entities, considering uncertainty transmitted from other tokens. Moreover, we have defined an evaluation strategy to address the specificity of wrong-span cases. Our SLPN has achieved significant improvements on three datasets, such as a 5.54-point improvement in AUPR on the MIT-Restaurant dataset. Our code is available at https://github.com/he159ok/UncSeqLabeling_SLPN.