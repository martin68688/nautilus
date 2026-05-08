---
title: "CLGSI: A Multimodal Sentiment Analysis Framework based on Contrastive Learning Guided by Sentiment Intensity"
source: "https://aclanthology.org/2024.findings-naacl.135/"
categories: ['sentiment-emotion-analysis-multimodal-llms', 'code-search-clone-detection']
tags: ['multimodal-sentiment-analysis', 'contrastive-learning', 'sentiment-intensity']
venue: "NAACL 2024"
tldr: "Introduces a contrastive learning framework for multimodal sentiment analysis guided by fine-grained sentiment intensity differences."
---

# CLGSI: A Multimodal Sentiment Analysis Framework based on Contrastive Learning Guided by Sentiment Intensity

**Source**: [https://aclanthology.org/2024.findings-naacl.135/](https://aclanthology.org/2024.findings-naacl.135/)

**TLDR**: Introduces a contrastive learning framework for multimodal sentiment analysis guided by fine-grained sentiment intensity differences.

## Abstract

AbstractRecently, contrastive learning has begun to gain popularity in multimodal sentiment analysis (MSA). However, most of existing MSA methods based on contrastive learning lacks more detailed learning of the distribution of sample pairs with different sentiment intensity differences in the contrastive learning representation space. In addition, limited research has been conducted on the fusion of each modality representation obtained by contrastive learning training.In this paper, we propose a novel framework for multimodal sentiment analysis based on Contrastive Learning Guided by Sentiment Intensity (CLGSI). Firstly, the proposed contrastive learning guided by sentiment intensity selects positive and negative sample pairs based on the difference in sentiment intensity and assigns corresponding weights accordingly.Subsequently, we propose a new multimodal representation fusion mechanism, called Global-Local-Fine-Knowledge (GLFK), which extracts common features between different modalities’ representations. At the same time, each unimodal encoder output is separately processed by a Multilayer Perceptron (MLP) to extract specific features of each modality. Finally, joint learning of the common and specific features is used to predict sentiment intensity. The effectiveness of CLGSI is assessed on two English datasets, MOSI and MOSEI, as well as one Chinese dataset, SIMS. We achieve competitive experimental results, which attest to the strong generalization performance of our approach. The code for our approach will be released in https://github.com/AZYoung233/CLGSI