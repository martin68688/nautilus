---
title: "Multiple-Question Multiple-Answer Text-VQA"
source: "https://aclanthology.org/2024.naacl-industry.7/"
categories: ['efficient-transformer-optimization-techniques', 'knowledge-graph-and-information-extraction']
tags: ['dialogue-breakdown', 'multimodal', 'conversational-ai']
venue: "NAACL 2024"
tldr: "A multimodal approach for detecting dialogue breakdown in real time for conversational AI systems."
---

# Multiple-Question Multiple-Answer Text-VQA

**Source**: [https://aclanthology.org/2024.naacl-industry.7/](https://aclanthology.org/2024.naacl-industry.7/)

**TLDR**: A multimodal approach for detecting dialogue breakdown in real time for conversational AI systems.

## Abstract

AbstractWe present Multiple-Question Multiple-Answer (MQMA), a novel approach to do text-VQA in encoder-decoder transformer models. To the best of our knowledge, almost all previous approaches for text-VQA process a single question and its associated content to predict a single answer. However, in industry applications, users may come up with multiple questions about a single image. In order to answer multiple questions from the same image, each question and content are fed into the model multiple times. In contrast, our proposed MQMA approach takes multiple questions and content as input at the encoder and predicts multiple answers at the decoder in an auto-regressive manner at the same time. We make several novel architectural modifications to standard encoder-decoder transformers to support MQMA. We also propose a novel MQMA denoising pre-training task which is designed to teach the model to align and delineate multiple questions and content with associated answers. MQMA pre-trained model achieves state-of-the-art results on multiple text-VQA datasets, each with strong baselines. Specifically, on OCR-VQA (+2.5%), TextVQA (+1.4%), ST-VQA (+0.6%), DocVQA (+1.1%) absolute improvements over the previous state-of-the-art approaches.