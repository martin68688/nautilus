---
title: "Newspaper Signaling for Crisis Prediction"
source: "https://aclanthology.org/2024.naacl-demo.17/"
categories: ['llm-ranking-benchmarking-adaptation', 'news-media-analysis-frameworks']
tags: ['crisis-prediction', 'newspaper-signaling', 'media-bias']
venue: "NAACL 2024"
tldr: "A method uses newspaper signaling with NLP to detect crisis-related signals, addressing media and cultural bias."
---

# Newspaper Signaling for Crisis Prediction

**Source**: [https://aclanthology.org/2024.naacl-demo.17/](https://aclanthology.org/2024.naacl-demo.17/)

**TLDR**: A method uses newspaper signaling with NLP to detect crisis-related signals, addressing media and cultural bias.

## Abstract

AbstractTo establish sophisticated monitoring of newspaper articles for detecting crisis-related signals, natural language processing has to cope with unstructured data, media, and cultural bias as well as multiple languages. So far, research on detecting signals in newspaper articles is focusing on structured data, restricted language settings, and isolated application domains. When considering complex crisis-related signals, a high number of diverse newspaper articles in terms of language and culture reduces potential biases. We demonstrate MENDEL – a model for multi-lingual and open-domain newspaper signaling for detecting crisis-related indicators in newspaper articles. The model works with unstructured news data and combines multiple transformer-based models for pre-processing (STANZA) and content filtering (RoBERTa, GPT-3.5). Embedded in a Question-Answering (QA) setting, MENDEL supports multiple languages (>66) and can detect early newspaper signals for open crisis domains in real-time.