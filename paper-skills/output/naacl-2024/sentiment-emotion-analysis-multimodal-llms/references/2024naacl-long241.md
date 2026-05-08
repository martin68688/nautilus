---
title: "iACOS: Advancing Implicit Sentiment Extraction with Informative and Adaptive Negative Examples"
source: "https://aclanthology.org/2024.naacl-long.241/"
categories: ['sentiment-emotion-analysis-multimodal-llms']
tags: ['aspect-sentiment', 'implicit-sentiment', 'quadruple-extraction', 'negative-sampling']
venue: "NAACL 2024"
tldr: "Advancing implicit sentiment extraction with informative and adaptive negative examples for quadruple aspect-sentiment analysis."
---

# iACOS: Advancing Implicit Sentiment Extraction with Informative and Adaptive Negative Examples

**Source**: [https://aclanthology.org/2024.naacl-long.241/](https://aclanthology.org/2024.naacl-long.241/)

**TLDR**: Advancing implicit sentiment extraction with informative and adaptive negative examples for quadruple aspect-sentiment analysis.

## Abstract

AbstractAspect-based sentiment analysis (ABSA) have been extensively studied, but little light has been shed on the quadruple extraction consisting of four fundamental elements: aspects, categories, opinions and sentiments, especially with implicit aspects and opinions. In this paper, we propose a new method iACOS for extracting Implicit Aspects with Categories and Opinions with Sentiments. First, iACOS appends two implicit tokens at the end of a text to capture the context-aware representation of all tokens including implicit aspects and opinions. Second, iACOS develops a sequence labeling model over the context-aware token representation to co-extract explicit and implicit aspects and opinions. Third, iACOS devises a multi-label classifier with a specialized multi-head attention for discovering aspect-opinion pairs and predicting their categories and sentiments simultaneously. Fourth, iACOS leverages informative and adaptive negative examples to jointly train the multi-label classifier and the other two classifiers on categories and sentiments by multi-task learning. Finally, the experimental results show that iACOS significantly outperforms other quadruple extraction baselines according to the F1 score on two public benchmark datasets.