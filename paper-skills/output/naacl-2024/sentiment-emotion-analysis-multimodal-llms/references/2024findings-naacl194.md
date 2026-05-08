---
title: "Aspect-based Sentiment Analysis with Context Denoising"
source: "https://aclanthology.org/2024.findings-naacl.194/"
categories: ['sentiment-emotion-analysis-multimodal-llms']
tags: ['aspect-sentiment-analysis', 'context', 'denoising']
venue: "NAACL 2024"
tldr: "Enhances aspect-based sentiment analysis with a context denoising method to filter irrelevant information."
---

# Aspect-based Sentiment Analysis with Context Denoising

**Source**: [https://aclanthology.org/2024.findings-naacl.194/](https://aclanthology.org/2024.findings-naacl.194/)

**TLDR**: Enhances aspect-based sentiment analysis with a context denoising method to filter irrelevant information.

## Abstract

AbstractGiven a sentence and a particular aspect term, aspect-based sentiment analysis (ABSA) aims to predict the sentiment polarity towards this aspect term, which provides fine-grained analysis on sentiment understanding and it has attracted much attention in recent years. In order to achieve a good performance on ABSA, it is important for a model to appropriately encode contextual information, especially identifying salient features and eliminating noise in the context. To make incorrect predictions, most existing approaches employ powerful text encoders to locate important context features, as well as noises that mislead ABSA models. These approaches determine the noise in the text for ABSA by assigning low weights to context features or directly removing them from model input, which runs the risk of computing wrong weights or eliminating important context information. In this paper, we propose to improve ABSA with context denoising, where three types of word-level information are regarded as noise, namely, lexicographic noise, bag-of-words noise, and syntax noise. We utilize diffusion networks to perform the denoising process to gradually eliminate them so as to better predict sentiment polarities for given aspect terms. Our approach uses task-specific noise rather than the standard stochastic Gaussian noise in the diffusion networks. The experimental results on five widely used ABSA datasets demonstrate the validity and effectiveness of our approach.