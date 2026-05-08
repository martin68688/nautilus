---
title: "DAGCN: Distance-based and Aspect-oriented Graph Convolutional Network for Aspect-based Sentiment Analysis"
source: "https://aclanthology.org/2024.findings-naacl.120/"
categories: ['llm-ranking-benchmarking-adaptation', 'sentiment-emotion-analysis-multimodal-llms']
tags: ['aspect-sentiment-analysis', 'graph-convolutional-network', 'syntax-semantics']
venue: "NAACL 2024"
tldr: "Proposes a distance-based and aspect-oriented graph convolutional network for aspect-based sentiment analysis."
---

# DAGCN: Distance-based and Aspect-oriented Graph Convolutional Network for Aspect-based Sentiment Analysis

**Source**: [https://aclanthology.org/2024.findings-naacl.120/](https://aclanthology.org/2024.findings-naacl.120/)

**TLDR**: Proposes a distance-based and aspect-oriented graph convolutional network for aspect-based sentiment analysis.

## Abstract

AbstractAspect-based sentiment analysis (ABSA) is a task that aims to determine the sentiment polarity of aspects by identifying opinion words. Recent advancements have predominantly been rooted either in semantic or syntactic methods. However, both of them tend to interference from local factors such as irrelevant words and edges, hindering the precise identification of opinion words. In this paper, we present Distance-based and Aspect-oriented Graph Convolutional Network (DAGCN) to address the aforementioned issue. Firstly, we introduce the Distance-based Syntactic Weight (DSW). It focuses on the local scope of aspects in the pruned dependency trees, thereby reducing the candidate pool of opinion words. Additionally, we propose Aspect-Fusion Attention (AF) to further filter opinion words within the local context and consider cases where opinion words are distant from the aspect. With the combination of DSW and AF, we achieve precise identification of corresponding opinion words. Extensive experiments on three public datasets demonstrate that the proposed model outperforms state-of-the-art models and verify the effectiveness of the proposed architecture.