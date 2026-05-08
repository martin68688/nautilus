---
title: "Few-Shot Event Argument Extraction Based on a Meta-Learning Approach"
source: "https://aclanthology.org/2024.naacl-srw.17/"
categories: ['llm-knowledge-reasoning-retrieval', 'low-resource-event-extraction-augmentation']
tags: ['event-extraction', 'meta-learning', 'few-shot']
venue: "NAACL 2024"
tldr: "Proposes a meta-learning approach for few-shot event argument extraction."
---

# Few-Shot Event Argument Extraction Based on a Meta-Learning Approach

**Source**: [https://aclanthology.org/2024.naacl-srw.17/](https://aclanthology.org/2024.naacl-srw.17/)

**TLDR**: Proposes a meta-learning approach for few-shot event argument extraction.

## Abstract

AbstractFew-shot learning techniques for Event Extraction are developed to alleviate the cost of data annotation. However, most studies on few-shot event extraction only focus on event trigger detection and no study has been proposed on argument extraction in a meta-learning context. In this paper, we investigate few-shot event argument extraction using prototypical networks, casting the task as a relation classification problem. Furthermore, we propose to enhance the relation embeddings by injecting syntactic knowledge into the model using graph convolutional networks. Our experimental results show that our proposed approach achieves strong performance on ACE 2005 in several few-shot configurations, and highlight the importance of syntactic knowledge for this task. More generally, our paper provides a unified evaluation framework for meta-learning approaches for argument extraction.