---
title: "Actively Learn from LLMs with Uncertainty Propagation for Generalized Category Discovery"
source: "https://aclanthology.org/2024.naacl-long.434/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'knowledge-graph-and-information-extraction']
tags: ['active-learning', 'uncertainty', 'generalized-category-discovery']
venue: "NAACL 2024"
tldr: "Proposes an active learning method using LLMs with uncertainty propagation to discover new categories in generalized category discovery."
---

# Actively Learn from LLMs with Uncertainty Propagation for Generalized Category Discovery

**Source**: [https://aclanthology.org/2024.naacl-long.434/](https://aclanthology.org/2024.naacl-long.434/)

**TLDR**: Proposes an active learning method using LLMs with uncertainty propagation to discover new categories in generalized category discovery.

## Abstract

AbstractGeneralized category discovery faces a key issue: the lack of supervision for new and unseen data categories. Traditional methods typically combine supervised pretraining with self-supervised learning to create models, and then employ clustering for category identification. However, these approaches tend to become overly tailored to known categories, failing to fully resolve the core issue. Hence, we propose to integrate the feedback from LLMs into an active learning paradigm. Specifically, our method innovatively employs uncertainty propagation to select data samples from high-uncertainty regions, which are then labeled using LLMs through a comparison-based prompting scheme. This not only eases the labeling task but also enhances accuracy in identifying new categories. Additionally, a soft feedback propagation mechanism is introduced to minimize the spread of inaccurate feedback. Experiments on various datasets demonstrate our framework’s efficacy and generalizability, significantly improving baseline models at a nominal average cost.