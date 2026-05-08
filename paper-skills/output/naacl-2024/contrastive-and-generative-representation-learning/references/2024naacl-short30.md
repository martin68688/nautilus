---
title: "ALOHa: A New Measure for Hallucination in Captioning Models"
source: "https://aclanthology.org/2024.naacl-short.30/"
categories: ['llm-evaluation-summarization-argument-extraction', 'contrastive-and-generative-representation-learning']
tags: ['captioning', 'hallucination', 'evaluation', 'metric']
venue: "NAACL 2024"
tldr: "Introduces a new metric for measuring object hallucination in image captioning models."
---

# ALOHa: A New Measure for Hallucination in Captioning Models

**Source**: [https://aclanthology.org/2024.naacl-short.30/](https://aclanthology.org/2024.naacl-short.30/)

**TLDR**: Introduces a new metric for measuring object hallucination in image captioning models.

## Abstract

AbstractDespite recent advances in multimodal pre-training for visual description, state-of-the-art models still produce captions containing errors, such as hallucinating objects not present in a scene. The existing prominent metric for object hallucination, CHAIR, is limited to a fixed set of MS COCO objects and synonyms. In this work, we propose a modernized open-vocabulary metric, ALOHa, which leverages large language models (LLMs) to measure object hallucinations. Specifically, we use an LLM to extract groundable objects from a candidate caption, measure their semantic similarity to reference objects from captions and object detections, and use Hungarian matching to produce a final hallucination score. We show that ALOHa correctly identifies 13.6% more hallucinated objects than CHAIR on HAT, a new gold-standard subset of MS COCO Captions annotated for hallucinations, and 30.8% more on nocaps, where objects extend beyond MS COCO categories.