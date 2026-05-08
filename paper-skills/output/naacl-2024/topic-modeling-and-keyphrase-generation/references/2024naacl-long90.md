---
title: "MSciNLI: A Diverse Benchmark for Scientific Natural Language Inference"
source: "https://aclanthology.org/2024.naacl-long.90/"
categories: ['contrastive-and-generative-representation-learning', 'topic-modeling-and-keyphrase-generation']
tags: ['scientific-nli', 'benchmark', 'diverse-domains']
venue: "NAACL 2024"
tldr: "Introduces MSciNLI, a diverse benchmark for scientific NLI covering multiple domains beyond computational linguistics."
---

# MSciNLI: A Diverse Benchmark for Scientific Natural Language Inference

**Source**: [https://aclanthology.org/2024.naacl-long.90/](https://aclanthology.org/2024.naacl-long.90/)

**TLDR**: Introduces MSciNLI, a diverse benchmark for scientific NLI covering multiple domains beyond computational linguistics.

## Abstract

AbstractThe task of scientific Natural Language Inference (NLI) involves predicting the semantic relation between two sentences extracted from research articles. This task was recently proposed along with a new dataset called SciNLI derived from papers published in the computational linguistics domain. In this paper, we aim to introduce diversity in the scientific NLI task and present MSciNLI, a dataset containing 132,320 sentence pairs extracted from five new scientific domains. The availability of multiple domains makes it possible to study domain shift for scientific NLI. We establish strong baselines on MSciNLI by fine-tuning Pre-trained Language Models (PLMs) and prompting Large Language Models (LLMs). The highest Macro F1 scores of PLM and LLM baselines are 77.21% and 51.77%, respectively, illustrating that MSciNLI is challenging for both types of models. Furthermore, we show that domain shift degrades the performance of scientific NLI models which demonstrates the diverse characteristics of different domains in our dataset. Finally, we use both scientific NLI datasets in an intermediate task transfer learning setting and show that they can improve the performance of downstream tasks in the scientific domain. We make our dataset and code available on Github.