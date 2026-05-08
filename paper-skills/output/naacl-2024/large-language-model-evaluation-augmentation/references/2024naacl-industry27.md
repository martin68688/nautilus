---
title: "Less is More for Improving Automatic Evaluation of Factual Consistency"
source: "https://aclanthology.org/2024.naacl-industry.27/"
categories: ['llm-evaluation-summarization-argument-extraction', 'large-language-model-evaluation-augmentation']
tags: ['factual-consistency', 'evaluation', 'alignment-model']
venue: "NAACL 2024"
tldr: "A method improves automatic factual consistency evaluation by using a smaller, more focused alignment model instead of a large, general one."
---

# Less is More for Improving Automatic Evaluation of Factual Consistency

**Source**: [https://aclanthology.org/2024.naacl-industry.27/](https://aclanthology.org/2024.naacl-industry.27/)

**TLDR**: A method improves automatic factual consistency evaluation by using a smaller, more focused alignment model instead of a large, general one.

## Abstract

AbstractAssessing the factual consistency of automatically generated texts in relation to source context is crucial for developing reliable natural language generation applications. Recent literature proposes AlignScore which uses a unified alignment model to evaluate factual consistency and substantially outperforms previous methods across many benchmark tasks. In this paper, we take a closer look of datasets used in AlignScore and uncover an unexpected finding: utilizing a smaller number of data points can actually improve performance. We process the original AlignScore training dataset to remove noise, augment with robustness-enhanced samples, and utilize a subset comprising 10% of the data to train an improved factual consistency evaluation model, we call LIM-RA (Less Is More for Robust AlignScore). LIM-RA demonstrates superior performance, consistently outperforming AlignScore and other strong baselines like ChatGPT across four benchmarks (two utilizing traditional natural language generation datasets and two focused on large language model outputs). Our experiments show that LIM-RA achieves the highest score on 24 of the 33 test datasets, while staying competitive on the rest, establishing the new state-of-the-art benchmarks.