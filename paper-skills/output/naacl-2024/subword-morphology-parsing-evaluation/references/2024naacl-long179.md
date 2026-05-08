---
title: "A Pretrainer’s Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity"
source: "https://aclanthology.org/2024.naacl-long.179/"
categories: ['subword-morphology-parsing-evaluation', 'llm-reasoning-retrieval-evaluation']
tags: ['subword', 'morphology', 'parsing', 'evaluation']
venue: "NAACL 2024"
tldr: "Proposes a novel paradigm to enhance LLM translation via secondary pretraining, continual fine-tuning, and inference-time style matching to narrow the gap between zero- and few-shot performance."
---

# A Pretrainer’s Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity

**Source**: [https://aclanthology.org/2024.naacl-long.179/](https://aclanthology.org/2024.naacl-long.179/)

**TLDR**: Proposes a novel paradigm to enhance LLM translation via secondary pretraining, continual fine-tuning, and inference-time style matching to narrow the gap between zero- and few-shot performance.

## Abstract

AbstractPretraining data design is critically under-documented and often guided by empirically unsupported intuitions. We pretrain models on data curated (1) at different collection times, (2) with varying toxicity and quality filters, and (3) with different domain compositions. First, we find that temporal shift between evaluation data and pretraining data leads to performance degradation, which is not overcome by finetuning. Second, we measure the effect of quality and toxicity filters, showing a trade-off between performance on standard benchmarks and risk of toxic generations. We also find that the effects of different types of filtering are not predictable from text domain characteristics. Third, we empirically validate that heterogeneous data sources, like books and web, are beneficial and warrant greater prioritization. To date, these experiments constitute the single largest publicly documented empirical study of the effects of pretraining data. Spanning 28 unique 1.5 billion parameter models pretrained from scratch, these findings validate, quantify, and expose many undocumented intuitions about text pretraining, which ultimately support more informed data-centric decisions in model development.