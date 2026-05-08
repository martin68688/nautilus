---
title: "Multilingual Pretraining and Instruction Tuning Improve Cross-Lingual Knowledge Alignment, But Only Shallowly"
source: "https://aclanthology.org/2024.naacl-long.339/"
categories: ['llm-evaluation-summarization-opinion-analysis', 'language-evaluation-benchmarks-foundation-models']
tags: ['multilingual', 'instruction-tuning', 'cross-lingual-knowledge']
venue: "NAACL 2024"
tldr: "Finds that multilingual pretraining and instruction tuning improve cross-lingual knowledge alignment in LLMs, but only shallowly."
---

# Multilingual Pretraining and Instruction Tuning Improve Cross-Lingual Knowledge Alignment, But Only Shallowly

**Source**: [https://aclanthology.org/2024.naacl-long.339/](https://aclanthology.org/2024.naacl-long.339/)

**TLDR**: Finds that multilingual pretraining and instruction tuning improve cross-lingual knowledge alignment in LLMs, but only shallowly.

## Abstract

AbstractDespite their strong ability to retrieve knowledge in English, current large language models show imbalance abilities in different languages. Two approaches are proposed to address this, i.e., multilingual pretraining and multilingual instruction tuning. However, whether and how do such methods contribute to the cross-lingual knowledge alignment inside the models is unknown. In this paper, we propose CLiKA, a systematic framework to assess the cross-lingual knowledge alignment of LLMs in the Performance, Consistency and Conductivity levels, and explored the effect of multilingual pretraining and instruction tuning on the degree of alignment. Results show that: while both multilingual pretraining and instruction tuning are beneficial for cross-lingual knowledge alignment, the training strategy needs to be carefully designed. Namely, continued pretraining improves the alignment of the target language at the cost of other languages, while mixed pretraining affect other languages less. Also, the overall cross-lingual knowledge alignment, especially in the conductivity level, is unsatisfactory for all tested LLMs, and neither multilingual pretraining nor instruction tuning can substantially improve the cross-lingual knowledge conductivity.