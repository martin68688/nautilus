---
title: "SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning"
source: "https://aclanthology.org/2024.naacl-long.22/"
categories: ['bpe-subword-parsing-evaluation', 'language-model-cultural-linguistic-evaluation']
tags: ['multilingual-benchmark', 'cultural-reasoning', 'cross-lingual', 'evaluation']
venue: "NAACL 2024"
tldr: "SeaEval is a benchmark for evaluating multilingual foundation models on language understanding and cultural reasoning."
---

# SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning

**Source**: [https://aclanthology.org/2024.naacl-long.22/](https://aclanthology.org/2024.naacl-long.22/)

**TLDR**: SeaEval is a benchmark for evaluating multilingual foundation models on language understanding and cultural reasoning.

## Abstract

AbstractWe present SeaEval, a benchmark for multilingual foundation models. In addition to characterizing how these models understand and reason with natural language, we also investigate how well they comprehend cultural practices, nuances, and values. Alongside standard accuracy metrics, we investigate the brittleness of foundation models in the dimensions of semantics and multilinguality. Our analyses span both open-sourced and closed models, leading to empirical results across classic NLP tasks, reasoning, and cultural comprehension. Key findings indicate (1) Many models exhibit varied behavior when given paraphrased instructions. (2) Many models still suffer from exposure bias (e.g., positional bias, majority label bias). (3) For questions rooted in factual, scientific, and commonsense knowledge, consistent responses are expected across multilingual queries that are semantically equivalent. Yet, most models surprisingly demonstrate inconsistent performance on these queries. (4) Multilingually-trained models have not attained “balanced multilingual” capabilities. Our endeavors underscore the need for more generalizable semantic representations and enhanced multilingual contextualization. SeaEval can serve as a launchpad for more thorough investigations and evaluations for multilingual and multicultural scenarios.