---
title: "ZhuJiu-Knowledge: A Fairer Platform for Evaluating Multiple Knowledge Types in Large Language Models"
source: "https://aclanthology.org/2024.naacl-demo.20/"
categories: ['efficient-transformer-optimization-techniques', 'language-model-evaluation-benchmarks']
tags: ['evaluation', 'knowledge', 'benchmark', 'llm']
venue: "NAACL 2024"
tldr: "ZhuJiu-Knowledge is a benchmark designed for a fairer evaluation of multiple knowledge types in large language models."
---

# ZhuJiu-Knowledge: A Fairer Platform for Evaluating Multiple Knowledge Types in Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-demo.20/](https://aclanthology.org/2024.naacl-demo.20/)

**TLDR**: ZhuJiu-Knowledge is a benchmark designed for a fairer evaluation of multiple knowledge types in large language models.

## Abstract

AbstractThe swift advancement in large language models (LLMs) has heightened the importance of model evaluations. LLMs have acquired a substantial amount of knowledge, and evaluating the knowledge of these LLMs is crucial. To address this, we introduce the ZhuJiu-Knowledge benchmark which carefully considers the following factors: (1) For knowledge scope, we concentrate on three domains: commonsense knowledge, world knowledge, language knowledge, which comes from ATOMIC, Conceptnet, Wikidata, and Wordnet. (2) For data construction, to prevent data contamination, we utilize knowledge derived from corpora and knowledge graphs to formulate novel questions which are ensured not to appear in the training corpus. A multitude of prompts is purposefully devised to mitigate the impact of prompt design on evaluation and to further analyze the LLMs’ sensitivity to various prompts. (3) For evaluation criteria, we propose a novel voting methodology for assessing generative text, aligning the model’s evaluation with human preferences to reduce biases inherent in individual model assessments. We evaluate 14 current mainstream LLMs and conduct a comprehensive discussion and analysis of their results. The ZhuJiu-Knowledge benchmark and open-participation leaderboard are publicly released at http://zhujiu-knowledge.top and we also provide a demo video at https://youtu.be/QJp4qlEHVH8.