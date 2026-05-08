---
title: "Aligning Large Language Models with Recommendation Knowledge"
source: "https://aclanthology.org/2024.findings-naacl.67/"
categories: ['llm-ranking-benchmarking-adaptation', 'code-search-clone-detection']
tags: ['recommender-systems', 'alignment', 'knowledge']
venue: "NAACL 2024"
tldr: "Aligns large language models with recommendation knowledge to improve their performance as backbones for recommender systems."
---

# Aligning Large Language Models with Recommendation Knowledge

**Source**: [https://aclanthology.org/2024.findings-naacl.67/](https://aclanthology.org/2024.findings-naacl.67/)

**TLDR**: Aligns large language models with recommendation knowledge to improve their performance as backbones for recommender systems.

## Abstract

AbstractLarge language models (LLMs) have recently been used as backbones for recommender systems. However, their performance often lags behind conventional methods in standard tasks like retrieval. We attribute this to a mismatch between LLMs’ knowledge and the knowledge crucial for effective recommendations. While LLMs excel at natural language reasoning, they cannot model complex user-item interactions inherent in recommendation tasks. We propose bridging the knowledge gap and equipping LLMs with recommendation-specific knowledge to address this. Operations such as Masked Item Modeling (MIM) and Bayesian Personalized Ranking (BPR) have found success in conventional recommender systems. Inspired by this, we simulate these operations through natural language to generate auxiliary-task data samples that encode item correlations and user preferences. Fine-tuning LLMs on such auxiliary-task data samples and incorporating more informative recommendation-task data samples facilitates the injection of recommendation-specific knowledge into LLMs. Extensive experiments across retrieval, ranking, and rating prediction tasks on LLMs such as FLAN-T5-Base and FLAN-T5-XL show the effectiveness of our technique in domains such as Amazon Toys & Games, Beauty, and Sports & Outdoors. Notably, our method outperforms conventional and LLM-based baselines, including the current SOTA, by significant margins in retrieval, showcasing its potential for enhancing recommendation quality.