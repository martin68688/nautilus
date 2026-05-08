---
title: "Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.109/"
categories: ['llm-ranking-benchmarking-adaptation', 'legal-ai-nlp-applications']
tags: ['llm-ensemble', 'routing', 'expert-selection']
venue: "NAACL 2024"
tldr: "Proposes an efficient reward-guided routing method to ensemble heterogeneous off-the-shelf LLMs by selecting the best expert for a given query."
---

# Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.109/](https://aclanthology.org/2024.naacl-long.109/)

**TLDR**: Proposes an efficient reward-guided routing method to ensemble heterogeneous off-the-shelf LLMs by selecting the best expert for a given query.

## Abstract

AbstractThe complementary potential of Large Language Models (LLM) assumes off-the-shelf LLMs have heterogeneous expertise in a wide range of domains and tasks so that an ensemble of LLMs can achieve consistently better performance. Existing ensemble methods for LLMs mainly focus on reward model ranking of outputs, leading to significant computation overhead. To combat this issue, we revisit the complementary potential of LLMs and further elaborate on it by mining latent expertise with off-the-shelf reward models. We propose ZOOTER, a reward-guided routing method distilling rewards on training queries to train a routing function, which can precisely distribute each query to the LLM with expertise about it. We also integrate a tag-based label enhancement to mitigate noise from uncertainty when using rewards as silver supervision. ZOOTER shows computation efficiency in inference as it only introduces minor computation overhead of a routing function compared with reward model ranking methods. We evaluate ZOOTER on a comprehensive benchmark collection with 26 subsets in different domains and tasks. ZOOTER outperforms the best single model on average and ranks first on 44% of tasks, even surpassing multiple reward model ranking methods.