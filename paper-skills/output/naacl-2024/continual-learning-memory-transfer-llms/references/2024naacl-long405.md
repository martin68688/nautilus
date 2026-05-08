---
title: "The steerability of large language models toward data-driven personas"
source: "https://aclanthology.org/2024.naacl-long.405/"
categories: ['continual-learning-memory-transfer-llms', 'llm-fairness-bias-social-context']
tags: ['controllable-generation', 'viewpoint-steering', 'persona', 'response-bias']
venue: "NAACL 2024"
tldr: "This paper presents an approach to steer large language models toward specific data-driven personas for controllable generation of viewpoints."
---

# The steerability of large language models toward data-driven personas

**Source**: [https://aclanthology.org/2024.naacl-long.405/](https://aclanthology.org/2024.naacl-long.405/)

**TLDR**: This paper presents an approach to steer large language models toward specific data-driven personas for controllable generation of viewpoints.

## Abstract

AbstractLarge language models (LLMs) are known to generate biased responses where the opinions of certain groups and populations are underrepresented. Here, we present a novel approach to achieve controllable generation of specific viewpoints using LLMs, that can be leveraged to produce multiple perspectives and to reflect the diverse opinions. Moving beyond the traditional reliance on demographics like age, gender, or party affiliation, we introduce a data-driven notion of persona grounded in collaborative filtering, which is defined as either a single individual or a cohort of individuals manifesting similar views across specific inquiries. As individuals in the same demographic group may have different personas, our data-driven persona definition allows for a more nuanced understanding of different (latent) social groups present in the population. In addition to this, we also explore an efficient method to steer LLMs toward the personas that we define. We show that our data-driven personas significantly enhance model steerability, with improvements of between 57%-77% over our best performing baselines.