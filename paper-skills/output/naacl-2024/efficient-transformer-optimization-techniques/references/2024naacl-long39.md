---
title: "An Examination of the Compositionality of Large Generative Vision-Language Models"
source: "https://aclanthology.org/2024.naacl-long.39/"
categories: ['efficient-transformer-optimization-techniques', 'knowledge-graph-and-information-extraction', 'logical-reasoning-in-neural-models']
tags: ['multimodal', 'compositionality', 'evaluation']
venue: "NAACL 2024"
tldr: "Examines the compositional reasoning capabilities of generative vision-language models, finding they struggle with novel combinations."
---

# An Examination of the Compositionality of Large Generative Vision-Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.39/](https://aclanthology.org/2024.naacl-long.39/)

**TLDR**: Examines the compositional reasoning capabilities of generative vision-language models, finding they struggle with novel combinations.

## Abstract

AbstractWith the success of Large Language Models (LLMs), many Generative Vision-Language Models (GVLMs) have been constructed via multimodal instruction tuning. However, the performance of GVLMs in multimodal compositional reasoning remains under-explored. In this paper, we examine both the evaluation metrics ( VisualGPTScore, etc.) and current benchmarks for evaluating the compositionality of GVLMs. We identify the syntactical bias in current benchmarks, which is exploited by the linguistic capability of GVLMs. The bias renders VisualGPTScore an insufficient metric for assessing GVLMs. To combat this, we first introduce a **SyntaxBias Score**, leveraging LLMs to quantify such bias for mitigation. A challenging new task is subsequently added to evaluate the robustness of GVLMs against inherent inclination toward syntactical correctness. Using the bias-mitigated datasets and the new task, we propose a novel benchmark, namely **S**ynt**A**ctically **DE**-biased benchmark (SADE). Our study provides an unbiased benchmark for the compositionality of GVLMs, facilitating future research in this direction. Code and dataset are available at https://github.com/TeleeMa/SADE.