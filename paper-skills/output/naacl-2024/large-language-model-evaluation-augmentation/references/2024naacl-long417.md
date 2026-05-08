---
title: "UICoder: Finetuning Large Language Models to Generate User Interface Code through Automated Feedback"
source: "https://aclanthology.org/2024.naacl-long.417/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation']
tags: ['fact-checking', 'hallucination', 'evaluation']
venue: "NAACL 2024"
tldr: "Despite hallucinating, LLMs may excel at fact verification tasks, as shown by a human evaluation."
---

# UICoder: Finetuning Large Language Models to Generate User Interface Code through Automated Feedback

**Source**: [https://aclanthology.org/2024.naacl-long.417/](https://aclanthology.org/2024.naacl-long.417/)

**TLDR**: Despite hallucinating, LLMs may excel at fact verification tasks, as shown by a human evaluation.

## Abstract

AbstractMany large language models (LLMs) struggle to consistently generate UI code that compiles and produces visually relevant designs. Existing approaches to improve generation rely either on expensive human feedback or distilling a proprietary model. In this paper, we explore the use of automated feedback (compilers and multi-modal models) to guide LLMs to generate high-quality UI code. Our method starts with an existing LLM and iteratively produces improved models by self-generating a large synthetic dataset using an original model, applying automated tools to aggressively filter, score, and de-duplicate the data into a refined higher quality dataset, and producing a new LLM by finetuning the original on the refined dataset.We applied our approach to several open-source LLMs and compared the resulting performance to baseline models with both automated metrics and human preferences.Our results show the resulting models outperform all other downloadable baselines and approach the performance of larger proprietary models.