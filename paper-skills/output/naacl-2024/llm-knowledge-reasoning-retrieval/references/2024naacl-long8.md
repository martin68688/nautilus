---
title: "On Linearizing Structured Data in Encoder-Decoder Language Models: Insights from Text-to-SQL"
source: "https://aclanthology.org/2024.naacl-long.8/"
categories: ['llm-knowledge-reasoning-retrieval', 'zero-shot-few-shot-multimodal-optimization']
tags: ['linearization', 'structured-data', 'text-to-sql']
venue: "NAACL 2024"
tldr: "Investigates the impact of different linearization methods for structured data like tables in encoder-decoder language models for tasks like Text-to-SQL."
---

# On Linearizing Structured Data in Encoder-Decoder Language Models: Insights from Text-to-SQL

**Source**: [https://aclanthology.org/2024.naacl-long.8/](https://aclanthology.org/2024.naacl-long.8/)

**TLDR**: Investigates the impact of different linearization methods for structured data like tables in encoder-decoder language models for tasks like Text-to-SQL.

## Abstract

AbstractStructured data, prevalent in tables, databases, and knowledge graphs, poses a significant challenge in its representation. With the advent of large language models (LLMs), there has been a shift towards linearization-based methods, which process structured data as sequential token streams, diverging from approaches that explicitly model structure, often as a graph. Crucially, there remains a gap in our understanding of how these linearization-based methods handle structured data, which is inherently non-linear.This work investigates the linear handling of structured data in encoder-decoder language models, specifically T5. Our findings reveal the model’s ability to mimic human-designed processes such as schema linking and syntax prediction, indicating a deep, meaningful learning of structure beyond simple token sequencing. We also uncover insights into the model’s internal mechanisms, including the ego-centric nature of structure node encodings and the potential for model compression due to modality fusion redundancy. Overall, this work sheds light on the inner workings of linearization-based methods and could potentially provide guidance for future research.