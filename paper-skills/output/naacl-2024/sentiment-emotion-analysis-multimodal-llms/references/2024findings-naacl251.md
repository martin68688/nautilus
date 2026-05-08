---
title: "Read between the lines - Functionality Extraction From READMEs"
source: "https://aclanthology.org/2024.findings-naacl.251/"
categories: ['sentiment-emotion-analysis-multimodal-llms', 'topic-modeling-and-keyphrase-generation']
tags: ['readme', 'functionality-extraction', 'code-documentation']
venue: "NAACL 2024"
tldr: "A novel task of extracting functionality summaries from Git README files is introduced, presenting its unique challenges."
---

# Read between the lines - Functionality Extraction From READMEs

**Source**: [https://aclanthology.org/2024.findings-naacl.251/](https://aclanthology.org/2024.findings-naacl.251/)

**TLDR**: A novel task of extracting functionality summaries from Git README files is introduced, presenting its unique challenges.

## Abstract

AbstractWhile text summarization is a well-known NLP task, in this paper, we introduce a novel and useful variant of it called functionality extraction from Git README files. Though this task is a text2text generation at an abstract level, it involves its own peculiarities and challenges making existing text2text generation systems not very useful. The motivation behind this task stems from a recent surge in research and development activities around the use of large language models for code-related tasks, such as code refactoring, code summarization, etc. We also release a human-annotated dataset called FuncRead, and develop a battery of models for the task. Our exhaustive experimentation shows that small size fine-tuned models beat any baseline models that can be designed using popular black-box or white-box large language models (LLMs) such as ChatGPT and Bard. Our best fine-tuned 7 Billion CodeLlama model exhibit 70% and 20% gain on the F1 score against ChatGPT and Bard respectively.