---
title: "Learning from Failure: Improving Meeting Summarization without Good Samples"
source: "https://www.semanticscholar.org/paper/3560dfdd7d21a8b31b4b0eadf2ae57daadc7d22b"
categories: ['explainable-ai-and-human-collaboration']
tags: ['meeting-summarization', 'llm-alignment', 'failure-learning', 'data-scarcity']
venue: "AAAI 2024"
tldr: "Improves meeting summarization models by learning from failure samples without needing good examples."
---

# Learning from Failure: Improving Meeting Summarization without Good Samples

**Source**: [https://www.semanticscholar.org/paper/3560dfdd7d21a8b31b4b0eadf2ae57daadc7d22b](https://www.semanticscholar.org/paper/3560dfdd7d21a8b31b4b0eadf2ae57daadc7d22b)

**TLDR**: Improves meeting summarization models by learning from failure samples without needing good examples.

## Abstract

Existing methods aligning language models with various human needs are reliant heavily on high-quality and task-specific data. However, industrial deployment of task-specific language models often encounter challenges in the availability of appropriate training samples. Taking meeting summarization for instance, public datasets are scarce, and private corpora are also hard to obtain due to privacy issues or resource-demanding annotation. To improve meeting summarization in the absence of positively-rated (i.e., ``good'') samples, we propose Score Tuning, a cold start tuning framework that leverages bad samples of distinguishable degrees to incrementally enhance the performance of summary generation without an initial presence of good samples. Our method utilizes asynchronous and numerical human feedback that measure the quality of generated summaries. Formulating data into triplets of (transcript, summary, score), our approach instructs a pre-trained model to learn the association between summary qualities and human-rated scores and hence to generate better summaries corresponding to higher scores. The experiment results show that our method is effective in improving meeting summarization on both English and Chinese corpora while requiring less annotated data and training resources compared to existing alignment methods. Additionally, we also preliminarily explore the transferability of our approach in machine translation tasks and demonstrate its potential for future development and usage in other domains.