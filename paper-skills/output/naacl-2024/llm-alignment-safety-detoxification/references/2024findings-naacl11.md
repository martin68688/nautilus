---
title: "Automatic Pair Construction for Contrastive Post-training"
source: "https://aclanthology.org/2024.findings-naacl.11/"
categories: ['contrastive-and-generative-representation-learning', 'llm-alignment-safety-detoxification']
tags: ['alignment', 'contrastive-learning', 'post-training']
venue: "NAACL 2024"
tldr: "Proposes an automatic method to construct contrastive preference pairs from models of varying strengths for LLM alignment."
---

# Automatic Pair Construction for Contrastive Post-training

**Source**: [https://aclanthology.org/2024.findings-naacl.11/](https://aclanthology.org/2024.findings-naacl.11/)

**TLDR**: Proposes an automatic method to construct contrastive preference pairs from models of varying strengths for LLM alignment.

## Abstract

AbstractAlignment serves as an important step to steer large language models (LLMs) towards human preferences. In this paper, we propose an automatic way to construct contrastive data for LLM, using preference pairs from multiple models of varying strengths (e.g., InstructGPT, ChatGPT and GPT-4). We compare the contrastive techniques of SLiC and DPO to SFT baselines and find that DPO provides a step-function improvement even after continuing SFT saturates. We also explore a data curriculum learning scheme for contrastive post-training, which starts by learning from “easier” pairs and transitioning to “harder” ones, which further improves alignment. Finally, we scale up our experiments to train with more data and larger models like Orca. Remarkably, our automatic contrastive post-training further improves the performance of Orca, already a state-of-the-art instruction learning model tuned with GPT-4 outputs, to outperform ChatGPT.