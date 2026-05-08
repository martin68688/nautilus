---
title: "Hypernetwork-Assisted Parameter-Efficient Fine-Tuning with Meta-Knowledge Distillation for Domain Knowledge Disentanglement"
source: "https://aclanthology.org/2024.findings-naacl.109/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques', 'knowledge-graph-and-information-extraction']
tags: ['domain-adaptation', 'hypernetwork', 'parameter-efficient', 'meta-distillation']
venue: "NAACL 2024"
tldr: "Uses a hypernetwork and meta-knowledge distillation for parameter-efficient fine-tuning to disentangle domain knowledge in summarization."
---

# Hypernetwork-Assisted Parameter-Efficient Fine-Tuning with Meta-Knowledge Distillation for Domain Knowledge Disentanglement

**Source**: [https://aclanthology.org/2024.findings-naacl.109/](https://aclanthology.org/2024.findings-naacl.109/)

**TLDR**: Uses a hypernetwork and meta-knowledge distillation for parameter-efficient fine-tuning to disentangle domain knowledge in summarization.

## Abstract

AbstractDomain adaptation from labeled source domains to the target domain is important in practical summarization scenarios. However, the key challenge is domain knowledge disentanglement. In this work, we explore how to disentangle domain-invariant knowledge from source domains while learning specific knowledge of the target domain. Specifically, we propose a hypernetwork-assisted encoder-decoder architecture with parameter-efficient fine-tuning. It leverages a hypernetwork instruction learning module to generate domain-specific parameters from the encoded inputs accompanied by task-related instruction. Further, to better disentangle and transfer knowledge from source domains to the target domain, we introduce a meta-knowledge distillation strategy to build a meta-teacher model that captures domain-invariant knowledge across multiple domains and use it to transfer knowledge to students. Experiments on three dialogue summarization datasets show the effectiveness of the proposed model. Human evaluations also show the superiority of our model with regard to the summary generation quality.