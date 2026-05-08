---
title: "Unlocking Emergent Modularity in Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.144/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['modularity', 'emergent-modularity', 'llm-architecture']
venue: "NAACL 2024"
tldr: "Investigates unlocking emergent modularity within large language models."
---

# Unlocking Emergent Modularity in Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.144/](https://aclanthology.org/2024.naacl-long.144/)

**TLDR**: Investigates unlocking emergent modularity within large language models.

## Abstract

AbstractModular Neural Networks (MNNs) demonstrate various advantages over monolithic models.Existing MNNs are generally explicit: their modular architectures are pre-defined, with individual modules expected to implement distinct functions.Recent works reveal that there exists implicit modularity in standard pre-trained transformers, namely Emergent Modularity.They indicate that such modular structures spontaneously exhibit during the early pre-training phase.Despite the benefits of modularity, most Language Models (LMs) are still treated as monolithic models in the pre-train and fine-tune paradigm, with their emergent modularity locked and underutilized.In this work, focusing on unlocking the emergent modularity in LMs, we showcase that standard LMs could be fine-tuned as their Mixture-of-Expert (MoEs) counterparts without introducing any extra parameters. Such MoEs are derived from emergent modularity and are referred to as Emergent MoEs (EMoE).Our experiments demonstrate that fine-tuning EMoE effectively improves downstream in-domain and out-of-domain generalization compared with vanilla fine-tuning.Our analysis and ablation studies further illustrate that it is robust to various configurations and can scale up to Large Language Models (i.e., Llama2-7B and Llama-30B). Code is available at https://github.com/qiuzh20/EMoE.