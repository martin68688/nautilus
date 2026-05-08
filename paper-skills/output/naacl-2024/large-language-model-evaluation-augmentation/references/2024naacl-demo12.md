---
title: "LMFlow: An Extensible Toolkit for Finetuning and Inference of Large Foundation Models"
source: "https://aclanthology.org/2024.naacl-demo.12/"
categories: ['efficient-large-model-training-optimization', 'llm-evaluation-summarization-argument-extraction', 'large-language-model-evaluation-augmentation']
tags: ['toolkit', 'finetuning', 'inference', 'foundation-models']
venue: "NAACL 2024"
tldr: "An extensible toolkit is presented for finetuning and inference of large foundation models, enabling efficient experimentation and adaptation."
---

# LMFlow: An Extensible Toolkit for Finetuning and Inference of Large Foundation Models

**Source**: [https://aclanthology.org/2024.naacl-demo.12/](https://aclanthology.org/2024.naacl-demo.12/)

**TLDR**: An extensible toolkit is presented for finetuning and inference of large foundation models, enabling efficient experimentation and adaptation.

## Abstract

AbstractFoundation models have demonstrated a great ability to achieve general human-level intelligence far beyond traditional approaches. As the technique keeps attracting attention from the AI community, more and more foundation models have become publicly available.However, most of those models exhibit a major deficiency in specialized-domain and specialized-task applications, where the step of domain- and task-aware finetuning is still required to obtain scientific language models. As the number of available foundation models and specialized tasks keeps growing, the job of training scientific language models becomes highly nontrivial. In this paper, we take the first step to address this issue. We introduce an extensible and lightweight toolkit, LMFlow, which aims to simplify the domain- and task-aware finetuning of general foundation models.LMFlow offers a complete finetuning workflow for a foundation model to support specialized training with limited computing resources.Furthermore, it supports continuous pretraining, instruction tuning, parameter-efficient finetuning, alignment tuning, inference acceleration, long context generalization, model customization, and even multimodal finetuning, along with carefully designed and extensible APIs. This toolkit has been thoroughly tested and is available at https://github.com/OptimalScale/LMFlow.