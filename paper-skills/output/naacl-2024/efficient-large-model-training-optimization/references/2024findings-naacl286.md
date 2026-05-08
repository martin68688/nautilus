---
title: "Personalized Federated Learning for Text Classification with Gradient-Free Prompt Tuning"
source: "https://aclanthology.org/2024.findings-naacl.286/"
categories: ['efficient-large-model-training-optimization', 'code-search-clone-detection']
tags: ['federated-learning', 'personalization', 'prompt-tuning', 'text-classification']
venue: "NAACL 2024"
tldr: "Personalized federated learning for text classification using gradient-free prompt tuning to reduce communication costs."
---

# Personalized Federated Learning for Text Classification with Gradient-Free Prompt Tuning

**Source**: [https://aclanthology.org/2024.findings-naacl.286/](https://aclanthology.org/2024.findings-naacl.286/)

**TLDR**: Personalized federated learning for text classification using gradient-free prompt tuning to reduce communication costs.

## Abstract

AbstractIn this paper, we study personalized federated learning for text classification with Pretrained Language Models (PLMs). We identify two challenges in efficiently leveraging PLMs for personalized federated learning: 1) Communication. PLMs are usually large in size, e.g., with hundreds of millions of parameters, inducing huge communication cost in a federated setting. 2) Local Training. Training with PLMs generally requires back-propagation, during which memory consumption can be several times that of the forward-propagation. This may not be affordable when the PLMs are trained locally on the clients that are resource constrained, e.g., mobile devices with limited access to memory resources. Additionally, the proprietary PLMs can be provided as concealed APIs, for which the back-propagation operations may not be available. In solving these, we propose a training framework that includes an approach of discrete local search for gradient-free local training, along with a compression mechanism inspired from the linear word analogy that allows communicating with discretely indexed tokens, thus significantly reducing the communication cost. Experiments show that our gradient-free framework achieves superior performance compared with baselines.