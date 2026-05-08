---
title: "Self-Adaptive Sampling for Accurate Video Question Answering on Image Text Models"
source: "https://aclanthology.org/2024.findings-naacl.162/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'large-language-model-evaluation-augmentation']
tags: ['multimodal', 'llm-evaluation', 'optimization']
venue: "NAACL 2024"
tldr: "Proposes a self-adaptive frame sampling method to improve video QA accuracy and efficiency for image-text models."
---

# Self-Adaptive Sampling for Accurate Video Question Answering on Image Text Models

**Source**: [https://aclanthology.org/2024.findings-naacl.162/](https://aclanthology.org/2024.findings-naacl.162/)

**TLDR**: Proposes a self-adaptive frame sampling method to improve video QA accuracy and efficiency for image-text models.

## Abstract

AbstractImage–text models (ITMs) is the prevalent architecture to solve video question–answering tasks, which requires only a few input frames to save huge computational cost compared to video–language models.However, we find existent ITM video question–answering solutions either 1) adopt simplistic and unintentional sampling strategies, which may miss key frames to offer the answer clues; or 2) sample a large number of frames into divided groups, which the computational sources can not accommodate. In this work, we aim at an efficient sampling method towards the few-frame situations.We first summarize a family of prior sampling methods based on question–frame correlation into a unified one, dubbed *Most Implied Frames* (MIF). Through some primary results and analysis, Through analysis, we form a hypothesis that question-aware sampling is not necessary, from which we further propose the other method *Most Dominant Frames* (MDF).Experimental results on four public datasets and three advanced ITMs demonstrate that our proposed strategies can boost the performance for image–text pretrained models, and have a wide application scenario in terms of model architectures and dataset types. Our code is available at https://github.com/declare-lab/Sealinghttps://github.com/declare-lab/Sealing.