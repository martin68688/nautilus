---
title: "VisLingInstruct: Elevating Zero-Shot Learning in Multi-Modal Language Models with Autonomous Instruction Optimization"
source: "https://aclanthology.org/2024.naacl-long.117/"
categories: ['knowledge-graph-and-information-extraction', 'zero-shot-few-shot-multimodal-optimization']
tags: ['multimodal', 'zero-shot', 'instruction-optimization']
venue: "NAACL 2024"
tldr: "Optimizes instructions autonomously to improve zero-shot learning in multi-modal language models."
---

# VisLingInstruct: Elevating Zero-Shot Learning in Multi-Modal Language Models with Autonomous Instruction Optimization

**Source**: [https://aclanthology.org/2024.naacl-long.117/](https://aclanthology.org/2024.naacl-long.117/)

**TLDR**: Optimizes instructions autonomously to improve zero-shot learning in multi-modal language models.

## Abstract

AbstractThis paper presents VisLingInstruct, a novel approach to advancing Multi-Modal Language Models (MMLMs) in zero-shot learning. Current MMLMs show impressive zero-shot abilities in multi-modal tasks, but their performance depends heavily on the quality of instructions. VisLingInstruct tackles this by autonomously evaluating and optimizing instructional texts through In-Context Learning, improving the synergy between visual perception and linguistic expression in MMLMs. Alongside this instructional advancement, we have also optimized the visual feature extraction modules in MMLMs, further augmenting their responsiveness to textual content. Our comprehensive experiments on MMLMs, based on FlanT5 and Vicuna, show that VisLingInstruct significantly improves zero-shot performance in visual multi-modal tasks. Notably, it achieves a 13.1% and 9% increase in accuracy over the prior state-of-the-art on the TextVQA and HatefulMemes datasets. Our main code is available at https://github.com/Zhudongsheng75/VisLingInstruct