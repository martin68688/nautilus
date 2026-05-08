---
title: "Toward Interactive Regional Understanding in Vision-Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.356/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'llm-alignment-safety-detoxification']
tags: ['vision-language', 'regional-understanding', 'interactive', 'vlp']
venue: "NAACL 2024"
tldr: "Enhancing regional understanding in vision-language models through interactive methods beyond coarse image-text pairs."
---

# Toward Interactive Regional Understanding in Vision-Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.356/](https://aclanthology.org/2024.naacl-long.356/)

**TLDR**: Enhancing regional understanding in vision-language models through interactive methods beyond coarse image-text pairs.

## Abstract

AbstractRecent Vision-Language Pre-training (VLP) models have demonstrated significant advancements. Nevertheless, these models heavily rely on image-text pairs that capture only coarse and global information of an image, leading to a limitation in their regional understanding ability. In this work, we introduce RegionVLM, equipped with explicit regional modeling capabilities, allowing them to understand user-indicated image regions. To achieve this, we design a simple yet innovative architecture, requiring no modifications to the model architecture or objective function. Additionally, we leverage a dataset that contains a novel source of information, namely Localized Narratives, which has been overlooked in previous VLP research. Our experiments demonstrate that our single generalist model not only achieves an interactive dialogue system but also exhibits superior performance on various zero-shot region understanding tasks, without compromising its ability for global image understanding.