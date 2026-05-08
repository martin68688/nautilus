---
title: "Efficient End-to-End Visual Document Understanding with Rationale Distillation"
source: "https://aclanthology.org/2024.naacl-long.465/"
categories: ['efficient-transformer-optimization-techniques', 'large-language-model-evaluation-augmentation']
tags: ['multimodal', 'document-understanding', 'distillation']
venue: "NAACL 2024"
tldr: "Presents an efficient end-to-end model for visual document understanding that distills layout reasoning from an LLM."
---

# Efficient End-to-End Visual Document Understanding with Rationale Distillation

**Source**: [https://aclanthology.org/2024.naacl-long.465/](https://aclanthology.org/2024.naacl-long.465/)

**TLDR**: Presents an efficient end-to-end model for visual document understanding that distills layout reasoning from an LLM.

## Abstract

AbstractUnderstanding visually situated language requires interpreting complex layouts of textual and visual elements. Pre-processing tools, such as optical character recognition (OCR), can map document image inputs to textual tokens, then large language models (LLMs) can reason over text.However, such methods have high computational and engineering complexity. Can small pretrained image-to-text models accurately understand visual documents through similar recognition and reasoning steps instead?We propose Rationale Distillation (RD), which incorporates the outputs of OCR tools, LLMs, and larger multimodal models as intermediate “rationales”, and trains a small student model to predict both rationales and answers. On three visual document understanding benchmarks representing infographics, scanned documents, and figures, our Pix2Struct (282M parameters) student model finetuned with RD outperforms the base model by 4-5% absolute accuracy with only 1% higher computational cost.