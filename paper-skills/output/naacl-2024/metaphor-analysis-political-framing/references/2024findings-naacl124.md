---
title: "When Quantization Affects Confidence of Large Language Models?"
source: "https://aclanthology.org/2024.findings-naacl.124/"
categories: ['efficient-transformer-optimization-techniques', 'metaphor-analysis-political-framing']
tags: ['model-quantization', 'llm-confidence', 'compression']
venue: "NAACL 2024"
tldr: "Analyzes how post-training quantization affects the confidence and performance of large language models."
---

# When Quantization Affects Confidence of Large Language Models?

**Source**: [https://aclanthology.org/2024.findings-naacl.124/](https://aclanthology.org/2024.findings-naacl.124/)

**TLDR**: Analyzes how post-training quantization affects the confidence and performance of large language models.

## Abstract

AbstractRecent studies introduced effective compression techniques for Large Language Models (LLMs) via post-training quantization or low-bit weight representation. Although quantized weights offer storage efficiency and allow for faster inference, existing works have indicated that quantization might compromise performance and exacerbate biases in LLMs.This study investigates the confidence and calibration of quantized models, considering factors such as language model type and scale as contributors to quantization loss.Firstly, we reveal that quantization with GPTQ to 4-bit results in a decrease in confidence regarding true labels, with varying impacts observed among different language models. Secondly, we observe fluctuations in the impact on confidence across different scales. Finally, we propose an explanation for quantization loss based on confidence levels, indicating that quantization disproportionately affects samples where the full model exhibited low confidence levels in the first place.We make our code and quantized models publicly available.