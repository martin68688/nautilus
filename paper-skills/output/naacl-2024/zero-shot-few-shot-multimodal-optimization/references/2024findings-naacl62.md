---
title: "Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs"
source: "https://aclanthology.org/2024.findings-naacl.62/"
categories: ['llm-knowledge-reasoning-retrieval', 'zero-shot-few-shot-multimodal-optimization', 'human-llm-opinion-dynamics-moderation']
tags: ['vision-language-models', 'reasoning-transfer', 'chart-based', 'capability-transfer']
venue: "NAACL 2024"
tldr: "Transfers reasoning capabilities from LLMs to VLMs using chart-based reasoning tasks."
---

# Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs

**Source**: [https://aclanthology.org/2024.findings-naacl.62/](https://aclanthology.org/2024.findings-naacl.62/)

**TLDR**: Transfers reasoning capabilities from LLMs to VLMs using chart-based reasoning tasks.

## Abstract

AbstractVision-language models (VLMs) are achieving increasingly strong performance on multimodal tasks. However, reasoning capabilities remain limited particularly for smaller VLMs, while those of large-language models (LLMs) have seen numerous improvements. We pro-pose a technique to transfer capabilities from LLMs to VLMs. On the recently introduced ChartQA, our method obtains state-of-the-artperformance when applied on the PaLI3-5B VLM by Chen et al. (2023c), while also enabling much better performance on PlotQA and FigureQA.We first improve the chart representation by continuing the pre-training stage using an improved version of the chart-to-table translation task by Liu et al. (2023a). We then propose constructing a 20x larger dataset than the original training set. To improve general reasoning capabilities and improve numerical operations, we synthesize reasoning traces using the table representation of charts. Lastly, our model is fine-tuned using the multitask loss introduced by Hsieh et al. (2023).Our variant ChartPaLI-5B outperforms even 10x larger models such as PaLIX-55B without using an upstream OCR system, while keeping inference time constant compared to the PaLI3-5B baseline. When rationales are further refined with a simple program-of-thought prompt (Chen et al., 2023a), our model outperforms the recently introduced Gemini Ultra and GPT-4V.