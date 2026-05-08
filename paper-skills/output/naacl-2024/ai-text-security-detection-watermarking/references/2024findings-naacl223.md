---
title: "WaterJudge: Quality-Detection Trade-off when Watermarking Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.223/"
categories: ['clinical-nlp-biomedical-applications', 'ai-text-security-detection-watermarking']
tags: ['watermarking', 'quality-detection', 'trade-off']
venue: "NAACL 2024"
tldr: "Analyzes the trade-off between output quality and watermark detectability when applying context-dependent watermarks to LLMs."
---

# WaterJudge: Quality-Detection Trade-off when Watermarking Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.223/](https://aclanthology.org/2024.findings-naacl.223/)

**TLDR**: Analyzes the trade-off between output quality and watermark detectability when applying context-dependent watermarks to LLMs.

## Abstract

AbstractWatermarking generative-AI systems, such as LLMs, has gained considerable interest, driven by their enhanced capabilities across a wide range of tasks. Although current approaches have demonstrated that small, context-dependent shifts in the word distributions can be used to apply and detect watermarks, there has been little work in analyzing the impact that these perturbations have on the quality of generated texts. Balancing high detectability with minimal performance degradation is crucial in terms of selecting the appropriate watermarking setting; therefore this paper proposes a simple analysis framework where comparative assessment, a flexible NLG evaluation framework, is used to assess the quality degradation caused by a particular watermark setting. We demonstrate that our framework provides easy visualization of the quality-detection trade-off of watermark settings, enabling a simple solution to find an LLM watermark operating point that provides a well-balanced performance. This approach is applied to two different summarization systems and a translation system, enabling cross-model analysis for a task, and cross-task analysis.