---
title: "i-Code V2: An Autoregressive Generation Framework over Vision, Language, and Speech Data"
source: "https://aclanthology.org/2024.findings-naacl.105/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'sentiment-emotion-analysis-multimodal-llms']
tags: ['multimodal', 'generative', 'autoregressive', 'vision-language-speech']
venue: "NAACL 2024"
tldr: "An autoregressive generation framework (i-Code V2) unifying vision, language, and speech data."
---

# i-Code V2: An Autoregressive Generation Framework over Vision, Language, and Speech Data

**Source**: [https://aclanthology.org/2024.findings-naacl.105/](https://aclanthology.org/2024.findings-naacl.105/)

**TLDR**: An autoregressive generation framework (i-Code V2) unifying vision, language, and speech data.

## Abstract

AbstractThe convergence of text, visual, and audio data is crucial towards human-like artificial intelligence, however the current Vision-Language-Speech landscape is dominated by encoder-only models that lack generative abilities. We propose closing this gap with i-Code V2, one of the first models capable of generating natural language from any combination of Vision, Language, and Speech data. i-Code V2 leverages state-of-the-art single-modality encoders, combining their outputs with a new modality-fusing encoder to project combinations of modalities into a shared representational space. Language tokens are generated from these representations via an autoregressive decoder. i-Code V2 is pretrained end-to-end on a large collection of dual- and single-modality datasets with a novel text completion objective that can be generalized across arbitrary combinations of modalities. i-Code V2 matches or outperforms state-of-the-art single- and dual-modality baselines on 7 multimodal tasks, demonstrating the power of generative multimodal pretraining across a diversity of tasks and signals.