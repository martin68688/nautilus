---
title: "MEGAVERSE: Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks"
source: "https://aclanthology.org/2024.naacl-long.143/"
categories: ['code-search-clone-detection-graph', 'language-evaluation-benchmarks-foundation-models', 'language-model-cultural-linguistic-evaluation']
tags: ['multilingual-evaluation', 'multimodal', 'benchmarking', 'llm-capabilities']
venue: "NAACL 2024"
tldr: "Benchmarks LLMs across languages, modalities, models, and tasks to address the gap in non-English evaluation."
---

# MEGAVERSE: Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks

**Source**: [https://aclanthology.org/2024.naacl-long.143/](https://aclanthology.org/2024.naacl-long.143/)

**TLDR**: Benchmarks LLMs across languages, modalities, models, and tasks to address the gap in non-English evaluation.

## Abstract

AbstractThere has been a surge in LLM evaluation research to understand LLM capabilities and limitations. However, much of this research has been confined to English, leaving LLM building and evaluation for non-English languages relatively unexplored. Several new LLMs have been introduced recently, necessitating their evaluation on non-English languages. This study aims to perform a thorough evaluation of the non-English capabilities of SoTA LLMs (GPT-3.5-Turbo, GPT-4, PaLM2, Gemini-Pro, Mistral, Llama2, and Gemma) by comparing them on the same set of multilingual datasets. Our benchmark comprises 22 datasets covering 83 languages, including low-resource African languages. We also include two multimodal datasets in the benchmark and compare the performance of LLaVA models, GPT-4-Vision and Gemini-Pro-Vision. Our experiments show that larger models such as GPT-4, Gemini-Pro and PaLM2 outperform smaller models on various tasks, notably on low-resource languages, with GPT-4 outperforming PaLM2 and Gemini-Pro on more datasets. We also perform a study on data contamination and find that several models are likely to be contaminated with multilingual evaluation benchmarks, necessitating approaches to detect and handle contamination while assessing the multilingual performance of LLMs.