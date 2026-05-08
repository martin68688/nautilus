---
title: "Do Vision-Language Models Understand Compound Nouns?"
source: "https://aclanthology.org/2024.naacl-short.43/"
categories: ['efficient-transformer-optimization-techniques', 'zero-shot-few-shot-multimodal-optimization']
tags: ['vision-language-models', 'compound-nouns', 'retrieval']
venue: "NAACL 2024"
tldr: "Evaluates whether vision-language models understand compound nouns by curating a benchmark and analyzing retrieval performance."
---

# Do Vision-Language Models Understand Compound Nouns?

**Source**: [https://aclanthology.org/2024.naacl-short.43/](https://aclanthology.org/2024.naacl-short.43/)

**TLDR**: Evaluates whether vision-language models understand compound nouns by curating a benchmark and analyzing retrieval performance.

## Abstract

AbstractOpen-vocabulary vision-language models (VLMs) like CLIP, trained using contrastive loss, have emerged as a promising new paradigm for text-to-image retrieval. However, do VLMs understand compound nouns (CNs) (e.g., *lab coat*) as well as they understand nouns (e.g., *lab*)? We curate Compun, a novel benchmark with 400 unique and commonly used CNs, to evaluate the effectiveness of VLMs in interpreting CNs. The Compun benchmark challenges a VLM for text-to-image retrieval where, given a text prompt with a CN, the task is to select the correct image that shows the CN among a pair of distractor images that show the constituent nouns that make up the CN. Next, we perform an in-depth analysis to highlight CLIPs’ limited understanding of certain types of CNs. Finally, we present an alternative framework that moves beyond hand-written templates for text prompts widely used by CLIP-like models. We employ a Large Language Model to generate multiple diverse captions that include the CN as an object in the scene described by the caption. Our proposed method improves CN understanding of CLIP by 8.25% on Compun. Code and benchmark are available.