---
title: "Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation"
source: "https://aclanthology.org/2024.naacl-long.122/"
categories: ['efficient-large-model-training-optimization', 'large-language-model-evaluation-augmentation']
tags: ['prompt-optimization', 'baseline', 'sampling']
venue: "NAACL 2024"
tldr: "Demonstrates that randomly sampling tokens as separators can be as effective as language model-generated prompts for optimization."
---

# Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation

**Source**: [https://aclanthology.org/2024.naacl-long.122/](https://aclanthology.org/2024.naacl-long.122/)

**TLDR**: Demonstrates that randomly sampling tokens as separators can be as effective as language model-generated prompts for optimization.

## Abstract

AbstractRecent prompt optimisation approaches use the generative nature of language models to produce prompts – even rivaling the performance of human-curated prompts. In this paper, we demonstrate that randomly sampling tokens from the model vocabulary as “separators” can be as effective as language models for prompt-style text classification. Our experiments show that random separators are competitive baselines, having less than a 1% difference compared to previous self-optimisation methods and showing a 12% average relative improvement over strong human baselines across nine text classification tasks and eight language models. We further analyse this phenomenon in detail using three different random generation strategies, establishing that the language space is rich with potentially good separators, with a greater than 40% average chance that a randomly drawn separator performs better than human-curated separators. These observations challenge the common assumption that an effective prompt should be human readable or task relevant and establish a strong baseline for prompt optimisation research.