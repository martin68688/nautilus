---
title: "Visual Grounding Helps Learn Word Meanings in Low-Data Regimes"
source: "https://aclanthology.org/2024.naacl-long.71/"
categories: ['knowledge-graph-and-information-extraction']
tags: ['visual-grounding', 'word-learning', 'low-data', 'language-acquisition']
venue: "NAACL 2024"
tldr: "This paper shows that visual grounding helps neural language models learn word meanings more effectively in low-data regimes."
---

# Visual Grounding Helps Learn Word Meanings in Low-Data Regimes

**Source**: [https://aclanthology.org/2024.naacl-long.71/](https://aclanthology.org/2024.naacl-long.71/)

**TLDR**: This paper shows that visual grounding helps neural language models learn word meanings more effectively in low-data regimes.

## Abstract

AbstractModern neural language models (LMs) are powerful tools for modeling human sentence production and comprehension, and their internal representations are remarkably well-aligned with representations of language in the human brain. But to achieve these results, LMs must be trained in distinctly un-human-like ways — requiring orders of magnitude more language data than children receive during development, and without perceptual or social context. Do models trained more naturalistically — with grounded supervision — exhibit more humanlike language learning? We investigate this question in the context of word learning, a key sub-task in language acquisition. We train a diverse set of LM architectures, with and without auxiliary visual supervision, on datasets of varying scales. We then evaluate these models’ learning of syntactic categories, lexical relations, semantic features, word similarity, and alignment with human neural representations. We find that visual supervision can indeed improve the efficiency of word learning. However, these improvements are limited: they are present almost exclusively in the low-dataregime, and sometimes canceled out by the inclusion of rich distributional signals from text. The information conveyed by text and images isnot redundant—models mainly driven by visual information yield qualitatively different from those mainly driven by word co-occurrences. However, our results suggest that current multimodal modeling approaches fail to effectively leverage visual information to build human-like word representations from human-scale data.