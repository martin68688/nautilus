---
title: "A Systematic Comparison of Contextualized Word Embeddings for Lexical Semantic Change"
source: "https://aclanthology.org/2024.naacl-long.240/"
categories: ['human-llm-opinion-dynamics-moderation', 'social-media-linguistic-variation']
tags: ['word-embeddings', 'lexical-semantic-change', 'evaluation']
venue: "NAACL 2024"
tldr: "Systematically compares contextualized word embeddings for the task of lexical semantic change."
---

# A Systematic Comparison of Contextualized Word Embeddings for Lexical Semantic Change

**Source**: [https://aclanthology.org/2024.naacl-long.240/](https://aclanthology.org/2024.naacl-long.240/)

**TLDR**: Systematically compares contextualized word embeddings for the task of lexical semantic change.

## Abstract

AbstractContextualized embeddings are the preferred tool for modeling Lexical Semantic Change (LSC). Current evaluations typically focus on a specific task known as Graded Change Detection (GCD). However, performance comparison across work are often misleading due to their reliance on diverse settings. In this paper, we evaluate state-of-the-art models and approaches for GCD under equal conditions. We further break the LSC problem into Word-in-Context (WiC) and Word Sense Induction (WSI) tasks, and compare models across these different levels. Our evaluation is performed across different languages on eight available benchmarks for LSC, and shows that (i) APD outperforms other approaches for GCD; (ii) XL-LEXEME outperforms other contextualized models for WiC, WSI, and GCD, while being comparable to GPT-4; (iii) there is a clear need for improving the modeling of word meanings, as well as focus on *how*, *when*, and *why* these meanings change, rather than solely focusing on the extent of semantic change.