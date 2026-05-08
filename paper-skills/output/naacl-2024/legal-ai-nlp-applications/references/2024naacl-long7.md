---
title: "Promptly Predicting Structures: The Return of Inference"
source: "https://aclanthology.org/2024.naacl-long.7/"
categories: ['llm-ranking-adaptation-benchmarking', 'legal-ai-nlp-applications']
tags: ['prompt-based', 'structured-prediction', 'inference']
venue: "NAACL 2024"
tldr: "Explores prompt-based methods for zero- and few-shot structured prediction tasks without extensive annotation."
---

# Promptly Predicting Structures: The Return of Inference

**Source**: [https://aclanthology.org/2024.naacl-long.7/](https://aclanthology.org/2024.naacl-long.7/)

**TLDR**: Explores prompt-based methods for zero- and few-shot structured prediction tasks without extensive annotation.

## Abstract

AbstractPrompt-based methods have been used extensively across NLP to build zero- and few-shot label predictors. Many NLP tasks are naturally structured: that is, their outputs consist of multiple labels which constrain each other. Annotating data for such tasks can be cumbersome. Can the promise of the prompt-based paradigm be extended to such structured outputs? In this paper, we present a framework for constructing zero- and few-shot linguistic structure predictors. Our key insight is that we can use structural constraints—and combinatorial inference derived from them—to filter out inconsistent structures predicted by large language models. We instantiated this framework on two structured prediction tasks, and five datasets. Across all cases, our results show that enforcing consistency not only constructs structurally valid outputs, but also improves performance over the unconstrained variants.