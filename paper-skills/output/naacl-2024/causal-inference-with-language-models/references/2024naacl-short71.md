---
title: "DoubleLingo: Causal Estimation with Large Language Models"
source: "https://aclanthology.org/2024.naacl-short.71/"
categories: ['causal-inference-with-language-models']
tags: ['causal-inference', 'llm-estimation', 'confounding']
venue: "NAACL 2024"
tldr: "Proposes DoubleLingo, a method using LLMs for causal effect estimation by adjusting for confounding variables."
---

# DoubleLingo: Causal Estimation with Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-short.71/](https://aclanthology.org/2024.naacl-short.71/)

**TLDR**: Proposes DoubleLingo, a method using LLMs for causal effect estimation by adjusting for confounding variables.

## Abstract

AbstractEstimating causal effects from non-randomized data requires assumptions about the underlying data-generating process. To achieve unbiased estimates of the causal effect of a treatment on an outcome, we typically adjust for any confounding variables that influence both treatment and outcome. When such confounders include text data, existing causal inference methods struggle due to the high dimensionality of the text. The simple statistical models which have sufficient convergence criteria for causal estimation are not well-equipped to handle noisy unstructured text, but flexible large language models that excel at predictive tasks with text data do not meet the statistical assumptions necessary for causal estimation. Our method enables theoretically consistent estimation of causal effects using LLM-based nuisance models by incorporating them within the framework of Double Machine Learning. On the best available dataset for evaluating such methods, we obtain a 10.4% reduction in the relative absolute error for the estimated causal effect over existing methods.