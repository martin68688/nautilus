---
title: "What Are We Measuring When We Evaluate Large Vision-Language Models? An Analysis of Latent Factors and Biases"
source: "https://aclanthology.org/2024.naacl-long.188/"
categories: ['clinical-nlp-biomedical-applications', 'language-model-evaluation-benchmarks']
tags: ['vision-language', 'evaluation', 'benchmark', 'latent-factors']
venue: "NAACL 2024"
tldr: "Analyzing latent factors and biases in vision-language model evaluation through large-scale transfer learning."
---

# What Are We Measuring When We Evaluate Large Vision-Language Models? An Analysis of Latent Factors and Biases

**Source**: [https://aclanthology.org/2024.naacl-long.188/](https://aclanthology.org/2024.naacl-long.188/)

**TLDR**: Analyzing latent factors and biases in vision-language model evaluation through large-scale transfer learning.

## Abstract

AbstractVision-language (VL) models, pretrained on colossal image-text datasets, have attained broad VL competence that is difficult to evaluate. A common belief is that a small number of VL skills underlie the variety of VL tests. In this paper, we perform a large-scale transfer learning experiment aimed at discovering latent VL skills from data. We reveal interesting characteristics that have important implications for test suite design. First, generation tasks suffer from a length bias, suggesting benchmarks should balance tasks with varying output lengths. Second, we demonstrate that factor analysis successfully identifies reasonable yet surprising VL skill factors, suggesting benchmarks could leverage similar analyses for task selection.Finally, we present a new dataset, OLIVE1, which simulates user instructions in the wild and presents challenges dissimilar to all datasets we tested. Our findings contribute to the design of balanced and broad-coverage vision-language evaluation methods. 1https://github.com/jq-zh/olive-dataset