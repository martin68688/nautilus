---
title: "Show Your Work with Confidence: Confidence Bands for Tuning Curves"
source: "https://aclanthology.org/2024.naacl-long.189/"
categories: ['efficient-transformer-acceleration-techniques', 'large-language-model-evaluation-augmentation']
tags: ['hyperparameter-tuning', 'model-evaluation', 'benchmarking', 'visualization']
venue: "NAACL 2024"
tldr: "Introduces tuning curves with confidence bands to account for tuning effort and ambiguity when comparing NLP methods."
---

# Show Your Work with Confidence: Confidence Bands for Tuning Curves

**Source**: [https://aclanthology.org/2024.naacl-long.189/](https://aclanthology.org/2024.naacl-long.189/)

**TLDR**: Introduces tuning curves with confidence bands to account for tuning effort and ambiguity when comparing NLP methods.

## Abstract

AbstractThe choice of hyperparameters greatly impacts performance in natural language processing. Often, it is hard to tell if a method is better than another or just better tuned. *Tuning curves* fix this ambiguity by accounting for tuning effort. Specifically, they plot validation performance as a function of the number of hyperparameter choices tried so far. While several estimators exist for these curves, it is common to use point estimates, which we show fail silently and give contradictory results when given too little data.Beyond point estimates, *confidence bands* are necessary to rigorously establish the relationship between different approaches. We present the first method to construct valid confidence bands for tuning curves. The bands are exact, simultaneous, and distribution-free, thus they provide a robust basis for comparing methods.Empirical analysis shows that while bootstrap confidence bands, which serve as a baseline, fail to approximate their target confidence, ours achieve it exactly. We validate our design with ablations, analyze the effect of sample size, and provide guidance on comparing models with our method. To promote confident comparisons in future work, we release opda: an easy-to-use library that you can install with pip. https://github.com/nicholaslourie/opda