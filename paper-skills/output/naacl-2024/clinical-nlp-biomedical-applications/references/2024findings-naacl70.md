---
title: "Re-evaluating the Need for Visual Signals in Unsupervised Grammar Induction"
source: "https://aclanthology.org/2024.findings-naacl.70/"
categories: ['clinical-nlp-biomedical-applications']
tags: ['grammar-induction', 'multimodal', 'unsupervised-learning']
venue: "NAACL 2024"
tldr: "Re-evaluates the necessity of visual signals for unsupervised grammar induction, finding strong text-only baselines can match multimodal performance."
---

# Re-evaluating the Need for Visual Signals in Unsupervised Grammar Induction

**Source**: [https://aclanthology.org/2024.findings-naacl.70/](https://aclanthology.org/2024.findings-naacl.70/)

**TLDR**: Re-evaluates the necessity of visual signals for unsupervised grammar induction, finding strong text-only baselines can match multimodal performance.

## Abstract

AbstractAre multimodal inputs necessary for grammar induction? Recent work has shown that multimodal training inputs can improve grammar induction. However, these improvements are based on comparisons to weak text-only baselines that were trained on relatively little textual data. To determine whether multimodal inputs are needed in regimes with large amounts of textual training data, we design a stronger text-only baseline, which we refer to as LC-PCFG. LC-PCFG is a C-PFCG that incorporates embeddings from text-only large language models (LLMs). We use a fixed grammar family to directly compare LC-PCFG to various multimodal grammar induction methods. We compare performance on four benchmark datasets. LC-PCFG provides an up to 17% relative improvement in Corpus-F1 compared to state-of-the-art multimodal grammar induction methods. LC-PCFG is also more computationally efficient, providing an up to 85% reduction in parameter count and 8.8× reduction in training time compared to multimodal approaches. These results suggest that multimodal inputs may not be necessary for grammar induction, and emphasize the importance of strong vision-free baselines for evaluating the benefit of multimodal approaches.