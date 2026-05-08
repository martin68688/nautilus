---
title: "Explanation Extraction from Hierarchical Classification Frameworks for Long Legal Documents"
source: "https://aclanthology.org/2024.findings-naacl.76/"
categories: ['large-language-model-evaluation-augmentation', 'legal-ai-nlp-applications']
tags: ['explainable-ai', 'legal-nlp', 'hierarchical-classification']
venue: "NAACL 2024"
tldr: "Develops a method to extract explanations from hierarchical classification models for long legal documents to improve transparency."
---

# Explanation Extraction from Hierarchical Classification Frameworks for Long Legal Documents

**Source**: [https://aclanthology.org/2024.findings-naacl.76/](https://aclanthology.org/2024.findings-naacl.76/)

**TLDR**: Develops a method to extract explanations from hierarchical classification models for long legal documents to improve transparency.

## Abstract

AbstractHierarchical classification frameworks have been widely used to process long sequences, especially in the legal domain for predictions from long legal documents. But being black-box models they are unable to explain their predictions making them less reliable for practical applications, more so in the legal domain. In this work, we develop an extractive explanation algorithm for hierarchical frameworks for long sequences based on the sensitivity of the trained model to its input perturbations. We perturb using occlusion and develop Ob-HEx; an Occlusion-based Hierarchical Explanation-extractor. We adapt Ob-HEx to Hierarchical Transformer models trained on long Indian legal texts. And use Ob-HEx to analyze them and extract their explanations for the ILDC-Expert dataset, achieving a minimum gain of 1 point over the previous benchmark on most of our performance evaluation metrics.