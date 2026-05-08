---
title: "Evaluation Dataset for Japanese Medical Text Simplification"
source: "https://aclanthology.org/2024.naacl-srw.23/"
categories: ['clinical-nlp-biomedical-applications', 'llm-ranking-benchmarking-adaptation', 'llm-education-assessment-augmentation']
tags: ['dataset', 'text-simplification', 'japanese', 'medical']
venue: "NAACL 2024"
tldr: "Presents a parallel corpus for Japanese medical text simplification to help patients understand complex medical terms."
---

# Evaluation Dataset for Japanese Medical Text Simplification

**Source**: [https://aclanthology.org/2024.naacl-srw.23/](https://aclanthology.org/2024.naacl-srw.23/)

**TLDR**: Presents a parallel corpus for Japanese medical text simplification to help patients understand complex medical terms.

## Abstract

AbstractWe create a parallel corpus for medical text simplification in Japanese, which simplifies medical terms into expressions that patients can understand without effort.While text simplification in the medial domain is strongly desired by society, it is less explored in Japanese because of the lack of language resources.In this study, we build a parallel corpus for Japanese text simplification evaluation in the medical domain using patients’ weblogs.This corpus consists of 1,425 pairs of complex and simple sentences with or without medical terms.To tackle medical text simplification without a training corpus of the corresponding domain, we repurpose a Japanese text simplification model of other domains.Furthermore, we propose a lexically constrained reranking method that allows to avoid technical terms to be output.Experimental results show that our method contributes to achieving higher simplification performance in the medical domain.