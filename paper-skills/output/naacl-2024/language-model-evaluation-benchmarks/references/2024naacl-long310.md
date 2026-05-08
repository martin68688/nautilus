---
title: "Whispers of Doubt Amidst Echoes of Triumph in NLP Robustness"
source: "https://aclanthology.org/2024.naacl-long.310/"
categories: ['sentiment-emotion-analysis-multimodal-llms', 'language-model-evaluation-benchmarks']
tags: ['robustness', 'model-evaluation', 'scaling']
venue: "NAACL 2024"
tldr: "Investigates whether larger, more performant NLP models resolve long-standing robustness issues using out-of-domain and challenge evaluations."
---

# Whispers of Doubt Amidst Echoes of Triumph in NLP Robustness

**Source**: [https://aclanthology.org/2024.naacl-long.310/](https://aclanthology.org/2024.naacl-long.310/)

**TLDR**: Investigates whether larger, more performant NLP models resolve long-standing robustness issues using out-of-domain and challenge evaluations.

## Abstract

Abstract*Do larger and more performant models resolve NLP’s longstanding robustness issues?* We investigate this question using over 20 models of different sizes spanning different architectural choices and pretraining objectives. We conduct evaluations using (a) out-of-domain and challenge test sets, (b) behavioral testing with CheckLists, (c) contrast sets, and (d) adversarial inputs. Our analysis reveals that not all out-of-domain tests provide insight into robustness. Evaluating with CheckLists and contrast sets shows significant gaps in model performance; merely scaling models does not make them adequately robust. Finally, we point out that current approaches for adversarial evaluations of models are themselves problematic: they can be easily thwarted, and in their current forms, do not represent a sufficiently deep probe of model robustness. We conclude that not only is the question of robustness in NLP as yet unresolved, but even some of the approaches to measure robustness need to be reassessed.