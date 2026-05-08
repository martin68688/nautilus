---
title: "Lower Bounds on the Expressivity of Recurrent Neural Language Models"
source: "https://aclanthology.org/2024.naacl-long.380/"
categories: ['efficient-transformer-optimization-techniques', 'transformer-language-model-probing']
tags: ['expressivity', 'lower-bounds', 'recurrent-neural-networks', 'language-models']
venue: "NAACL 2024"
tldr: "Establishes lower bounds on the expressivity of recurrent neural language models to better understand their representational capacity."
---

# Lower Bounds on the Expressivity of Recurrent Neural Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.380/](https://aclanthology.org/2024.naacl-long.380/)

**TLDR**: Establishes lower bounds on the expressivity of recurrent neural language models to better understand their representational capacity.

## Abstract

AbstractThe recent successes and spread of large neural language models (LMs) call for a thorough understanding of their abilities. Describing their abilities through LMs’ representational capacity is a lively area of research. Investigations of the representational capacity of neural LMs have predominantly focused on their ability to recognize formal languages. For example, recurrent neural networks (RNNs) as classifiers are tightly linked to regular languages, i.e., languages defined by finite-state automata (FSAs). Such results, however, fall short of describing the capabilities of RNN language models (LMs), which are definitionally distributions over strings. We take a fresh look at the represen- tational capacity of RNN LMs by connecting them to probabilistic FSAs and demonstrate that RNN LMs with linearly bounded precision can express arbitrary regular LMs.