---
title: "Multi-Operational Mathematical Derivations in Latent Space"
source: "https://aclanthology.org/2024.naacl-long.80/"
categories: ['latent-space-mathematical-derivations']
tags: ['latent-space', 'mathematical-derivation', 'geometric-transformation']
venue: "NAACL 2024"
tldr: "This paper introduces multi-operational representation paradigms that model mathematical operations as explicit geometric transformations in latent space for approximating expression derivations."
---

# Multi-Operational Mathematical Derivations in Latent Space

**Source**: [https://aclanthology.org/2024.naacl-long.80/](https://aclanthology.org/2024.naacl-long.80/)

**TLDR**: This paper introduces multi-operational representation paradigms that model mathematical operations as explicit geometric transformations in latent space for approximating expression derivations.

## Abstract

AbstractThis paper investigates the possibility of approximating multiple mathematical operations in latent space for expression derivation. To this end, we introduce different multi-operational representation paradigms, modelling mathematical operations as explicit geometric transformations. By leveraging a symbolic engine, we construct a large-scale dataset comprising 1.7M derivation steps stemming from 61K premises and 6 operators, analysing the properties of each paradigm when instantiated with state-of-the-art neural encoders.Specifically, we investigate how different encoding mechanisms can approximate expression manipulation in latent space, exploring the trade-off between learning different operators and specialising within single operations, as well as the ability to support multi-step derivations and out-of-distribution generalisation. Our empirical analysis reveals that the multi-operational paradigm is crucial for disentangling different operators, while discriminating the conclusions for a single operation is achievable in the original expression encoder. Moreover, we show that architectural choices can heavily affect the training dynamics, structural organisation, and generalisation of the latent space, resulting in significant variations across paradigms and classes of encoders.