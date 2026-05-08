---
title: "Quantitative Predictive Monitoring and Control for Safe Human-Machine Interaction"
source: "https://www.semanticscholar.org/paper/edab555ce0dc70a82ac2dedc2e3ed11c91bd58a9"
categories: ['explainable-ai-and-human-collaboration', 'fair-division-matching-mechanism-design', 'digital-twin-driven-teat-localization']
tags: ['predictive-monitoring', 'human-machine-interaction', 'safety', 'control']
venue: "AAAI 2024"
tldr: "Predicts future states for safe human-machine interaction using quantitative monitoring and control."
---

# Quantitative Predictive Monitoring and Control for Safe Human-Machine Interaction

**Source**: [https://www.semanticscholar.org/paper/edab555ce0dc70a82ac2dedc2e3ed11c91bd58a9](https://www.semanticscholar.org/paper/edab555ce0dc70a82ac2dedc2e3ed11c91bd58a9)

**TLDR**: Predicts future states for safe human-machine interaction using quantitative monitoring and control.

## Abstract

There is a growing trend toward AI systems interacting with humans to revolutionize a range of application domains such as healthcare and transportation. However, unsafe human-machine interaction can lead to catastrophic failures. We propose a novel approach that predicts future states by accounting for the uncertainty of human interaction, monitors whether predictions satisfy or violate safety requirements, and adapts control actions based on the predictive monitoring results. Specifically, we develop a new quantitative predictive monitor based on Signal Temporal Logic with Uncertainty (STL-U) to compute a robustness degree interval, which indicates the extent to which a sequence of uncertain predictions satisfies or violates an STL-U requirement. We also develop a new loss function to guide the uncertainty calibration of Bayesian deep learning and a new adaptive control method, both of which leverage STL-U quantitative predictive monitoring results. We apply the proposed approach to two case studies: Type 1 Diabetes management and semi-autonomous driving. Experiments show that the proposed approach improves safety and effectiveness in both case studies.