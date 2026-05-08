---
title: "FedAA: A Reinforcement Learning Perspective on Adaptive Aggregation for Fair and Robust Federated Learning"
source: "https://www.semanticscholar.org/paper/6afd39bfa2b721effdf0a3ea460ba1949c03f3ed"
categories: ['federated-learning-optimization-and-robustness']
tags: ['federated-learning', 'robustness', 'fairness', 'reinforcement-learning', 'aggregation']
venue: "AAAI 2024"
tldr: "Uses reinforcement learning to adaptively aggregate client models in federated learning for improved fairness and robustness against attacks."
---

# FedAA: A Reinforcement Learning Perspective on Adaptive Aggregation for Fair and Robust Federated Learning

**Source**: [https://www.semanticscholar.org/paper/6afd39bfa2b721effdf0a3ea460ba1949c03f3ed](https://www.semanticscholar.org/paper/6afd39bfa2b721effdf0a3ea460ba1949c03f3ed)

**TLDR**: Uses reinforcement learning to adaptively aggregate client models in federated learning for improved fairness and robustness against attacks.

## Abstract

Federated Learning (FL) has emerged as a promising approach for privacy-preserving model training across decentralized devices. However, it faces challenges such as statistical heterogeneity and susceptibility to adversarial attacks, which can impact model robustness and fairness. Personalized FL attempts to provide some relief by customizing models for individual clients. However, it falls short in addressing server-side aggregation vulnerabilities. We introduce a novel method called FedAA, which optimizes client contributions via Adaptive Aggregation to enhance model robustness against malicious clients and ensure fairness across participants in non-identically distributed settings. To achieve this goal, we propose an approach involving a Deep Deterministic Policy Gradient-based algorithm for continuous control of aggregation weights, an innovative client selection method based on model parameter distances, and a reward mechanism guided by validation set performance. Empirically, extensive experiments demonstrate that, in terms of robustness, FedAA outperforms the state-of-the-art methods, while maintaining comparable levels of fairness, offering a promising solution to build resilient and fair federated systems.