---
title: "Enhancing Off-Policy Constrained Reinforcement Learning through Adaptive Ensemble C Estimation"
source: "https://www.semanticscholar.org/paper/e6e9ed6a0dd86ed71fa552c8fcccb27a487d7ea3"
categories: ['reinforcement-learning-bandits-planning-optimization', 'multi-objective-optimization-and-learning']
tags: ['constrained-rl', 'off-policy', 'ensemble-methods', 'safety']
venue: "AAAI 2024"
tldr: "Enhances off-policy constrained RL via an adaptive ensemble method for more accurate cost estimation."
---

# Enhancing Off-Policy Constrained Reinforcement Learning through Adaptive Ensemble C Estimation

**Source**: [https://www.semanticscholar.org/paper/e6e9ed6a0dd86ed71fa552c8fcccb27a487d7ea3](https://www.semanticscholar.org/paper/e6e9ed6a0dd86ed71fa552c8fcccb27a487d7ea3)

**TLDR**: Enhances off-policy constrained RL via an adaptive ensemble method for more accurate cost estimation.

## Abstract

In the domain of real-world agents, the application of Reinforcement Learning (RL) remains challenging due to the necessity for safety constraints. Previously, Constrained Reinforcement Learning (CRL) has predominantly focused on on-policy algorithms. Although these algorithms exhibit a degree of efficacy, their interactivity efficiency in real-world settings is sub-optimal, highlighting the demand for more efficient off-policy methods. However, off-policy CRL algorithms grapple with challenges in precise estimation of the C-function, particularly due to the fluctuations in the constrained Lagrange multiplier. Addressing this gap, our study focuses on the nuances of C-value estimation in off-policy CRL and introduces the Adaptive Ensemble C-learning (AEC) approach to reduce these inaccuracies. Building on state-of-the-art off-policy algorithms, we propose AEC-based CRL algorithms designed for enhanced task optimization. Extensive experiments on nine constrained robotics tasks reveal the superior interaction efficiency and performance of our algorithms in comparison to preceding methods.