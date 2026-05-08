---
title: "An Implicit Trust Region Approach to Behavior Regularized Offline Reinforcement Learning"
source: "https://www.semanticscholar.org/paper/ca6eee8ac3f980885aa6ea232b8c83bab1fd37ea"
categories: ['machine-learning-optimization-generalization', 'reinforcement-learning-bandits-planning-optimization']
tags: ['offline-reinforcement-learning', 'trust-region', 'behavior-regularization', 'reward-shaping']
venue: "AAAI 2024"
tldr: "Introduces an implicit trust region approach via reward shaping to stabilize behavior regularization in offline RL."
---

# An Implicit Trust Region Approach to Behavior Regularized Offline Reinforcement Learning

**Source**: [https://www.semanticscholar.org/paper/ca6eee8ac3f980885aa6ea232b8c83bab1fd37ea](https://www.semanticscholar.org/paper/ca6eee8ac3f980885aa6ea232b8c83bab1fd37ea)

**TLDR**: Introduces an implicit trust region approach via reward shaping to stabilize behavior regularization in offline RL.

## Abstract

We revisit behavior regularization, a popular approach to mitigate the extrapolation error in offline reinforcement learning (RL), showing that current behavior regularization may suffer from unstable learning and hinder policy improvement. Motivated by this, a novel reward shaping-based behavior regularization method is proposed, where the log-probability ratio between the learned policy and the behavior policy is monitored during learning. We show that this is equivalent to an implicit but computationally lightweight trust region mechanism, which is beneficial to mitigate the influence of estimation errors of the value function, leading to more stable performance improvement. Empirical results on the popular D4RL benchmark verify the effectiveness of the presented method with promising performance compared with some state-of-the-art offline RL algorithms.