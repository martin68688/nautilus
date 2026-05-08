---
title: "The Distributional Reward Critic Framework for Reinforcement Learning Under Perturbed Rewards"
source: "https://www.semanticscholar.org/paper/3124f59f1870b303c9abffcbae14f18ecfcedac7"
categories: ['reinforcement-learning-bandits-planning-optimization', 'federated-learning-optimization-and-robustness']
tags: ['reinforcement-learning', 'distributional-robustness', 'perturbed-rewards']
venue: "AAAI 2024"
tldr: "A distributional reward critic framework for reinforcement learning under perturbed or noisy rewards."
---

# The Distributional Reward Critic Framework for Reinforcement Learning Under Perturbed Rewards

**Source**: [https://www.semanticscholar.org/paper/3124f59f1870b303c9abffcbae14f18ecfcedac7](https://www.semanticscholar.org/paper/3124f59f1870b303c9abffcbae14f18ecfcedac7)

**TLDR**: A distributional reward critic framework for reinforcement learning under perturbed or noisy rewards.

## Abstract

The reward signal plays a central role in defining the desired behaviors of agents in reinforcement learning (RL). Rewards collected from realistic environments could be perturbed, corrupted, or noisy due to an adversary, sensor error, or because they come from subjective human feedback. Thus, it is important to construct agents that can learn under such rewards. Existing methodologies for this problem make strong assumptions, including that the perturbation is known in advance, clean rewards are accessible, or that the perturbation preserves the optimal policy. We study a new, more general, class of unknown perturbations, and introduce a distributional reward critic framework for estimating reward distributions and perturbations during training. Our proposed methods are compatible with any RL algorithm. Despite their increased generality, we show that they achieve comparable or better rewards than existing methods in a variety of environments, including those with clean rewards. Under the challenging and generalized perturbations we study, we win/tie the highest return in 44/48 tested settings (compared to 11/48 for the best baseline). Our results broaden and deepen our ability to perform RL in reward-perturbed environments.