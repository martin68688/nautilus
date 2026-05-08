---
title: "Imagine, Initialize, and Explore: An Effective Exploration Method in Multi-Agent Reinforcement Learning"
source: "https://www.semanticscholar.org/paper/5dce3f2cfe018ada4bebeaa15aac6fe1d01b18ba"
categories: ['reinforcement-learning-bandits-planning-optimization', 'autonomous-driving-multi-agent-navigation']
tags: ['multi-agent-reinforcement-learning', 'exploration', 'coordination']
venue: "AAAI 2024"
tldr: "Introduces an exploration method for MARL that uses imagined latent goals, role-based initialization, and decentralized exploration to improve coordination."
---

# Imagine, Initialize, and Explore: An Effective Exploration Method in Multi-Agent Reinforcement Learning

**Source**: [https://www.semanticscholar.org/paper/5dce3f2cfe018ada4bebeaa15aac6fe1d01b18ba](https://www.semanticscholar.org/paper/5dce3f2cfe018ada4bebeaa15aac6fe1d01b18ba)

**TLDR**: Introduces an exploration method for MARL that uses imagined latent goals, role-based initialization, and decentralized exploration to improve coordination.

## Abstract

Effective exploration is crucial to discovering optimal strategies for multi-agent reinforcement learning (MARL) in complex coordination tasks. Existing methods mainly utilize intrinsic rewards to enable committed exploration or use role-based learning for decomposing joint action spaces instead of directly conducting a collective search in the entire action-observation space. However, they often face challenges obtaining specific joint action sequences to reach successful states in long-horizon tasks. To address this limitation, we propose Imagine, Initialize, and Explore (IIE), a novel method that offers a promising solution for efficient multi-agent exploration in complex scenarios. IIE employs a transformer model to imagine how the agents reach a critical state that can influence each other's transition functions. Then, we initialize the environment at this state using a simulator before the exploration phase. We formulate the imagination as a sequence modeling problem, where the states, observations, prompts, actions, and rewards are predicted autoregressively. The prompt consists of timestep-to-go, return-to-go, influence value, and one-shot demonstration, specifying the desired state and trajectory as well as guiding the action generation. By initializing agents at the critical states, IIE significantly increases the likelihood of discovering potentially important under-explored regions. Despite its simplicity, empirical results demonstrate that our method outperforms multi-agent exploration baselines on the StarCraft Multi-Agent Challenge (SMAC) and SMACv2 environments. Particularly, IIE shows improved performance in the sparse-reward SMAC tasks and produces more effective curricula over the initialized states than other generative methods, such as CVAE-GAN and diffusion models.