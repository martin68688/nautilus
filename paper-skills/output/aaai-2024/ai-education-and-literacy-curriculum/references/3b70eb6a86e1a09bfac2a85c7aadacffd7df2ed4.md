---
title: "Online Reinforcement Learning-Based Pedagogical Planning for Narrative-Centered Learning Environments"
source: "https://www.semanticscholar.org/paper/3b70eb6a86e1a09bfac2a85c7aadacffd7df2ed4"
categories: ['explainable-ai-and-human-collaboration', 'ai-education-and-literacy-curriculum']
tags: ['reinforcement-learning', 'pedagogical-planning', 'adaptive-learning']
venue: "AAAI 2024"
tldr: "Online RL is used for pedagogical planning to provide adaptive support in narrative-centered learning."
---

# Online Reinforcement Learning-Based Pedagogical Planning for Narrative-Centered Learning Environments

**Source**: [https://www.semanticscholar.org/paper/3b70eb6a86e1a09bfac2a85c7aadacffd7df2ed4](https://www.semanticscholar.org/paper/3b70eb6a86e1a09bfac2a85c7aadacffd7df2ed4)

**TLDR**: Online RL is used for pedagogical planning to provide adaptive support in narrative-centered learning.

## Abstract

Pedagogical planners can provide adaptive support to students in narrative-centered learning environments by dynamically scaffolding student learning and tailoring problem scenarios. Reinforcement learning (RL) is frequently used for pedagogical planning in narrative-centered learning environments. However, RL-based pedagogical planning raises significant challenges due to the scarcity of data for training RL policies. Most prior work has relied on limited-size datasets and offline RL techniques for policy learning. Unfortunately, offline RL techniques do not support on-demand exploration and evaluation, which can adversely impact the quality of induced policies. To address the limitation of data scarcity and offline RL, we propose INSIGHT, an online RL framework for training data-driven pedagogical policies that optimize student learning in narrative-centered learning environments. The INSIGHT framework consists of three components: a narrative-centered learning environment simulator, a simulated student agent, and an RL-based pedagogical planner agent, which uses a reward metric that is associated with effective student learning processes. The framework enables the generation of synthetic data for on-demand exploration and evaluation of RL-based pedagogical planning. We have implemented INSIGHT with OpenAI Gym for a narrative-centered learning environment testbed with rule-based simulated student agents and a deep Q-learning-based pedagogical planner. Our results show that online deep RL algorithms can induce near-optimal pedagogical policies in the INSIGHT framework, while offline deep RL algorithms only find suboptimal policies even with large amounts of data.