---
title: "Partial Identifiability in Inverse Reinforcement Learning For Agents With Non-Exponential Discounting"
source: "https://www.semanticscholar.org/paper/cd7a8ffbf010fbae83c8c7e57e1e27a66cbbc1e8"
categories: ['reasoning-learning-optimization-in-llms']
tags: ['inverse-rl', 'partial-identifiability', 'discounting']
venue: "AAAI 2024"
tldr: "Analyzes partial identifiability of preferences in inverse RL for agents with non-exponential discounting."
---

# Partial Identifiability in Inverse Reinforcement Learning For Agents With Non-Exponential Discounting

**Source**: [https://www.semanticscholar.org/paper/cd7a8ffbf010fbae83c8c7e57e1e27a66cbbc1e8](https://www.semanticscholar.org/paper/cd7a8ffbf010fbae83c8c7e57e1e27a66cbbc1e8)

**TLDR**: Analyzes partial identifiability of preferences in inverse RL for agents with non-exponential discounting.

## Abstract

The aim of inverse reinforcement learning (IRL) is to infer an agent's preferences from observing their behaviour. Usually, preferences are modelled as a reward function, R, and behaviour is modelled as a policy, pi. One of the central difficulties in IRL is that multiple preferences may lead to the same observed behaviour. That is, R is typically underdetermined by pi, which means that R is only partially identifiable. Recent work has characterised the extent of this partial identifiability for different types of agents, including optimal and Boltzmann-rational agents. However, work so far has only considered agents that discount future reward exponentially: this is a serious limitation, especially given that extensive work in the behavioural sciences suggests that humans are better modelled as discounting hyperbolically. In this work, we newly characterise partial identifiability in IRL for agents with non-exponential discounting: our results are in particular relevant for hyperbolical discounting, but they also more generally apply to agents that use other types of (non-exponential) discounting. We significantly show that generally IRL is unable to infer enough information about R to identify the correct optimal policy, which entails that IRL alone can be insufficient to adequately characterise the preferences of such agents.