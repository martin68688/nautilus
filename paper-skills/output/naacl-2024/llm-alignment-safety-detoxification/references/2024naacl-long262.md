---
title: "Aligning as Debiasing: Causality-Aware Alignment via Reinforcement Learning with Interventional Feedback"
source: "https://aclanthology.org/2024.naacl-long.262/"
categories: ['llm-backdoor-attacks-defense', 'llm-alignment-safety-detoxification']
tags: ['alignment', 'debiasing', 'causal-inference', 'reinforcement-learning']
venue: "NAACL 2024"
tldr: "A causality-aware alignment method uses reinforcement learning with interventional feedback to reduce bias in LLM outputs by addressing confounding factors."
---

# Aligning as Debiasing: Causality-Aware Alignment via Reinforcement Learning with Interventional Feedback

**Source**: [https://aclanthology.org/2024.naacl-long.262/](https://aclanthology.org/2024.naacl-long.262/)

**TLDR**: A causality-aware alignment method uses reinforcement learning with interventional feedback to reduce bias in LLM outputs by addressing confounding factors.

## Abstract

AbstractLarge language models (LLMs) often generate biased outputs containing offensive, toxic, or stereotypical text. Existing LLM alignment methods such as reinforcement learning from human feedback (RLHF) alleviate biases primarily based on reward signals from current model outputs without considering the source of biases. In this work, to explore how biases are formed, we revisit LLMs’ text generation from a causal perspective. We identify pretraining data and input prompts, which contain semantic correlations of textual phrases, as two confounders between LLMs and model outputs causing biases. Inspired by our causal view, we leverage the reward model in RL alignment as an instrumental variable to perform causal intervention on LLMs. Utilizing the reward difference between an initial LLM and intervened LLM as interventional feedback to guide RL finetuning, we propose Causality-Aware Alignment (CAA) for LLM debiasing. Experiments on two text generation tasks with three different alignment objectives demonstrate the advantages of our method in aligning LLMs to generate less biased and safer outputs.