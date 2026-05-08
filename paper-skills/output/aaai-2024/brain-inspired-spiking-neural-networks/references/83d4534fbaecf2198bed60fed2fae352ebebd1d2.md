---
title: "Practicable Black-box Evasion Attacks on Link Prediction in Dynamic Graphs - A Graph Sequential Embedding Method"
source: "https://www.semanticscholar.org/paper/83d4534fbaecf2198bed60fed2fae352ebebd1d2"
categories: ['graph-learning-clustering-multiview-federated', 'brain-inspired-spiking-neural-networks']
tags: ['link-prediction', 'dynamic-graphs', 'evasion-attack', 'black-box']
venue: "AAAI 2024"
tldr: "Proposes a black-box evasion attack on link prediction in dynamic graphs using a graph sequential embedding method."
---

# Practicable Black-box Evasion Attacks on Link Prediction in Dynamic Graphs - A Graph Sequential Embedding Method

**Source**: [https://www.semanticscholar.org/paper/83d4534fbaecf2198bed60fed2fae352ebebd1d2](https://www.semanticscholar.org/paper/83d4534fbaecf2198bed60fed2fae352ebebd1d2)

**TLDR**: Proposes a black-box evasion attack on link prediction in dynamic graphs using a graph sequential embedding method.

## Abstract

Link prediction in dynamic graphs (LPDG) has been widely applied to real-world applications such as website recommendation, traffic flow prediction, organizational studies, etc. These models are usually kept local and secure, with only the interactive interface restrictively available to the public. Thus, the problem of the black-box evasion attack on the LPDG model, where model interactions and data perturbations are restricted, seems to be essential and meaningful in practice. In this paper, we propose the first practicable black-box evasion attack method that achieves effective attacks against the target LPDG model, within a limited amount of interactions and perturbations. To perform effective attacks under limited perturbations, we develop a graph sequential embedding model to find the desired state embedding of the dynamic graph sequences, under a deep reinforcement learning framework. To overcome the scarcity of interactions, we design a multi-environment training pipeline and train our agent for multiple instances, by sharing an aggregate interaction buffer. Finally, we evaluate our attack against three advanced LPDG models on three real-world graph datasets of different scales and compare its performance with related methods under the interaction and perturbation constraints. Experimental results show that our attack is both effective and practicable.