---
title: "Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks"
source: "https://aclanthology.org/2024.naacl-long.102/"
categories: ['efficient-transformer-optimization-techniques', 'logical-reasoning-in-neural-models']
tags: ['evaluation', 'reasoning', 'generalization']
venue: "NAACL 2024"
tldr: "Investigates the abstract reasoning capabilities of LMs by evaluating them on counterfactual tasks to determine if skills are general or task-specific."
---

# Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks

**Source**: [https://aclanthology.org/2024.naacl-long.102/](https://aclanthology.org/2024.naacl-long.102/)

**TLDR**: Investigates the abstract reasoning capabilities of LMs by evaluating them on counterfactual tasks to determine if skills are general or task-specific.

## Abstract

AbstractThe impressive performance of recent language models across a wide range of tasks suggests that they possess a degree of abstract reasoning skills. Are these skills general and transferable, or specialized to specific tasks seen during pretraining? To disentangle these effects, we propose an evaluation framework based on “counterfactual” task variants that deviate from the default assumptions underlying standard tasks. Across a suite of 11 tasks, we observe nontrivial performance on the counterfactual variants, but nevertheless find that performance substantially and consistently degrades compared to the default conditions. This suggests that while current LMs may possess abstract task-solving skills to an extent, they often also rely on narrow, non-transferable procedures for task-solving. These results motivate a more careful interpretation of language model performance that teases apart these aspects.