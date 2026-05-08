---
title: "OpenFMNav: Towards Open-Set Zero-Shot Object Navigation via Vision-Language Foundation Models"
source: "https://aclanthology.org/2024.findings-naacl.24/"
categories: ['language-grounded-embodied-navigation']
tags: ['embodied-ai', 'zero-shot', 'vision-language']
venue: "NAACL 2024"
tldr: "Proposes OpenFMNav, an open-set zero-shot object navigation method using vision-language foundation models without task-specific training."
---

# OpenFMNav: Towards Open-Set Zero-Shot Object Navigation via Vision-Language Foundation Models

**Source**: [https://aclanthology.org/2024.findings-naacl.24/](https://aclanthology.org/2024.findings-naacl.24/)

**TLDR**: Proposes OpenFMNav, an open-set zero-shot object navigation method using vision-language foundation models without task-specific training.

## Abstract

AbstractObject navigation (ObjectNav) requires an agent to navigate through unseen environments to find queried objects. Many previous methods attempted to solve this task by relying on supervised or reinforcement learning, where they are trained on limited household datasets with close-set objects. However, two key challenges are unsolved: understanding free-form natural language instructions that demand open-set objects, and generalizing to new environments in a zero-shot manner. Aiming to solve the two challenges, in this paper, we propose **OpenFMNav**, an **Open**-set **F**oundation **M**odel based framework for zero-shot object **Nav**igation. We first unleash the reasoning abilities of large language models (LLMs) to extract proposed objects from natural language instructions that meet the user’s demand. We then leverage the generalizability of large vision language models (VLMs) to actively discover and detect candidate objects from the scene, building a *Versatile Semantic Score Map (VSSM)*. Then, by conducting common sense reasoning on *VSSM*, our method can perform effective language-guided exploration and exploitation of the scene and finally reach the goal. By leveraging the reasoning and generalizing abilities of foundation models, our method can understand free-form human instructions and perform effective open-set zero-shot navigation in diverse environments. Extensive experiments on the HM3D ObjectNav benchmark show that our method surpasses all the strong baselines on all metrics, proving our method’s effectiveness. Furthermore, we perform real robot demonstrations to validate our method’s open-set-ness and generalizability to real-world environments.