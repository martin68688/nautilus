---
title: "RS-DPO: A Hybrid Rejection Sampling and Direct Preference Optimization Method for Alignment of Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.108/"
categories: ['llm-alignment-safety-detoxification']
tags: ['alignment', 'rlhf', 'dpo', 'rejection-sampling']
venue: "NAACL 2024"
tldr: "Proposes a hybrid method combining rejection sampling and direct preference optimization for more stable and efficient LLM alignment."
---

# RS-DPO: A Hybrid Rejection Sampling and Direct Preference Optimization Method for Alignment of Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.108/](https://aclanthology.org/2024.findings-naacl.108/)

**TLDR**: Proposes a hybrid method combining rejection sampling and direct preference optimization for more stable and efficient LLM alignment.

## Abstract

AbstractReinforcement learning from human feedback (RLHF) has been extensively employed to align large language models with user intent. However, proximal policy optimization (PPO) based RLHF is occasionally unstable requiring significant hyperparameter finetuning, and computationally expensive to maximize the estimated reward during alignment. Recently, direct preference optimization (DPO) is proposed to address those challenges. However, DPO often relies on contrastive responses generated from human annotator and alternative LLM, instead of the policy model, limiting the effectiveness of the RLHF. In this paper, we addresses both challenges by systematically combining rejection sampling (RS) and DPO. Our proposed method, RS-DPO, initiates with the development of a supervised fine-tuned policy model (SFT). A varied set of k responses per prompt are sampled directly from the SFT model. RS-DPO identifies pairs of contrastive samples based on their reward distribution. Finally, we apply DPO with the contrastive samples to align the model to human preference. Our experiments indicate that our proposed method effectively fine-tunes LLMs with limited resource environments, leading to improved alignment with user intent. Furthermore, it outperforms existing methods, including RS, PPO, and DPO.