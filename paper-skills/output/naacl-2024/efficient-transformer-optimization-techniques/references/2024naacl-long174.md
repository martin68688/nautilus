---
title: "LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.174/"
categories: ['efficient-transformer-optimization-techniques']
tags: ['parameter-efficient-finetuning', 'tensor-train', 'low-rank']
venue: "NAACL 2024"
tldr: "Introduces LoRETTA, a low-rank tensor-train adaptation method for ultra-low-parameter fine-tuning of large language models."
---

# LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.174/](https://aclanthology.org/2024.naacl-long.174/)

**TLDR**: Introduces LoRETTA, a low-rank tensor-train adaptation method for ultra-low-parameter fine-tuning of large language models.

## Abstract

AbstractVarious parameter-efficient fine-tuning (PEFT) techniques have been proposed to enable computationally efficient fine-tuning while maintaining model performance. However, existing PEFT methods are still limited by the growing number of trainable parameters with the rapid deployment of Large Language Models (LLMs). To address this challenge, we present LoRETTA, an ultra-parameter-efficient framework that significantly reduces trainable parameters through tensor-train decomposition. Specifically, we propose two methods, named LoRETTA_adp and LoRETTA_rep. The former employs tensorized adapters, offering a high-performance yet lightweight approach for the fine-tuning of LLMs. The latter emphasizes fine-tuning via weight reparameterization with a set of small tensor factors. LoRETTA achieves comparable or better performance than most widely used PEFT methods with up to 100× fewer parameters on the LLaMA-2-7B models. Furthermore, empirical results demonstrate that the proposed methods exhibit remarkable anti-overfitting capability, effectively improve training efficiency, and enjoy better multi-task learning performance. Plug-and-play loretta library built upon the Huggingface framework and PEFT library are provided.