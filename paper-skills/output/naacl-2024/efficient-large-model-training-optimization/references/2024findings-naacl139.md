---
title: "Plug-in Language Model: Controlling Text Generation with a Simple Regression Model"
source: "https://aclanthology.org/2024.findings-naacl.139/"
categories: ['efficient-large-model-training-optimization', 'code-search-clone-detection']
tags: ['controlled-generation', 'language-model', 'plug-in', 'regression']
venue: "NAACL 2024"
tldr: "Controlling text generation from pre-trained LMs using a simple plug-in regression model without fine-tuning."
---

# Plug-in Language Model: Controlling Text Generation with a Simple Regression Model

**Source**: [https://aclanthology.org/2024.findings-naacl.139/](https://aclanthology.org/2024.findings-naacl.139/)

**TLDR**: Controlling text generation from pre-trained LMs using a simple plug-in regression model without fine-tuning.

## Abstract

AbstractLarge-scale pre-trained language models have displayed unrivaled capacity in generating text that closely resembles human-written text. Nevertheless, generating texts adhering to specific conditions without fine-tuning or adding new parameters can be challenging. Contemporary approaches commonly rely on either prompts or auxiliary models to avoid modifying the language models. These auxiliary models are designed to assess whether a generated token contributes to meeting the desired requirements. These approaches adjust the distribution of the next token during the inference phase by leveraging the prediction score of the desired attribute to calculate gradients. However, these auxiliary models typically require the language model’s latent states. This prerequisite challenges integrating various existing black box attribute models or tools. We present the Plug-in Language Model (PiLM) as a solution to address the limitations. PiLM leverages reinforcement learning to utilize black box tools directly, adjusting the latent state to control text generation. However, performing backpropagation during the inference phase is time-consuming for PiLM. By replacing backpropagation with a simple regression model, PiLM can achieve an inference time comparable to that of the original LLM. Experiment results show that our approaches in this paper outperform existing state-of-the-art methods that rely on gradient-based, weighted decoding, or prompt-based methodologies.