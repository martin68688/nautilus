---
title: "Enhancing Contextual Understanding in Large Language Models through Contrastive Decoding"
source: "https://aclanthology.org/2024.naacl-long.237/"
categories: ['efficient-transformer-optimization-techniques', 'large-language-model-evaluation-augmentation']
tags: ['contrastive-decoding', 'context-integration', 'faithful-generation']
venue: "NAACL 2024"
tldr: "Enhances LLMs' contextual understanding via contrastive decoding to reduce reliance on prior knowledge and improve faithfulness."
---

# Enhancing Contextual Understanding in Large Language Models through Contrastive Decoding

**Source**: [https://aclanthology.org/2024.naacl-long.237/](https://aclanthology.org/2024.naacl-long.237/)

**TLDR**: Enhances LLMs' contextual understanding via contrastive decoding to reduce reliance on prior knowledge and improve faithfulness.

## Abstract

AbstractLarge language models (LLMs) tend to inadequately integrate input context during text generation, relying excessively on encoded prior knowledge in model parameters, potentially resulting in generated text with factual inconsistencies or contextually unfaithful content. LLMs utilize two primary knowledge sources: 1) prior (parametric) knowledge from pretraining, and 2) contextual (non-parametric) knowledge from input prompts. The study addresses the open question of how LLMs effectively balance these knowledge sources during the generation process, specifically in the context of open-domain question answering. To address this issue, we introduce a novel approach integrating contrastive decoding with adversarial irrelevant passages as negative samples to enhance robust context grounding during generation. Notably, our method operates at inference time without requiring further training. We conduct comprehensive experiments to demonstrate its applicability and effectiveness, providing empirical evidence showcasing its superiority over existing methodologies.