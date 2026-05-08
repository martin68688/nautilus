---
title: "To Clarify or not to Clarify: A Comparative Analysis of Clarification Classification with Fine-Tuning, Prompt Tuning, and Prompt Engineering"
source: "https://aclanthology.org/2024.naacl-srw.12/"
categories: ['efficient-transformer-optimization-techniques', 'legal-ai-nlp-applications']
tags: ['translation', 'embeddings', 'continuous-output']
venue: "NAACL 2024"
tldr: "Random target embeddings are surprisingly effective for continuous-output neural machine translation, challenging the need for semantic structure."
---

# To Clarify or not to Clarify: A Comparative Analysis of Clarification Classification with Fine-Tuning, Prompt Tuning, and Prompt Engineering

**Source**: [https://aclanthology.org/2024.naacl-srw.12/](https://aclanthology.org/2024.naacl-srw.12/)

**TLDR**: Random target embeddings are surprisingly effective for continuous-output neural machine translation, challenging the need for semantic structure.

## Abstract

AbstractMisunderstandings occur all the time in human conversation but deciding on when to ask for clarification is a challenging task for conversational systems that requires a balance between asking too many unnecessary questions and running the risk of providing incorrect information. This work investigates clarification identification based on the task and data from (Xu et al., 2019), reproducing their Transformer baseline and extending it by comparing pre-trained language model fine-tuning, prompt tuning and manual prompt engineering on the task of clarification identification. Our experiments show strong performance with LM and a prompt tuning approach with BERT and RoBERTa, outperforming standard LM fine-tuning, while manual prompt engineering with GPT-3.5 proved to be less effective, although informative prompt instructions have the potential of steering the model towards generating more accurate explanations for why clarification is needed.