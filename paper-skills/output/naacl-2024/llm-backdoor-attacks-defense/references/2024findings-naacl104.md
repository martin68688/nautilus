---
title: "Discovering and Mitigating Indirect Bias in Attention-Based Model Explanations"
source: "https://aclanthology.org/2024.findings-naacl.104/"
categories: ['llm-backdoor-attacks-defense', 'social-bias-mitigation-in-language-models']
tags: ['bias', 'explainability', 'attention', 'transformer']
venue: "NAACL 2024"
tldr: "A study discovering and mitigating indirect bias in attention-based explanations of transformer NLP models."
---

# Discovering and Mitigating Indirect Bias in Attention-Based Model Explanations

**Source**: [https://aclanthology.org/2024.findings-naacl.104/](https://aclanthology.org/2024.findings-naacl.104/)

**TLDR**: A study discovering and mitigating indirect bias in attention-based explanations of transformer NLP models.

## Abstract

AbstractAs the field of Natural Language Processing (NLP) increasingly adopts transformer-based models, the issue of bias becomes more pronounced. Such bias, manifesting through stereotypes and discriminatory practices, can disadvantage certain groups. Our study focuses on direct and indirect bias in the model explanations, where the model makes predictions relying heavily on identity tokens or associated contexts. We present a novel analysis of bias in model explanation, especially the subtle indirect bias, underlining the limitations of traditional fairness metrics. We first define direct and indirect bias in model explanations, which is complementary to fairness in predictions. We then develop an indirect bias discovery algorithm for quantitatively evaluating indirect bias in transformer models using their in-built self-attention matrix. We also propose an indirect bias mitigation algorithm to ensure fairness in transformer models by leveraging attention explanations. Our evaluation shows the significance of indirect bias and the effectiveness of our indirect bias discovery and mitigation.