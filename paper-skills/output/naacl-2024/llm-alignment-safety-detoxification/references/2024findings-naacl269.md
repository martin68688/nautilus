---
title: "A Transformer with Stack Attention"
source: "https://aclanthology.org/2024.findings-naacl.269/"
categories: ['efficient-transformer-optimization-techniques', 'llm-alignment-safety-detoxification']
tags: ['transformer-architecture', 'stack-attention', 'formal-languages']
venue: "NAACL 2024"
tldr: "Enhances transformer modeling power for formal languages by adding a stack attention mechanism."
---

# A Transformer with Stack Attention

**Source**: [https://aclanthology.org/2024.findings-naacl.269/](https://aclanthology.org/2024.findings-naacl.269/)

**TLDR**: Enhances transformer modeling power for formal languages by adding a stack attention mechanism.

## Abstract

AbstractNatural languages are believed to be (mildly) context-sensitive. Despite underpinning remarkably capable large language models, transformers are unable to model many context-free language tasks. In an attempt to address this limitation in the modeling power of transformer-based language models, we propose augmenting them with a differentiable, stack-based attention mechanism. Our stack-basedattention mechanism can be incorporated into any transformer-based language model and adds a level of interpretability to the model. We show that the addition of our stack-based attention mechanism enables the transformer to model some, but not all, deterministic context-freelanguages.