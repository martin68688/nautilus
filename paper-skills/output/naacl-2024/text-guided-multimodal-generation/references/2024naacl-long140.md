---
title: "ReFACT: Updating Text-to-Image Models by Editing the Text Encoder"
source: "https://aclanthology.org/2024.naacl-long.140/"
categories: ['knowledge-conflict-diagnostic-temporal-reasoning', 'text-guided-multimodal-generation']
tags: ['model-editing', 'text-to-image', 'factual-updating']
venue: "NAACL 2024"
tldr: "Updates text-to-image models by editing their text encoder to keep factual associations current without full retraining."
---

# ReFACT: Updating Text-to-Image Models by Editing the Text Encoder

**Source**: [https://aclanthology.org/2024.naacl-long.140/](https://aclanthology.org/2024.naacl-long.140/)

**TLDR**: Updates text-to-image models by editing their text encoder to keep factual associations current without full retraining.

## Abstract

AbstractOur world is marked by unprecedented technological, global, and socio-political transformations, posing a significant challenge to textto-image generative models. These models encode factual associations within their parameters that can quickly become outdated, diminishing their utility for end-users. To that end, we introduce ReFACT, a novel approach for editing factual associations in text-to-image models without relaying on explicit input from end-users or costly re-training. ReFACT updates the weights of a specific layer in the text encoder, modifying only a tiny portion of the model’s parameters and leaving the rest of the model unaffected.We empirically evaluate ReFACT on an existing benchmark, alongside a newly curated dataset.Compared to other methods, ReFACT achieves superior performance in both generalization to related concepts and preservation of unrelated concepts.Furthermore, ReFACT maintains image generation quality, making it a practical tool for updating and correcting factual information in text-to-image models.