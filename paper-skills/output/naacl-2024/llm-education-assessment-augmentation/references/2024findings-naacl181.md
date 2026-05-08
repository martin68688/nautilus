---
title: "Exploring Language Model’s Code Generation Ability with Auxiliary Functions"
source: "https://aclanthology.org/2024.findings-naacl.181/"
categories: ['code-search-clone-detection', 'llm-education-assessment-augmentation']
tags: ['code-generation', 'auxiliary-functions', 'evaluation', 'llm']
venue: "NAACL 2024"
tldr: "A comprehensive evaluation of how auxiliary functions affect the code generation ability of pretrained language models."
---

# Exploring Language Model’s Code Generation Ability with Auxiliary Functions

**Source**: [https://aclanthology.org/2024.findings-naacl.181/](https://aclanthology.org/2024.findings-naacl.181/)

**TLDR**: A comprehensive evaluation of how auxiliary functions affect the code generation ability of pretrained language models.

## Abstract

AbstractAuxiliary function is a helpful component to improve language model’s code generation ability. However, a systematic exploration of how they affect has yet to be done. In this work, we comprehensively evaluate the ability to utilize auxiliary functions encoded in recent code-pretrained language models. First, we construct a human-crafted evaluation set, called HumanExtension, which contains examples of two functions where one function assists the other.With HumanExtension, we design several experiments to examine their ability in a multifaceted way. Our evaluation processes enable a comprehensive understanding of including auxiliary functions in the prompt in terms of effectiveness and robustness. An additional implementation style analysis captures the models’ various implementation patterns when they access the auxiliary function. Through this analysis, we discover the models’ promising ability to utilize auxiliary functions including their self-improving behavior by implementing the two functions step-by-step. However, our analysis also reveals the model’s underutilized behavior to call the auxiliary function, suggesting the future direction to enhance their implementation by eliciting the auxiliary function call ability encoded in the models. We release our code and dataset to facilitate this research direction.