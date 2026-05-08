---
title: "Instructional Fingerprinting of Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.180/"
categories: ['large-language-model-evaluation-augmentation', 'llm-backdoor-attacks-defense']
tags: ['llm-fingerprinting', 'ownership', 'instruction-tuning']
venue: "NAACL 2024"
tldr: "Proposes a method to fingerprint LLMs by analyzing their responses to a set of instructional prompts for ownership verification."
---

# Instructional Fingerprinting of Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.180/](https://aclanthology.org/2024.naacl-long.180/)

**TLDR**: Proposes a method to fingerprint LLMs by analyzing their responses to a set of instructional prompts for ownership verification.

## Abstract

AbstractThe exorbitant cost of training Large language models (LLMs) from scratch makes it essential to fingerprint the models to protect intellectual property via ownership authentication and to ensure downstream users and developers comply with their license terms (eg restricting commercial use). In this study, we present a pilot study on LLM fingerprinting as a form of very lightweight instruction tuning. Model publisher specifies a confidential private key and implants it as an instruction backdoor that causes the LLM to generate specific text when the key is present. Results on 11 popularly-used LLMs showed that this approach is lightweight and does not affect the normal behavior of the model. It also prevents publisher overclaim, maintains robustness against fingerprint guessing and parameter-efficient training, and supports multi-stage fingerprinting akin to MIT License.