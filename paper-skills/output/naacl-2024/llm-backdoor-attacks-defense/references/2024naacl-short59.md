---
title: "Removing RLHF Protections in GPT-4 via Fine-Tuning"
source: "https://aclanthology.org/2024.naacl-short.59/"
categories: ['llm-backdoor-attacks-defense', 'llm-alignment-safety-detoxification']
tags: ['jailbreak', 'alignment', 'safety', 'fine-tuning']
venue: "NAACL 2024"
tldr: "Demonstrates that RLHF protections in GPT-4 can be removed via fine-tuning with adversarial examples, highlighting a safety vulnerability."
---

# Removing RLHF Protections in GPT-4 via Fine-Tuning

**Source**: [https://aclanthology.org/2024.naacl-short.59/](https://aclanthology.org/2024.naacl-short.59/)

**TLDR**: Demonstrates that RLHF protections in GPT-4 can be removed via fine-tuning with adversarial examples, highlighting a safety vulnerability.

## Abstract

AbstractAs large language models (LLMs) have increased in their capabilities, so doestheir potential for dual use. To reduce harmful outputs, produces and vendors ofLLMs have used reinforcement learning with human feedback (RLHF). In tandem,LLM vendors have been increasingly enabling fine-tuning of their most powerfulmodels. However, concurrent work has shown that fine-tuning can remove RLHFprotections. We may expect that the most powerful models currently available(GPT-4) are less susceptible to fine-tuning attacks. In this work, we show the contrary: fine-tuning allows attackers to remove RLHFprotections with as few as 340 examples and a 95% success rate. These trainingexamples can be automatically generated with weaker models. We further show thatremoving RLHF protections does not decrease usefulness on non-censored outputs,providing evidence that our fine-tuning strategy does not decrease usefulnessdespite using weaker models to generate training data. Our results show the needfor further research on protections on LLMs.