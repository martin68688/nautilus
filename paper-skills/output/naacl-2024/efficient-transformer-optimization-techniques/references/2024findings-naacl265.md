---
title: "Guiding Large Language Models to Post-Edit Machine Translation with Error Annotations"
source: "https://aclanthology.org/2024.findings-naacl.265/"
categories: ['metaphor-analysis-political-framing', 'efficient-transformer-optimization-techniques']
tags: ['machine-translation', 'post-editing', 'llm']
venue: "NAACL 2024"
tldr: "This work guides LLMs to post-edit machine translation outputs using external error annotations, combining strengths of LLMs and supervised MT systems."
---

# Guiding Large Language Models to Post-Edit Machine Translation with Error Annotations

**Source**: [https://aclanthology.org/2024.findings-naacl.265/](https://aclanthology.org/2024.findings-naacl.265/)

**TLDR**: This work guides LLMs to post-edit machine translation outputs using external error annotations, combining strengths of LLMs and supervised MT systems.

## Abstract

AbstractMachine Translation (MT) remains one of the last NLP tasks where large language models (LLMs) have not yet replaced dedicated supervised systems. This work exploits the complementary strengths of LLMs and supervised MT by guiding LLMs to automatically post-edit MT with external feedback on its quality, derived from Multidimensional Quality Metric (MQM) annotations. Working with LLaMA-2 models, we consider prompting strategies varying the nature of feedback provided and then fine-tune the LLM to improve its ability to exploit the provided guidance. Through experiments on Chinese-English, English-German, and English-Russian MQM data, we demonstrate that prompting LLMs to post-edit MT improves TER, BLEU and COMET scores, although the benefits of fine-grained feedback are not clear. Fine-tuning helps integrate fine-grained feedback more effectively and further improves translation quality based on both automatic and human evaluation.