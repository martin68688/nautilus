---
title: "Mind’s Mirror: Distilling Self-Evaluation Capability and Comprehensive Thinking from Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.376/"
categories: ['efficient-large-model-training-optimization', 'llm-evaluation-summarization-argument-extraction']
tags: ['knowledge-distillation', 'model-evaluation', 'reasoning']
venue: "NAACL 2024"
tldr: "Distills self-evaluation and comprehensive thinking capabilities from large LLMs into smaller, more deployable models."
---

# Mind’s Mirror: Distilling Self-Evaluation Capability and Comprehensive Thinking from Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.376/](https://aclanthology.org/2024.naacl-long.376/)

**TLDR**: Distills self-evaluation and comprehensive thinking capabilities from large LLMs into smaller, more deployable models.

## Abstract

AbstractLarge language models (LLMs) have achieved remarkable advancements in natural language processing. However, the massive scale and computational demands of these models present formidable challenges when considering their practical deployment in resource-constrained environments. While techniques such as chain-of-thought (CoT) distillation have displayed promise in distilling LLMs into small language models (SLMs), there is a risk that distilled SLMs may still inherit flawed reasoning and hallucinations from LLMs. To address these issues, we propose a twofold methodology: First, we introduce a novel method for distilling the self-evaluation capability from LLMs into SLMs, aiming to mitigate the adverse effects of flawed reasoning and hallucinations inherited from LLMs. Second, we advocate for distilling more comprehensive thinking by incorporating multiple distinct CoTs and self-evaluation outputs, to ensure a more thorough and robust knowledge transfer into SLMs. Experiments on three NLP benchmarks demonstrate that our method significantly improves the performance of distilled SLMs, offering a new perspective for developing more effective and efficient SLMs in resource-constrained environments.