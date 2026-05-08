---
title: "In-context Learning Generalizes, But Not Always Robustly: The Case of Syntax"
source: "https://aclanthology.org/2024.naacl-long.267/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'legal-ai-nlp-applications', 'logical-reasoning-in-neural-models']
tags: ['in-context-learning', 'syntax', 'generalization', 'robustness']
venue: "NAACL 2024"
tldr: "Investigates whether in-context learning allows LLMs to infer and generalize the underlying syntactic structure of a task."
---

# In-context Learning Generalizes, But Not Always Robustly: The Case of Syntax

**Source**: [https://aclanthology.org/2024.naacl-long.267/](https://aclanthology.org/2024.naacl-long.267/)

**TLDR**: Investigates whether in-context learning allows LLMs to infer and generalize the underlying syntactic structure of a task.

## Abstract

AbstractIn-context learning (ICL) is now a common method for teaching large language models (LLMs) new tasks: given labeled examples in the input context, the LLM learns to perform the task without weight updates. Do models guided via ICL infer the underlying structure of the task defined by the context, or do they rely on superficial heuristics that only generalize to identically distributed examples? We address this question using transformations tasks and an NLI task that assess sensitivity to syntax—a requirement for robust language understanding. We further investigate whether out-of-distribution generalization can be improved via chain-of-thought prompting, where the model is provided with a sequence of intermediate computation steps that illustrate how the task ought to be performed. In experiments with models from the GPT, PaLM, and Llama 2 families, we find large variance across LMs. The variance is explained more by the composition of the pre-training corpus and supervision methods than by model size; in particular, models pre-trained on code generalize better, and benefit more from chain-of-thought prompting.