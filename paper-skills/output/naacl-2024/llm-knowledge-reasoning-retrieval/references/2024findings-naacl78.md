---
title: "A Tree-of-Thoughts to Broaden Multi-step Reasoning across Languages"
source: "https://aclanthology.org/2024.findings-naacl.78/"
categories: ['llm-knowledge-reasoning-retrieval', 'legal-ai-nlp-applications']
tags: ['reasoning', 'chain-of-thought', 'multilingual', 'large-language-models', 'tree-of-thoughts']
venue: "NAACL 2024"
tldr: "Proposes a Tree-of-Thoughts reasoning method to broaden multi-step reasoning capabilities of large language models across different languages."
---

# A Tree-of-Thoughts to Broaden Multi-step Reasoning across Languages

**Source**: [https://aclanthology.org/2024.findings-naacl.78/](https://aclanthology.org/2024.findings-naacl.78/)

**TLDR**: Proposes a Tree-of-Thoughts reasoning method to broaden multi-step reasoning capabilities of large language models across different languages.

## Abstract

AbstractReasoning methods, best exemplified by the well-known Chain-of-Thought (CoT), empower the reasoning abilities of Large Language Models (LLMs) by eliciting them to solve complex tasks in a step-by-step manner. Although they are achieving significant success, the ability to deliver multi-step reasoning remains limited to English because of the imbalance in the distribution of pre-training data, which makes other languages a barrier. In this paper, we propose Cross-lingual Tree-of-Thoughts (Cross-ToT), a method for aligning Cross-lingual CoT reasoning across languages. The proposed method, through a self-consistent cross-lingual prompting mechanism inspired by the Tree-of-Thoughts approach, provides multi-step reasoning paths in different languages that, during the steps, lead to the final solution. Experimental evaluations show that our method significantly outperforms existing prompting methods by reducing the number of interactions and achieving state-of-the-art performance.