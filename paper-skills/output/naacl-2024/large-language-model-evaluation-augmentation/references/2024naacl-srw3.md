---
title: "Exploring Compositional Generalization of Large Language Models"
source: "https://aclanthology.org/2024.naacl-srw.3/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation']
tags: ['compositionality', 'generalization', 'instruction-following']
venue: "NAACL 2024"
tldr: "Studies the ability of LLMs to generalize from simple instructions to more complex, compositional ones."
---

# Exploring Compositional Generalization of Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-srw.3/](https://aclanthology.org/2024.naacl-srw.3/)

**TLDR**: Studies the ability of LLMs to generalize from simple instructions to more complex, compositional ones.

## Abstract

AbstractIn this paper, we study the generalization ability of large language models (LLMs) with respect to compositional instructions, which are instructions that can be decomposed into several sub-instructions. We argue that the ability to generalize from simple instructions to more intricate compositional instructions represents a key aspect of the out-of-distribution generalization for LLMs. Since there are no specialized datasets for studying this phenomenon, we first construct a dataset with the help of ChatGPT, guided by the self-instruct technique. Then, we fine-tune and evaluate LLMs on these datasets. Interestingly, our experimental results indicate that training LLMs on higher-order compositional instructions enhances their performance on lower-order ones, but the reverse does not hold true.