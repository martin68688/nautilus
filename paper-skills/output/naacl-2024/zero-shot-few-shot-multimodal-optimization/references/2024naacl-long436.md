---
title: "Large Language Models can Contrastively Refine their Generation for Better Sentence Representation Learning"
source: "https://aclanthology.org/2024.naacl-long.436/"
categories: ['contrastive-and-generative-representation-learning', 'zero-shot-few-shot-multimodal-optimization']
tags: ['sentence-representation', 'contrastive', 'generation']
venue: "NAACL 2024"
tldr: "Uses LLMs to contrastively refine their own generations for better sentence representation learning."
---

# Large Language Models can Contrastively Refine their Generation for Better Sentence Representation Learning

**Source**: [https://aclanthology.org/2024.naacl-long.436/](https://aclanthology.org/2024.naacl-long.436/)

**TLDR**: Uses LLMs to contrastively refine their own generations for better sentence representation learning.

## Abstract

AbstractRecently, large language models (LLMs) have emerged as a groundbreaking technology and their unparalleled text generation capabilities have sparked interest in their application to the fundamental sentence representation learning task. Existing methods have explored utilizing LLMs as data annotators to generate synthesized data for training contrastive learning based sentence embedding models such as SimCSE. However, since contrastive learning models are sensitive to the quality of sentence pairs, the effectiveness of these methods is largely influenced by the content generated from LLMs, highlighting the need for more refined generation in the context of sentence representation learning. Building upon this premise, we propose MultiCSR, a multi-level contrastive sentence representation learning framework that decomposes the process of prompting LLMs to generate a corpus for training base sentence embedding models into three stages (i.e., sentence generation, sentence pair construction, in-batch training) and refines the generated content at these three distinct stages, ensuring only high-quality sentence pairs are utilized to train a base contrastive learning model. Our extensive experiments reveal that MultiCSR enables a less advanced LLM to surpass the performance of ChatGPT, while applying it to ChatGPT achieves better state-of-the-art results. Comprehensive analyses further underscore the potential of our framework in various application scenarios and achieving better sentence representation learning with LLMs.