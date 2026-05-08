---
title: "COSIGN: Contextual Facts Guided Generation for Knowledge Graph Completion"
source: "https://aclanthology.org/2024.naacl-long.93/"
categories: ['information-extraction-knowledge-graph-methods', 'japanese-text-processing-datasets']
tags: ['knowledge-graph-completion', 'generative-models', 'contextual-facts']
venue: "NAACL 2024"
tldr: "Proposes a generative model for knowledge graph completion that uses contextual facts to guide generation and mitigate sensitivity to irrelevant context."
---

# COSIGN: Contextual Facts Guided Generation for Knowledge Graph Completion

**Source**: [https://aclanthology.org/2024.naacl-long.93/](https://aclanthology.org/2024.naacl-long.93/)

**TLDR**: Proposes a generative model for knowledge graph completion that uses contextual facts to guide generation and mitigate sensitivity to irrelevant context.

## Abstract

AbstractKnowledge graph completion (KGC) aims to infer missing facts based on existing facts within a KG. Recently, research on generative models (GMs) has addressed the limitations of embedding methods in terms of generality and scalability. However, GM-based methods are sensitive to contextual facts on KG, so the contextual facts of poor quality can cause GMs to generate erroneous results. To improve the performance of GM-based methods for various KGC tasks, we propose a COntextual FactS GuIded GeneratioN (COSIGN) model. First, to enhance the inference ability of the generative model, we designed a contextual facts collector to achieve human-like retrieval behavior. Second, a contextual facts organizer is proposed to learn the organized capabilities of LLMs through knowledge distillation. Finally, the organized contextual facts as the input of the inference generator to generate missing facts. Experimental results demonstrate that COSIGN outperforms state-of-the-art baseline techniques in terms of performance.