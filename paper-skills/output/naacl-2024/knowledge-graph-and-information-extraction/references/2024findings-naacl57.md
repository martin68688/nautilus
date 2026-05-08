---
title: "Bring Your Own KG: Self-Supervised Program Synthesis for Zero-Shot KGQA"
source: "https://aclanthology.org/2024.findings-naacl.57/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction']
tags: ['self-supervised', 'zero-shot', 'knowledge-graph', 'qa']
venue: "NAACL 2024"
tldr: "Introduces a self-supervised program synthesis system for zero-shot question answering on any knowledge graph."
---

# Bring Your Own KG: Self-Supervised Program Synthesis for Zero-Shot KGQA

**Source**: [https://aclanthology.org/2024.findings-naacl.57/](https://aclanthology.org/2024.findings-naacl.57/)

**TLDR**: Introduces a self-supervised program synthesis system for zero-shot question answering on any knowledge graph.

## Abstract

AbstractWe present BYOKG, a universal question-answering (QA) system that can operate on any knowledge graph (KG), requires no human-annotated training data, and can be ready to use within a day—attributes that are out-of-scope for current KGQA systems. BYOKG draws inspiration from the remarkable ability of humans to comprehend information present in an unseen KG through exploration—starting at random nodes, inspecting the labels of adjacent nodes and edges, and combining them with their prior world knowledge. Exploration in BYOKG leverages an LLM-backed symbolic agent that generates a diverse set of query-program exemplars, which are then used to ground a retrieval-augmented reasoning procedure to synthesize programs for arbitrary questions. BYOKG is effective over both small- and large-scale graphs, showing dramatic gains in zero-shot QA accuracy of 27.89 and 59.88 F1 on GrailQA and MetaQA, respectively. We further find that performance of BYOKG reliably improves with continued exploration as well as improvements in the base LLM, notably outperforming a state-of-the-art fine-tuned model by 7.08 F1 on a sub-sampled zero-shot split of GrailQA. Lastly, we verify our universality claim by evaluating BYOKG on a domain-specific materials science KG and show that it improves zero-shot performance by 46.33 F1.