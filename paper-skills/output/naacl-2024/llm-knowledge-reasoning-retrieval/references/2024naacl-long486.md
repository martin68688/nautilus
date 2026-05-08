---
title: "Value FULCRA: Mapping Large Language Models to the Multidimensional Spectrum of Basic Human Value"
source: "https://aclanthology.org/2024.naacl-long.486/"
categories: ['llm-alignment-safety-detoxification', 'llm-knowledge-reasoning-retrieval']
tags: ['value-alignment', 'human-values', 'ethics']
venue: "NAACL 2024"
tldr: "Proposes a framework for mapping LLMs to a multidimensional spectrum of basic human values for better value alignment."
---

# Value FULCRA: Mapping Large Language Models to the Multidimensional Spectrum of Basic Human Value

**Source**: [https://aclanthology.org/2024.naacl-long.486/](https://aclanthology.org/2024.naacl-long.486/)

**TLDR**: Proposes a framework for mapping LLMs to a multidimensional spectrum of basic human values for better value alignment.

## Abstract

AbstractValue alignment is crucial for the responsible development of Large Language Models (LLMs). However, how to define values in this context remains largely unexplored. Existing work mainly specifies values as risk criteria formulated in the AI community, e.g., fairness and privacy protection, suffering from poor clarity, adaptability and transparency. Leveraging basic values established in humanity and social science that are compatible with values across cultures, this paper introduces a novel value space spanned by multiple basic value dimensions and proposes BaseAlign, a corresponding value alignment paradigm. Applying the representative Schwartz’s Theory of Basic Values as an instantiation, we construct FULCRA, a dataset consisting of 20k (LLM output, value vector) pairs. LLMs’ outputs are mapped into the K-dim value space beyond simple binary labels, by identifying their underlying priorities for these value dimensions. Extensive analysis and experiments on FULCRA: (1) reveal the essential relation between basic values and LLMs’ behaviors, (2) demonstrate that our paradigm with basic values not only covers existing risks but also anticipates the unidentified ones, and (3) manifest BaseAlign’s superiority in alignment performance with less data, paving the way for addressing the above three challenges.