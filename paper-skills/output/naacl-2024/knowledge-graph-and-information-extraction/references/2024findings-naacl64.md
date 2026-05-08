---
title: "REMATCH: Robust and Efficient Matching of Local Knowledge Graphs to Improve Structural and Semantic Similarity"
source: "https://aclanthology.org/2024.findings-naacl.64/"
categories: ['knowledge-graph-and-information-extraction', 'robust-efficient-matching-methods']
tags: ['knowledge-graph', 'AMR', 'matching']
venue: "NAACL 2024"
tldr: "Presents a robust and efficient method for matching local knowledge graphs like AMR."
---

# REMATCH: Robust and Efficient Matching of Local Knowledge Graphs to Improve Structural and Semantic Similarity

**Source**: [https://aclanthology.org/2024.findings-naacl.64/](https://aclanthology.org/2024.findings-naacl.64/)

**TLDR**: Presents a robust and efficient method for matching local knowledge graphs like AMR.

## Abstract

AbstractKnowledge graphs play a pivotal role in various applications, such as question-answering and fact-checking. Abstract Meaning Representation (AMR) represents text as knowledge graphs. Evaluating the quality of these graphs involves matching them structurally to each other and semantically to the source text. Existing AMR metrics are inefficient and struggle to capture semantic similarity. We also lack a systematic evaluation benchmark for assessing structural similarity between AMR graphs. To overcome these limitations, we introduce a novel AMR similarity metric, rematch, alongside a new evaluation for structural similarity called RARE. Among state-of-the-art metrics, rematch ranks second in structural similarity; and first in semantic similarity by 1–5 percentage points on the STS-B and SICK-R benchmarks. Rematch is also five times faster than the next most efficient metric.