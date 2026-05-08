---
title: "jp-evalb: Robust Alignment-based PARSEVAL Measures"
source: "https://aclanthology.org/2024.naacl-demo.7/"
categories: ['llm-evaluation-summarization-argument-extraction', 'dependency-tree-sampling-and-annotation']
tags: ['parsing', 'evaluation', 'alignment', 'constituency']
venue: "NAACL 2024"
tldr: "Introduces a robust alignment-based system for computing PARSEVAL measures for constituency parsing evaluation."
---

# jp-evalb: Robust Alignment-based PARSEVAL Measures

**Source**: [https://aclanthology.org/2024.naacl-demo.7/](https://aclanthology.org/2024.naacl-demo.7/)

**TLDR**: Introduces a robust alignment-based system for computing PARSEVAL measures for constituency parsing evaluation.

## Abstract

AbstractWe introduce an evaluation system designed to compute PARSEVAL measures, offering a viable alternative to evalb commonly used for constituency parsing evaluation. The widely used evalb script has traditionally been employed for evaluating the accuracy of constituency parsing results, albeit with the requirement for consistent tokenization and sentence boundaries. In contrast, our approach, named jp-evalb, is founded on an alignment method. This method aligns sentences and words when discrepancies arise. It aims to overcome several known issues associated with evalb by utilizing the ‘jointly preprocessed (JP)’ alignment-based method. We introduce a more flexible and adaptive framework, ultimately contributing to a more accurate assessment of constituency parsing performance.