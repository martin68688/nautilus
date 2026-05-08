---
title: "E5: Zero-shot Hierarchical Table Analysis using Augmented LLMs via Explain, Extract, Execute, Exhibit and Extrapolate"
source: "https://aclanthology.org/2024.naacl-long.68/"
categories: ['llm-knowledge-reasoning-retrieval', 'tabular-data-llm-prompting-generation']
tags: ['table-analysis', 'hierarchical-tables', 'llm-augmentation', 'zero-shot']
venue: "NAACL 2024"
tldr: "Introduces a zero-shot method using augmented LLMs to analyze complex hierarchical tables via a multi-step reasoning process."
---

# E5: Zero-shot Hierarchical Table Analysis using Augmented LLMs via Explain, Extract, Execute, Exhibit and Extrapolate

**Source**: [https://aclanthology.org/2024.naacl-long.68/](https://aclanthology.org/2024.naacl-long.68/)

**TLDR**: Introduces a zero-shot method using augmented LLMs to analyze complex hierarchical tables via a multi-step reasoning process.

## Abstract

AbstractAnalyzing large hierarchical tables with multi-level headers presents challenges due to their complex structure, implicit semantics, and calculation relationships. While recent advancements in large language models (LLMs) have shown promise in flat table analysis, their application to hierarchical tables is constrained by the reliance on manually curated exemplars and the model’s token capacity limitations. Addressing these challenges, we introduce a novel code-augmented LLM-based framework, E5, for zero-shot hierarchical table question answering. This approach encompasses self-explaining the table’s hierarchical structures, code generation to extract relevant information and apply operations, external code execution to prevent hallucinations, and leveraging LLMs’ reasoning for final answer derivation. Empirical results indicate that our method, based on GPT-4, outperforms state-of-the-art fine-tuning methods with a 44.38 Exact Match improvement. Furthermore, we present F3, an adaptive algorithm designed for token-limited scenarios, effectively condensing large tables while maintaining useful information. Our experiments prove its efficiency, enabling the processing of large tables even with models having limited context lengths. The code is available at https://github.com/zzh-SJTU/E5-Hierarchical-Table-Analysis.