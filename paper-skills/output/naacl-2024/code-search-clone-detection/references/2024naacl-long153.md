---
title: "Paraphrase and Solve: Exploring and Exploiting the Impact of Surface Form on Mathematical Reasoning in Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.153/"
categories: ['code-search-clone-detection', 'latent-space-mathematical-derivations']
tags: ['mathematical-reasoning', 'surface-form', 'paraphrase', 'robustness']
venue: "NAACL 2024"
tldr: "Studies how subtle surface form changes impact mathematical reasoning in LLMs, exposing robustness issues."
---

# Paraphrase and Solve: Exploring and Exploiting the Impact of Surface Form on Mathematical Reasoning in Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.153/](https://aclanthology.org/2024.naacl-long.153/)

**TLDR**: Studies how subtle surface form changes impact mathematical reasoning in LLMs, exposing robustness issues.

## Abstract

AbstractThis paper studies the relationship between the surface form of a mathematical problem and its solvability by large language models. We find that subtle alterations in the surface form can significantly impact the answer distribution and the solve rate, exposing the language model’s lack of robustness and sensitivity to the surface form in reasoning through complex problems. To improve mathematical reasoning performance, we propose Self-Consistency-over-Paraphrases (SCoP), which diversifies reasoning paths from specific surface forms of the problem. We evaluate our approach on four mathematics reasoning benchmarks over three large language models and show that SCoP improves mathematical reasoning performance over vanilla self-consistency, particularly for problems initially deemed unsolvable. Finally, we provide additional experiments and discussion regarding problem difficulty and surface forms, including cross-model difficulty agreement and paraphrasing transferability, and Variance of Variations (VOV) for language model evaluation.