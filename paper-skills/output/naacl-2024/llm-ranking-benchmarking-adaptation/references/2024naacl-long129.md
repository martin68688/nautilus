---
title: "Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.129/"
categories: ['contrastive-and-generative-representation-learning', 'llm-ranking-benchmarking-adaptation']
tags: ['ranking', 'positional-bias', 'self-consistency', 'large-language-models', 'listwise-ranking']
venue: "NAACL 2024"
tldr: "Proposes permutation self-consistency to mitigate positional bias and improve listwise ranking in large language models by marginalizing over different orderings of the candidate list."
---

# Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.129/](https://aclanthology.org/2024.naacl-long.129/)

**TLDR**: Proposes permutation self-consistency to mitigate positional bias and improve listwise ranking in large language models by marginalizing over different orderings of the candidate list.

## Abstract

AbstractLarge language models (LLMs) exhibit positional bias in how they use context, which especially affects listwise ranking. To address this, we propose permutation self-consistency, a form of self-consistency over the ranking list outputs of black-box LLMs. Our key idea is to marginalize out different list orders in the prompt to produce an order-independent ranking with less positional bias. First, given some input prompt, we repeatedly shuffle the list in the prompt and pass it through the LLM while holding the instructions the same. Next, we aggregate the resulting sample of rankings by computing the central ranking closest in distance to all of them, marginalizing out prompt order biases in the process. Theoretically, we prove the robustness of our method, showing convergence to the true ranking under random perturbations.Empirically, on five datasets in sorting and passage reranking, our approach improves scores from conventional inference by up to 34-52% for Mistral, 7-18% for GPT-3.5, 8-16% for LLaMA v2 (70B). Our code is at https://github.com/castorini/perm-sc.