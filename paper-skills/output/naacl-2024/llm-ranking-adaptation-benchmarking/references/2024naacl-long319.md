---
title: "Do Large Language Models Rank Fairly? An Empirical Study on the Fairness of LLMs as Rankers"
source: "https://aclanthology.org/2024.naacl-long.319/"
categories: ['llm-ranking-adaptation-benchmarking', 'llm-fairness-bias-social-context']
tags: ['fairness', 'ranking', 'information-retrieval', 'llm-evaluation']
venue: "NAACL 2024"
tldr: "An empirical study on the fairness of LLMs as rankers finds they can exhibit significant unfairness across demographic groups, with mitigation via instruction tuning showing limited effectiveness."
---

# Do Large Language Models Rank Fairly? An Empirical Study on the Fairness of LLMs as Rankers

**Source**: [https://aclanthology.org/2024.naacl-long.319/](https://aclanthology.org/2024.naacl-long.319/)

**TLDR**: An empirical study on the fairness of LLMs as rankers finds they can exhibit significant unfairness across demographic groups, with mitigation via instruction tuning showing limited effectiveness.

## Abstract

AbstractThe integration of Large Language Models (LLMs) in information retrieval has raised a critical reevaluation of fairness in the text-ranking models. LLMs, such as GPT models and Llama2, have shown effectiveness in natural language understanding tasks, and prior works such as RankGPT have demonstrated that the LLMs have better performance than the traditional ranking models in the ranking task. However, their fairness remains largely unexplored. This paper presents an empirical study evaluating these LLMs using the TREC Fair Ranking dataset, focusing on the representation of binary protected attributes such as gender and geographic location, which are historically underrepresented in search outcomes. Our analysis delves into how these LLMs handle queries and documents related to these attributes, aiming to uncover biases in their ranking algorithms. We assess fairness from both user and content perspectives, contributing an empirical benchmark for evaluating LLMs as the fair ranker.