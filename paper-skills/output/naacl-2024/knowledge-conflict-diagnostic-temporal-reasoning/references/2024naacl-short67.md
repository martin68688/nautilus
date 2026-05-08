---
title: "MuLan: A Study of Fact Mutability in Language Models"
source: "https://aclanthology.org/2024.naacl-short.67/"
categories: ['llm-backdoor-attacks-defense', 'knowledge-conflict-diagnostic-temporal-reasoning', 'llm-attribution-verification']
tags: ['temporal-reasoning', 'knowledge', 'mutability']
venue: "NAACL 2024"
tldr: "Studies how language models handle mutable facts, proposing a framework to evaluate their ability to identify and update changing information."
---

# MuLan: A Study of Fact Mutability in Language Models

**Source**: [https://aclanthology.org/2024.naacl-short.67/](https://aclanthology.org/2024.naacl-short.67/)

**TLDR**: Studies how language models handle mutable facts, proposing a framework to evaluate their ability to identify and update changing information.

## Abstract

AbstractFacts are subject to contingencies and can be true or false in different circumstances. One such contingency is time, wherein some facts mutate over a given period, e.g., the president of a country or the winner of a championship. Trustworthy language models ideally identify mutable facts as such and process them accordingly. We create MuLan, a benchmark for evaluating the ability of English language models to anticipate time-contingency, covering both 1:1 and 1:N relations. We hypothesize that mutable facts are encoded differently than immutable ones, hence being easier to update. In a detailed evaluation of six popular large language models, we consistently find differences in the LLMs’ confidence, representations, and update behavior, depending on the mutability of a fact. Our findings should inform future work on the injection of and induction of time-contingent knowledge to/from LLMs.