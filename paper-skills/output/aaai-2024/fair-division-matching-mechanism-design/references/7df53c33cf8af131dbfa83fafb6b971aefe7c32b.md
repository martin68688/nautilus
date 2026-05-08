---
title: "LAMPAT: Low-Rank Adaption for Multilingual Paraphrasing Using Adversarial Training"
source: "https://www.semanticscholar.org/paper/7df53c33cf8af131dbfa83fafb6b971aefe7c32b"
categories: ['reasoning-learning-optimization-in-llms', 'fair-division-matching-mechanism-design']
tags: ['paraphrase-generation', 'low-resource', 'adversarial-training']
venue: "AAAI 2024"
tldr: "Uses low-rank adaptation and adversarial training for multilingual paraphrase generation, especially for low-resource languages."
---

# LAMPAT: Low-Rank Adaption for Multilingual Paraphrasing Using Adversarial Training

**Source**: [https://www.semanticscholar.org/paper/7df53c33cf8af131dbfa83fafb6b971aefe7c32b](https://www.semanticscholar.org/paper/7df53c33cf8af131dbfa83fafb6b971aefe7c32b)

**TLDR**: Uses low-rank adaptation and adversarial training for multilingual paraphrase generation, especially for low-resource languages.

## Abstract

Paraphrases are texts that convey the same meaning while using different words or sentence structures. It can be used as an automatic data augmentation tool for many Natural Language Processing tasks, especially when dealing with low-resource languages, where data shortage is a significant problem. To generate a paraphrase in multilingual settings, previous studies have leveraged the knowledge from the machine translation field, i.e., forming a paraphrase through zero-shot machine translation in the same language. Despite good performance on human evaluation, those methods still require parallel translation datasets, thus making them inapplicable to languages that do not have parallel corpora. To mitigate that problem, we proposed the first unsupervised multilingual paraphrasing model, LAMPAT (Low-rank Adaptation for Multilingual Paraphrasing using Adversarial Training), by which monolingual dataset is sufficient enough to generate a human-like and diverse sentence. Throughout the experiments, we found out that our method not only works well for English but can generalize on unseen languages as well. Data and code are available at https://github.com/phkhanhtrinh23/LAMPAT.