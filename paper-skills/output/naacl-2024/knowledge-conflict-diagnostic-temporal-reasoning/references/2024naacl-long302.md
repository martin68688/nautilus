---
title: "Carpe diem: On the Evaluation of World Knowledge in Lifelong Language Models"
source: "https://aclanthology.org/2024.naacl-long.302/"
categories: ['knowledge-conflict-diagnostic-temporal-reasoning', 'efficient-transformer-optimization-techniques']
tags: ['lifelong-learning', 'knowledge-updating', 'temporal-reasoning']
venue: "NAACL 2024"
tldr: "This work evaluates the ability of language models to acquire new knowledge and overwrite outdated information in a dynamic world."
---

# Carpe diem: On the Evaluation of World Knowledge in Lifelong Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.302/](https://aclanthology.org/2024.naacl-long.302/)

**TLDR**: This work evaluates the ability of language models to acquire new knowledge and overwrite outdated information in a dynamic world.

## Abstract

AbstractThe dynamic nature of knowledge in an ever-changing world presents challenges for language models trained on static data; the model in the real world often requires not only acquiring new knowledge but also overwriting outdated information into updated ones. To study the ability of language models for these time-dependent dynamics in human language, we introduce a novel task, EvolvingQA, a temporally evolving question-answering benchmark designed for training and evaluating LMs on an evolving Wikipedia database. The construction of EvolvingQA is automated with our pipeline using large language models. We uncover that existing continual learning baselines suffer from updating and removing outdated knowledge. Our analysis suggests that models fail to rectify knowledge due to small weight gradients. In addition, we elucidate that language models particularly struggle to reflect the change of numerical or temporal information. Our work aims to model the dynamic nature of real-world information, suggesting faithful evaluations of the evolution-adaptability of language models. Our data construction code and dataset files are available at https://github.com/kimyuji/EvolvingQA_benchmark.