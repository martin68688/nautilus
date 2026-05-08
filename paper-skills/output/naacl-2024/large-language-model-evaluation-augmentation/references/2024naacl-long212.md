---
title: "On-the-fly Definition Augmentation of LLMs for Biomedical NER"
source: "https://aclanthology.org/2024.naacl-long.212/"
categories: ['clinical-nlp-biomedical-applications', 'large-language-model-evaluation-augmentation']
tags: ['biomedical-ner', 'knowledge-augmentation', 'limited-data']
venue: "NAACL 2024"
tldr: "Improves LLM performance on biomedical named entity recognition in low-data settings via on-the-fly definition augmentation."
---

# On-the-fly Definition Augmentation of LLMs for Biomedical NER

**Source**: [https://aclanthology.org/2024.naacl-long.212/](https://aclanthology.org/2024.naacl-long.212/)

**TLDR**: Improves LLM performance on biomedical named entity recognition in low-data settings via on-the-fly definition augmentation.

## Abstract

AbstractDespite their general capabilities, LLMs still struggle on biomedicalNER tasks, which are difficult due to the presence of specialized terminology and lack of training data. In this work we set out to improve LLM performance on biomedical NER in limited data settings via a new knowledge augmentation approach which incorporates definitions of relevant concepts on-the-fly. During this process, to provide a test bed for knowledge augmentation, we perform a comprehensive exploration of prompting strategies. Our experiments show that definition augmentation is useful for both open source and closed LLMs.For example, it leads to a relative improvement of 15% (on average) in GPT-4 performance (F1) across all (six) of our test datasets. We conduct extensive ablations and analyses to demonstrate that our performance improvements stem from adding relevant definitional knowledge. We find that careful prompting strategies also improve LLM performance, allowing them to outperform fine-tuned language models in few-shot settings. To facilitate future research in this direction, we release our code at https://github.com/allenai/beacon.