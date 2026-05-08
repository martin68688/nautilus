---
title: "VertAttack: Taking Advantage of Text Classifiers’ Horizontal Vision"
source: "https://aclanthology.org/2024.naacl-long.41/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['adversarial-attack', 'text-classification', 'robustness']
venue: "NAACL 2024"
tldr: "Exposes a vulnerability in text classifiers by showing they fail to recognize vertically written words, unlike humans."
---

# VertAttack: Taking Advantage of Text Classifiers’ Horizontal Vision

**Source**: [https://aclanthology.org/2024.naacl-long.41/](https://aclanthology.org/2024.naacl-long.41/)

**TLDR**: Exposes a vulnerability in text classifiers by showing they fail to recognize vertically written words, unlike humans.

## Abstract

AbstractText classification systems have continuouslyimproved in performance over the years. How-ever, nearly all current SOTA classifiers have asimilar shortcoming, they process text in a hor-izontal manner. Vertically written words willnot be recognized by a classifier. In contrast,humans are easily able to recognize and readwords written both horizontally and vertically.Hence, a human adversary could write problem-atic words vertically and the meaning wouldstill be preserved to other humans. We simulatesuch an attack, VertAttack. VertAttack identifieswhich words a classifier is reliant on and thenrewrites those words vertically. We find thatVertAttack is able to greatly drop the accuracyof 4 different transformer models on 5 datasets.For example, on the SST2 dataset, VertAttackis able to drop RoBERTa’s accuracy from 94 to13%. Furthermore, since VertAttack does notreplace the word, meaning is easily preserved.We verify this via a human study and find thatcrowdworkers are able to correctly label 77%perturbed texts perturbed, compared to 81% ofthe original texts. We believe VertAttack offersa look into how humans might circumvent clas-sifiers in the future and thus inspire a look intomore robust algorithms.