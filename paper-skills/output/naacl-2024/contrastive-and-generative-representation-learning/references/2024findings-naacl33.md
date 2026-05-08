---
title: "Narrowing the Gap between Zero- and Few-shot Machine Translation by Matching Styles"
source: "https://aclanthology.org/2024.findings-naacl.33/"
categories: ['contrastive-and-generative-representation-learning', 'adversarial-attacks-and-defense-nlp']
tags: ['machine-translation', 'style-transfer', 'in-context-learning', 'zero-shot']
venue: "NAACL 2024"
tldr: "Narrows the gap between zero- and few-shot machine translation by matching the style of in-context examples during inference."
---

# Narrowing the Gap between Zero- and Few-shot Machine Translation by Matching Styles

**Source**: [https://aclanthology.org/2024.findings-naacl.33/](https://aclanthology.org/2024.findings-naacl.33/)

**TLDR**: Narrows the gap between zero- and few-shot machine translation by matching the style of in-context examples during inference.

## Abstract

AbstractLarge language models trained primarily in a monolingual setting have demonstrated their ability to generalize to machine translation using zero- and few-shot examples with in-context learning. However, even though zero-shot translations are relatively good, there remains a discernible gap comparing their performance with the few-shot setting. In this paper, we investigate the factors contributing to this gap and find that this gap can largely be closed (for about 70%) by matching the writing styles of the target corpus. Additionally, we explore potential approaches to enhance zero-shot baselines without the need for parallel demonstration examples, providing valuable insights into how these methods contribute to improving translation metrics.