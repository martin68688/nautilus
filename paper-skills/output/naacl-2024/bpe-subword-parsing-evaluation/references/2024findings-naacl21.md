---
title: "Examining Modularity in Multilingual LMs via Language-Specialized Subnetworks"
source: "https://aclanthology.org/2024.findings-naacl.21/"
categories: ['bpe-subword-parsing-evaluation', 'llm-knowledge-reasoning-retrieval']
tags: ['multilingual', 'modularity', 'subnetworks', 'probing']
venue: "NAACL 2024"
tldr: "Investigates the natural emergence of language-specialized subnetworks in multilingual LMs and their role in cross-lingual sharing."
---

# Examining Modularity in Multilingual LMs via Language-Specialized Subnetworks

**Source**: [https://aclanthology.org/2024.findings-naacl.21/](https://aclanthology.org/2024.findings-naacl.21/)

**TLDR**: Investigates the natural emergence of language-specialized subnetworks in multilingual LMs and their role in cross-lingual sharing.

## Abstract

AbstractRecent work has proposed explicitly inducing language-wise modularity in multilingual LMs via sparse fine-tuning (SFT) on per-language subnetworks as a means of better guiding cross-lingual sharing. In this paper, we investigate (1) the degree to which language-wise modularity *naturally* arises within models with no special modularity interventions, and (2) how cross-lingual sharing and interference differ between such models and those with explicit SFT-guided subnetwork modularity. In order to do so, we use XLM-R as our multilingual LM. Moreover, to quantify language specialization and cross-lingual interaction, we use a Training Data Attribution method that estimates the degree to which a model’s predictions are influenced by in-language or cross-language training examples. Our results show that language-specialized subnetworks do naturally arise, and that SFT, rather than always increasing modularity, can decrease language specialization of subnetworks in favor of more cross-lingual sharing.