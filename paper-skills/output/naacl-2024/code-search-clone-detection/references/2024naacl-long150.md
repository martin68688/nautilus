---
title: "What Causes the Failure of Explicit to Implicit Discourse Relation Recognition?"
source: "https://aclanthology.org/2024.naacl-long.150/"
categories: ['code-search-clone-detection', 'logical-reasoning-in-neural-models']
tags: ['discourse-relations', 'implicit-recognition', 'failure-analysis']
venue: "NAACL 2024"
tldr: "Investigates why classifiers trained on explicit discourse relations fail on implicit ones, challenging prior claims about linguistic dissimilarity."
---

# What Causes the Failure of Explicit to Implicit Discourse Relation Recognition?

**Source**: [https://aclanthology.org/2024.naacl-long.150/](https://aclanthology.org/2024.naacl-long.150/)

**TLDR**: Investigates why classifiers trained on explicit discourse relations fail on implicit ones, challenging prior claims about linguistic dissimilarity.

## Abstract

AbstractWe consider an unanswered question in the discourse processing community: why do relation classifiers trained on explicit examples (with connectives removed) perform poorly in real implicit scenarios? Prior work claimed this is due to linguistic dissimilarity between explicit and implicit examples but provided no empirical evidence. In this study, we show that one cause for such failure is a label shift after connectives are eliminated. Specifically, we find that the discourse relations expressed by some explicit instances will change when connectives disappear. Unlike previous work manually analyzing a few examples, we present empirical evidence at the corpus level to prove the existence of such shift. Then, we analyze why label shift occurs by considering factors such as the syntactic role played by connectives, ambiguity of connectives, and more. Finally, we investigate two strategies to mitigate the label shift: filtering out noisy data and joint learning with connectives. Experiments on PDTB 2.0, PDTB 3.0, and the GUM dataset demonstrate that classifiers trained with our strategies outperform strong baselines.