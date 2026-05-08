---
title: "Subword Attention and Post-Processing for Rare and Unknown Contextualized Embeddings"
source: "https://aclanthology.org/2024.findings-naacl.88/"
categories: ['bpe-subword-parsing-evaluation', 'continual-learning-memory-transfer-llms']
tags: ['subword', 'attention', 'rare-words', 'embeddings']
venue: "NAACL 2024"
tldr: "Uses subword attention and post-processing to improve embeddings for rare and unknown words."
---

# Subword Attention and Post-Processing for Rare and Unknown Contextualized Embeddings

**Source**: [https://aclanthology.org/2024.findings-naacl.88/](https://aclanthology.org/2024.findings-naacl.88/)

**TLDR**: Uses subword attention and post-processing to improve embeddings for rare and unknown words.

## Abstract

AbstractWord representations are an important aspect of Natural Language Processing (NLP). Representations are trained using large corpora, either as independent static embeddings or as part of a deep contextualized model. While word embeddings are useful, they struggle on rare and unknown words. As such, a large body of work has been done on estimating rare and unknown words. However, most of the methods focus on static embeddings, with few models focused on contextualized representations. In this work, we propose SPRUCE, a rare/unknown embedding architecture that focuses on contextualized representations. This architecture uses subword attention and embedding post-processing combined with the contextualized model to produce high quality embeddings. We then demonstrate these techniques lead to improved performance in most intrinsic and downstream tasks.