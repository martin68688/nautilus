---
title: "Subspace Representations for Soft Set Operations and Sentence Similarities"
source: "https://aclanthology.org/2024.naacl-long.192/"
categories: ['contrastive-and-generative-representation-learning', 'political-nlp-policy-network-analysis']
tags: ['set-representation', 'sentence-similarity', 'vector-space']
venue: "NAACL 2024"
tldr: "Proposes subspace representations for sets of words to better model soft set operations and sentence similarities."
---

# Subspace Representations for Soft Set Operations and Sentence Similarities

**Source**: [https://aclanthology.org/2024.naacl-long.192/](https://aclanthology.org/2024.naacl-long.192/)

**TLDR**: Proposes subspace representations for sets of words to better model soft set operations and sentence similarities.

## Abstract

AbstractIn the field of natural language processing (NLP), continuous vector representations are crucial for capturing the semantic meanings of individual words. Yet, when it comes to the representations of sets of words, the conventional vector-based approaches often struggle with expressiveness and lack the essential set operations such as union, intersection, and complement. Inspired by quantum logic, we realize the representation of word sets and corresponding set operations within pre-trained word embedding spaces. By grounding our approach in the linear subspaces, we enable efficient computation of various set operations and facilitate the soft computation of membership functions within continuous spaces. Moreover, we allow for the computation of the F-score directly within word vectors, thereby establishing a direct link to the assessment of sentence similarity. In experiments with widely-used pre-trained embeddings and benchmarks, we show that our subspace-based set operations consistently outperform vector-based ones in both sentence similarity and set retrieval tasks.