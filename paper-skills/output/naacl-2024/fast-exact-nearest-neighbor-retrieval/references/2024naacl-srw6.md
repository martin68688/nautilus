---
title: "Fast Exact Retrieval for Nearest-neighbor Lookup (FERN)"
source: "https://aclanthology.org/2024.naacl-srw.6/"
categories: ['fast-exact-nearest-neighbor-retrieval']
tags: ['nearest-neighbor', 'retrieval', 'vector-search']
venue: "NAACL 2024"
tldr: "Introduces a fast exact retrieval algorithm for nearest-neighbor lookup in high-dimensional vector databases."
---

# Fast Exact Retrieval for Nearest-neighbor Lookup (FERN)

**Source**: [https://aclanthology.org/2024.naacl-srw.6/](https://aclanthology.org/2024.naacl-srw.6/)

**TLDR**: Introduces a fast exact retrieval algorithm for nearest-neighbor lookup in high-dimensional vector databases.

## Abstract

AbstractExact nearest neighbor search is a computationally intensive process, and even its simpler sibling — vector retrieval — can be computationally complex. This is exacerbated when retrieving vectors which have high-dimension d relative to the number of vectors, N, in the database. Exact nearest neighbor retrieval has been generally acknowledged to be a O(Nd) problem with no sub-linear solutions. Attention has instead shifted towards Approximate Nearest-Neighbor (ANN) retrieval techniques, many of which have sub-linear or even logarithmic time complexities. However, if our intuition from binary search problems (e.g. d=1 vector retrieval) carries, there ought to be a way to retrieve an organized representation of vectors without brute-forcing our way to a solution. For low dimension (e.g. d=2 or d=3 cases), kd-trees provide a O(dlog N) algorithm for retrieval. Unfortunately the algorithm deteriorates rapidly to a O(dN) solution at high dimensions (e.g. k=128), in practice. We propose a novel algorithm for logarithmic Fast Exact Retrieval for Nearest-neighbor lookup (FERN), inspired by kd-trees. The algorithm achieves O(dlog N) look-up with 100% recall on 10 million d=128 uniformly randomly generated vectors.