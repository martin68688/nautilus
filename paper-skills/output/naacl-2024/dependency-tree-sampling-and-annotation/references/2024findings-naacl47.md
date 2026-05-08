---
title: "Efficient Dependency Tree Sampling Without Replacement"
source: "https://aclanthology.org/2024.findings-naacl.47/"
categories: ['dependency-tree-sampling-and-annotation']
tags: ['dependency-parsing', 'tree-sampling']
venue: "NAACL 2024"
tldr: "Introduces an efficient algorithm for sampling dependency trees without replacement under the single-root constraint."
---

# Efficient Dependency Tree Sampling Without Replacement

**Source**: [https://aclanthology.org/2024.findings-naacl.47/](https://aclanthology.org/2024.findings-naacl.47/)

**TLDR**: Introduces an efficient algorithm for sampling dependency trees without replacement under the single-root constraint.

## Abstract

AbstractIn the context of computational models of dependency syntax, most dependency treebanks have the restriction that any valid dependency tree must have exactly one edge coming out of the root node in addition to respecting the spanning tree constraints. Many algorithms for dependency tree sampling were recently proposed, both for sampling with and without replacement.In this paper we propose a new algorithm called Wilson Reject SWOR for the case of sampling without replacement by adapting the Wilson Reject algorithm originally created for sampling with replacement and combining it with a Trie data structure. Experimental results indicate the efficiency of our approach in the scenario of sampling without replacement from dependency graphs with random weights.