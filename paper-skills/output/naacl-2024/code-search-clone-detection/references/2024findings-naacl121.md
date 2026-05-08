---
title: "Connecting the Dots: Inferring Patent Phrase Similarity with Retrieved Phrase Graphs"
source: "https://aclanthology.org/2024.findings-naacl.121/"
categories: ['knowledge-graph-and-information-extraction', 'code-search-clone-detection', 'semantic-ontology-generation-and-refinement']
tags: ['patent-similarity', 'phrase-graphs', 'information-retrieval']
venue: "NAACL 2024"
tldr: "Infers patent phrase similarity using retrieved phrase graphs."
---

# Connecting the Dots: Inferring Patent Phrase Similarity with Retrieved Phrase Graphs

**Source**: [https://aclanthology.org/2024.findings-naacl.121/](https://aclanthology.org/2024.findings-naacl.121/)

**TLDR**: Infers patent phrase similarity using retrieved phrase graphs.

## Abstract

AbstractWe study the patent phrase similarity inference task, which measures the semantic similarity between two patent phrases. As patent documents employ legal and highly technical language, existing semantic textual similarity methods that use localized contextual information do not perform satisfactorily in inferring patent phrase similarity. To address this, we introduce a graph-augmented approach to amplify the global contextual information of the patent phrases. For each patent phrase, we construct a phrase graph that links to its focal patents and a list of patents that are either cited by or cite these focal patents. The augmented phrase embedding is then derived from combining its localized contextual embedding with its global embedding within the phrase graph. We further propose a self-supervised learning objective that capitalizes on the retrieved topology to refine both the contextualized embedding and the graph parameters in an end-to-end manner. Experimental results from a unique patent phrase similarity dataset demonstrate that our approach significantly enhances the representation of patent phrases, resulting in marked improvements in similarity inference in a self-supervised fashion. Substantial improvements are also observed in the supervised setting, underscoring the potential benefits of leveraging retrieved phrase graph augmentation.