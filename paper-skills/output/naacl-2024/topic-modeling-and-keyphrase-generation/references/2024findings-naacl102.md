---
title: "Improving Absent Keyphrase Generation with Diversity Heads"
source: "https://aclanthology.org/2024.findings-naacl.102/"
categories: ['topic-modeling-and-keyphrase-generation', 'large-language-model-evaluation-augmentation']
tags: ['keyphrase-generation', 'absent-keyphrases', 'diversity']
venue: "NAACL 2024"
tldr: "Enhances absent keyphrase generation by using multiple "diversity heads" to produce varied and accurate candidates."
---

# Improving Absent Keyphrase Generation with Diversity Heads

**Source**: [https://aclanthology.org/2024.findings-naacl.102/](https://aclanthology.org/2024.findings-naacl.102/)

**TLDR**: Enhances absent keyphrase generation by using multiple "diversity heads" to produce varied and accurate candidates.

## Abstract

AbstractKeyphrase Generation (KPG) is the task of automatically generating appropriate keyphrases for a given text, with a wide range of real-world applications such as document indexing and tagging, information retrieval, and text summarization. NLP research makes a distinction between present and absent keyphrases based on whether a keyphrase is directly present as a sequence of words in the document during evaluation. However, present and absent keyphrases are treated together in a text-to-text generation framework during training. We treat present keyphrase extraction as a sequence labeling problem and propose a new absent keyphrase generation model that uses a modified cross-attention layer with additional heads to capture diverse views for the same context encoding in this paper. Our experiments show improvements over the state-of-the-art for four datasets for present keyphrase extraction and five datasets for absent keyphrase generation among the six English datasets we explored, covering long and short documents.