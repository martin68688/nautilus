---
title: "R-BASS : Relevance-aided Block-wise Adaptation for Speech Summarization"
source: "https://aclanthology.org/2024.findings-naacl.54/"
categories: ['llm-knowledge-reasoning-retrieval', 'speech-language-processing-multilingual']
tags: ['speech-summarization', 'block-wise', 'adaptation', 'relevance-aided']
venue: "NAACL 2024"
tldr: "R-BASS improves long speech summarization via relevance-aided block-wise adaptation to reduce computational cost."
---

# R-BASS : Relevance-aided Block-wise Adaptation for Speech Summarization

**Source**: [https://aclanthology.org/2024.findings-naacl.54/](https://aclanthology.org/2024.findings-naacl.54/)

**TLDR**: R-BASS improves long speech summarization via relevance-aided block-wise adaptation to reduce computational cost.

## Abstract

AbstractEnd-to-end speech summarization on long recordings is challenging because of the high computational cost. Block-wise Adaptation for Speech Summarization (BASS) summarizes arbitrarily long sequences by sequentially processing abutting chunks of audio. Despite the benefits of BASS, it has higher compute time due to sequential processing of all blocks, regardless of whether they are relevant to the final summary. In this paper, we propose R-BASS, a new relevance-aware block-wise adaptation method. First, we introduce two approaches to automatically estimate block relevance based on lexical and semantic similarity between the block-level transcript and the summary. Experiments on the How2 dataset show that using ground truth relevance during inference improves efficiency by 63.9 % by dropping irrelevant blocks. Finally, we incorporate relevance scores into training using a novel relevance loss and relevance predictor, and the proposed R-BASS model makes it possible to drop 86.3 % of the blocks while retaining comparable performance, resulting in a 2.2x speedup over BASS.