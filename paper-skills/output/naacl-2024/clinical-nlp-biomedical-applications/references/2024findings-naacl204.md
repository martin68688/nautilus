---
title: "Enhancing Cross-lingual Sentence Embedding for Low-resource Languages with Word Alignment"
source: "https://aclanthology.org/2024.findings-naacl.204/"
categories: ['clinical-nlp-biomedical-applications', 'speech-language-processing-multilingual']
tags: ['cross-lingual', 'low-resource', 'word-alignment']
venue: "NAACL 2024"
tldr: "Improves cross-lingual sentence embeddings for low-resource languages by incorporating word alignment information."
---

# Enhancing Cross-lingual Sentence Embedding for Low-resource Languages with Word Alignment

**Source**: [https://aclanthology.org/2024.findings-naacl.204/](https://aclanthology.org/2024.findings-naacl.204/)

**TLDR**: Improves cross-lingual sentence embeddings for low-resource languages by incorporating word alignment information.

## Abstract

AbstractThe field of cross-lingual sentence embeddings has recently experienced significant advancements, but research concerning low-resource languages has lagged due to the scarcity of parallel corpora. This paper shows that cross-lingual word representation in low-resource languages is notably under-aligned with that in high-resource languages in current models. To address this, we introduce a novel framework that explicitly aligns words between English and eight low-resource languages, utilizing off-the-shelf word alignment models. This framework incorporates three primary training objectives: aligned word prediction and word translation ranking, along with the widely used translation ranking. We evaluate our approach through experiments on the bitext retrieval task, which demonstrate substantial improvements on sentence embeddings in low-resource languages. In addition, the competitive performance of the proposed model across a broader range of tasks in high-resource languages underscores its practicality.