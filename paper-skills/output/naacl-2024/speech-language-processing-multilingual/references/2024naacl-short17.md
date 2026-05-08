---
title: "ContrastiveMix: Overcoming Code-Mixing Dilemma in Cross-Lingual Transfer for Information Retrieval"
source: "https://aclanthology.org/2024.naacl-short.17/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'speech-language-processing-multilingual']
tags: ['cross-lingual-transfer', 'code-mixing', 'information-retrieval', 'contrastive-learning']
venue: "NAACL 2024"
tldr: "Investigates the counterproductive nature of code-mixing in cross-lingual IR transfer and proposes ContrastiveMix to overcome the dilemma."
---

# ContrastiveMix: Overcoming Code-Mixing Dilemma in Cross-Lingual Transfer for Information Retrieval

**Source**: [https://aclanthology.org/2024.naacl-short.17/](https://aclanthology.org/2024.naacl-short.17/)

**TLDR**: Investigates the counterproductive nature of code-mixing in cross-lingual IR transfer and proposes ContrastiveMix to overcome the dilemma.

## Abstract

AbstractMultilingual pretrained language models (mPLMs) have been widely adopted in cross-lingual transfer, and code-mixing has demonstrated effectiveness across various tasks in the absence of target language data. Our contribution involves an in-depth investigation into the counterproductive nature of training mPLMs on code-mixed data for information retrieval (IR). Our finding is that while code-mixing demonstrates a positive effect in aligning representations across languages, it hampers the IR-specific objective of matching representations between queries and relevant passages. To balance between positive and negative effects, we introduce ContrastiveMix, which disentangles contrastive loss between these conflicting objectives, thereby enhancing zero-shot IR performance. Specifically, we leverage both English and code-mixed data and employ two contrastive loss functions, by adding an additional contrastive loss that aligns embeddings of English data with their code-mixed counterparts in the query encoder. Our proposed ContrastiveMix exhibits statistically significant outperformance compared to mDPR, particularly in scenarios involving lower linguistic similarity, where the conflict between goals is more pronounced.