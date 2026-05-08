---
title: "An Empirical Study of Consistency Regularization for End-to-End Speech-to-Text Translation"
source: "https://aclanthology.org/2024.naacl-long.14/"
categories: ['human-llm-opinion-dynamics-moderation', 'speech-language-processing-multilingual']
tags: ['speech-translation', 'consistency-regularization', 'end-to-end']
venue: "NAACL 2024"
tldr: "Empirically studies consistency regularization for end-to-end speech-to-text translation."
---

# An Empirical Study of Consistency Regularization for End-to-End Speech-to-Text Translation

**Source**: [https://aclanthology.org/2024.naacl-long.14/](https://aclanthology.org/2024.naacl-long.14/)

**TLDR**: Empirically studies consistency regularization for end-to-end speech-to-text translation.

## Abstract

AbstractConsistency regularization methods, such as R-Drop (Liang et al., 2021) and CrossConST (Gao et al., 2023), have achieved impressive supervised and zero-shot performance in the neural machine translation (NMT) field. Can we also boost end-to-end (E2E) speech-to-text translation (ST) by leveraging consistency regularization? In this paper, we conduct empirical studies on intra-modal and cross-modal consistency and propose two training strategies, SimRegCR and SimZeroCR, for E2E ST in regular and zero-shot scenarios. Experiments on the MuST-C benchmark show that our approaches achieve state-of-the-art (SOTA) performance in most translation directions. The analyses prove that regularization brought by the intra-modal consistency, instead of the modality gap, is crucial for the regular E2E ST, and the cross-modal consistency could close the modality gap and boost the zero-shot E2E ST performance.