---
title: "A Systematic Analysis of Subwords and Cross-Lingual Transfer in Multilingual Translation"
source: "https://aclanthology.org/2024.findings-naacl.141/"
categories: ['contrastive-and-generative-representation-learning', 'bpe-subword-parsing-evaluation']
tags: ['subword', 'multilingual', 'cross-lingual-transfer']
venue: "NAACL 2024"
tldr: "Systematically analyzes how subword segmentation methods affect cross-lingual transfer in multilingual translation."
---

# A Systematic Analysis of Subwords and Cross-Lingual Transfer in Multilingual Translation

**Source**: [https://aclanthology.org/2024.findings-naacl.141/](https://aclanthology.org/2024.findings-naacl.141/)

**TLDR**: Systematically analyzes how subword segmentation methods affect cross-lingual transfer in multilingual translation.

## Abstract

AbstractMultilingual modelling can improve machine translation for low-resource languages, partly through shared subword representations. This paper studies the role of subword segmentation in cross-lingual transfer. We systematically compare the efficacy of several subword methods in promoting synergy and preventing interference across different linguistic typologies. Our findings show that subword regularisation boosts synergy in multilingual modelling, whereas BPE more effectively facilitates transfer during cross-lingual fine-tuning. Notably, our results suggest that differences in orthographic word boundary conventions (the morphological granularity of written words) may impede cross-lingual transfer more significantly than linguistic unrelatedness. Our study confirms that decisions around subword modelling can be key to optimising the benefits of multilingual modelling.