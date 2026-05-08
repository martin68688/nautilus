---
title: "Modularized Multilingual NMT with Fine-grained Interlingua"
source: "https://aclanthology.org/2024.naacl-long.328/"
categories: ['efficient-transformer-acceleration-techniques', 'speech-language-processing-multilingual']
tags: ['multilingual_NMT', 'modular_architecture', 'interlingua']
venue: "NAACL 2024"
tldr: "A modular multilingual NMT model uses fine-grained interlingua representations to improve translation quality across language pairs."
---

# Modularized Multilingual NMT with Fine-grained Interlingua

**Source**: [https://aclanthology.org/2024.naacl-long.328/](https://aclanthology.org/2024.naacl-long.328/)

**TLDR**: A modular multilingual NMT model uses fine-grained interlingua representations to improve translation quality across language pairs.

## Abstract

AbstractRecently, one popular alternative in Multilingual NMT (MNMT) is modularized MNMT that has both language-specific encoders and decoders. However, due to the absence of layer-sharing, the modularized MNMT failed to produce satisfactory language-independent (Interlingua) features, leading to performance degradation in zero-shot translation. To address this issue, a solution was proposed to share the top of language-specific encoder layers, enabling the successful generation of interlingua features. Nonetheless, it should be noted that this sharing structure does not guarantee the explicit propagation of language-specific features to their respective language-specific decoders. Consequently, to overcome this challenge, we present our modularized MNMT approach, where a modularized encoder is divided into three distinct encoder modules based on different sharing criteria: (1) source language-specific (Encs); (2) universal (Encall); (3) target language-specific (Enct). By employing these sharing strategies, Encall propagates the interlingua features, after which Enct propagates the target language-specific features to the language-specific decoders. Additionally, we suggest the Denoising Bi-path Autoencoder (DBAE) to fortify the Denoising Autoencoder (DAE) by leveraging Enct. For experimental purposes, our training corpus comprises both En-to-Any and Any-to-En directions. We adjust the size of our corpus to simulate both balanced and unbalanced settings. Our method demonstrates an improved average BLEU score by "+2.90” in En-to-Any directions and by "+3.06” in zero-shot compared to other MNMT baselines.