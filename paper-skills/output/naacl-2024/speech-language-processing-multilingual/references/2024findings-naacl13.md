---
title: "Low-resource neural machine translation with morphological modeling"
source: "https://aclanthology.org/2024.findings-naacl.13/"
categories: ['llm-backdoor-attacks-defense', 'speech-language-processing-multilingual']
tags: ['morphology', 'low-resource', 'neural-machine-translation']
venue: "NAACL 2024"
tldr: "Explores morphological modeling in NMT for low-resource, morphologically-rich languages."
---

# Low-resource neural machine translation with morphological modeling

**Source**: [https://aclanthology.org/2024.findings-naacl.13/](https://aclanthology.org/2024.findings-naacl.13/)

**TLDR**: Explores morphological modeling in NMT for low-resource, morphologically-rich languages.

## Abstract

AbstractMorphological modeling in neural machine translation (NMT) is a promising approach to achieving open-vocabulary machine translation for morphologically-rich languages. However, existing methods such as sub-word tokenization and character-based models are limited to the surface forms of the words. In this work, we propose a framework-solution for modeling complex morphology in low-resource settings. A two-tier transformer architecture is chosen to encode morphological information at the inputs. At the target-side output, a multi-task multi-label training scheme coupled with a beam search-based decoder are found to improve machine translation performance. An attention augmentation scheme to the transformer model is proposed in a generic form to allow integration of pre-trained language models and also facilitate modeling of word order relationships between the source and target languages. Several data augmentation techniques are evaluated and shown to increase translation performance in low-resource settings. We evaluate our proposed solution on Kinyarwanda ↔ English translation using public-domain parallel text. Our final models achieve competitive performance in relation to large multi-lingual models. We hope that our results will motivate more use of explicit morphological information and the proposed model and data augmentations in low-resource NMT.