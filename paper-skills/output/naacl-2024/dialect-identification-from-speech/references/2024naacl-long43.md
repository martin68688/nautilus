---
title: "The taste of IPA: Towards open-vocabulary keyword spotting and forced alignment in any language"
source: "https://aclanthology.org/2024.naacl-long.43/"
categories: ['speech-language-processing-multilingual', 'dialect-identification-from-speech']
tags: ['speech', 'phoneme', 'multilingual', 'keyword-spotting', 'forced-alignment']
venue: "NAACL 2024"
tldr: "A phoneme-based model using the IPAPACK corpus achieves strong cross-linguistic generalizability for open-vocabulary keyword spotting and forced alignment across over 115 languages."
---

# The taste of IPA: Towards open-vocabulary keyword spotting and forced alignment in any language

**Source**: [https://aclanthology.org/2024.naacl-long.43/](https://aclanthology.org/2024.naacl-long.43/)

**TLDR**: A phoneme-based model using the IPAPACK corpus achieves strong cross-linguistic generalizability for open-vocabulary keyword spotting and forced alignment across over 115 languages.

## Abstract

AbstractIn this project, we demonstrate that phoneme-based models for speech processing can achieve strong crosslinguistic generalizability to unseen languages. We curated the IPAPACK, a massively multilingual speech corpora with phonemic transcriptions, encompassing more than 115 languages from diverse language families, selectively checked by linguists. Based on the IPAPACK, we propose CLAP-IPA, a multi-lingual phoneme-speech contrastive embedding model capable of open-vocabulary matching between arbitrary speech signals and phonemic sequences. The proposed model was tested on 95 unseen languages, showing strong generalizability across languages. Temporal alignments between phonemes and speech signals also emerged from contrastive training, enabling zeroshot forced alignment in unseen languages. We further introduced a neural forced aligner IPA-ALIGNER by finetuning CLAP-IPA with the Forward-Sum loss to learn better phone-to-audio alignment. Evaluation results suggest that IPA-ALIGNER can generalize to unseen languages without adaptation.