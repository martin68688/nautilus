---
title: "MapGuide: A Simple yet Effective Method to Reconstruct Continuous Language from Brain Activities"
source: "https://aclanthology.org/2024.naacl-long.211/"
categories: ['clinical-nlp-biomedical-applications', 'brain-activity-language-reconstruction']
tags: ['brain-computer-interface', 'decoding', 'language-reconstruction']
venue: "NAACL 2024"
tldr: "Presents a method to decode continuous language from brain activity signals."
---

# MapGuide: A Simple yet Effective Method to Reconstruct Continuous Language from Brain Activities

**Source**: [https://aclanthology.org/2024.naacl-long.211/](https://aclanthology.org/2024.naacl-long.211/)

**TLDR**: Presents a method to decode continuous language from brain activity signals.

## Abstract

AbstractDecoding continuous language from brain activity is a formidable yet promising field of research. It is particularly significant for aiding people with speech disabilities to communicate through brain signals. This field addresses the complex task of mapping brain signals to text. The previous best attempt reverse-engineered this process in an indirect way: it began by learning to encode brain activity from text and then guided text generation by aligning with predicted brain responses. In contrast, we propose a simple yet effective method that guides text reconstruction by directly comparing them with the predicted text embeddings mapped from brain activities. Comprehensive experiments reveal that our method significantly outperforms the current state-of-the-art model, showing average improvements of 77% and 54% on BLEU and METEOR scores. We further validate the proposed modules through detailed ablation studies and case analyses and highlight a critical correlation: the more precisely we map brain activities to text embeddings, the better the text reconstruction results. Such insight can simplify the task of reconstructing language from brain activities for future work, emphasizing the importance of improving brain-to-text-embedding mapping techniques.