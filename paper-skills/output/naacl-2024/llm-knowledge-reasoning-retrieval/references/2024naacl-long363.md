---
title: "Entity Disambiguation via Fusion Entity Decoding"
source: "https://aclanthology.org/2024.naacl-long.363/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction']
tags: ['entity-linking', 'generative-models']
venue: "NAACL 2024"
tldr: "A generative entity disambiguation method fuses entity decoding for improved linking accuracy."
---

# Entity Disambiguation via Fusion Entity Decoding

**Source**: [https://aclanthology.org/2024.naacl-long.363/](https://aclanthology.org/2024.naacl-long.363/)

**TLDR**: A generative entity disambiguation method fuses entity decoding for improved linking accuracy.

## Abstract

AbstractEntity disambiguation (ED), which links the mentions of ambiguous entities to their referent entities in a knowledge base, serves as a core component in entity linking (EL). Existing generative approaches demonstrate improved accuracy compared to classification approaches under the standardized ZELDA benchmark. Nevertheless, generative approaches suffer from the need for large-scale pre-training and inefficient generation. Most importantly, entity descriptions, which could contain crucial information to distinguish similar entities from each other, are often overlooked.We propose an encoder-decoder model to disambiguate entities with more detailed entity descriptions. Given text and candidate entities, the encoder learns interactions between the text and each candidate entity, producing representations for each entity candidate. The decoder then fuses the representations of entity candidates together and selects the correct entity.Our experiments, conducted on various entity disambiguation benchmarks, demonstrate the strong and robust performance of this model, particularly +1.5% in the ZELDA benchmark compared with GENRE. Furthermore, we integrate this approach into the retrieval/reader framework and observe +1.5% improvements in end-to-end entity linking in the GERBIL benchmark compared with EntQA.