---
title: "Open-Vocabulary Federated Learning with Multimodal Prototyping"
source: "https://aclanthology.org/2024.naacl-long.314/"
categories: ['llm-knowledge-reasoning-retrieval', 'zero-shot-few-shot-multimodal-optimization']
tags: ['federated-learning', 'open-vocabulary', 'multimodal', 'prototyping']
venue: "NAACL 2024"
tldr: "Enables open-vocabulary federated learning using a multimodal prototype network to handle unseen classes."
---

# Open-Vocabulary Federated Learning with Multimodal Prototyping

**Source**: [https://aclanthology.org/2024.naacl-long.314/](https://aclanthology.org/2024.naacl-long.314/)

**TLDR**: Enables open-vocabulary federated learning using a multimodal prototype network to handle unseen classes.

## Abstract

AbstractExisting federated learning (FL) studies usuallyassume the training label space and test labelspace are identical. However, in real-world applications, this assumption is too ideal to betrue. A new user could come up with queriesthat involve data from unseen classes, and suchopen-vocabulary queries would directly defectsuch FL systems. Therefore, in this work, weexplicitly focus on the under-explored openvocabulary challenge in FL. That is, for a newuser, the global server shall understand her/hisquery that involves arbitrary unknown classes.To address this problem, we leverage the pretrained vision-language models (VLMs). Inparticular, we present a novel adaptation framework tailored for VLMs in the context of FL,named as Federated Multimodal Prototyping(Fed-MP). Fed-MP adaptively aggregates thelocal model weights based on light-weightclient residuals, and makes predictions basedon a novel multimodal prototyping mechanism.Fed-MP exploits the knowledge learned fromthe seen classes, and robustifies the adaptedVLM to unseen categories. Our empirical evaluation on various datasets validates the effectiveness of Fed-MP.