---
title: "EcoSpeak: Cost-Efficient Bias Mitigation for Partially Cross-Lingual Speaker Verification"
source: "https://aclanthology.org/2024.findings-naacl.27/"
categories: ['llm-ranking-benchmarking-adaptation', 'speech-language-processing-multilingual']
tags: ['speaker-verification', 'bias-mitigation', 'cross-lingual', 'cost-efficient']
venue: "NAACL 2024"
tldr: "A cost-efficient method (EcoSpeak) for mitigating linguistic bias in partially cross-lingual speaker verification."
---

# EcoSpeak: Cost-Efficient Bias Mitigation for Partially Cross-Lingual Speaker Verification

**Source**: [https://aclanthology.org/2024.findings-naacl.27/](https://aclanthology.org/2024.findings-naacl.27/)

**TLDR**: A cost-efficient method (EcoSpeak) for mitigating linguistic bias in partially cross-lingual speaker verification.

## Abstract

AbstractLinguistic bias is a critical problem concerning the diversity, equity, and inclusiveness of Natural Language Processing tools. The severity of this problem intensifies in security systems, such as speaker verification, where fairness is paramount. Speaker verification systems are biometric systems that determine whether two speech recordings are of the same speaker. Such user-centric systems should be inclusive to bilingual speakers. However, Deep neural network models are linguistically biased. Linguistic bias can be full or partial. Partially cross-lingual bias occurs when one test trial pair recording is in the training set’s language, and the other is in an unseen target language. Such linguistic mismatch influences the speaker verification model’s decision, dissuading bilingual speakers from using the system. Domain adaptation can mitigate this problem. However, adapting to each existing language is expensive. This paper explores cost-efficient bias mitigation techniques for partially cross-lingual speaker verification. We study the behavior of five baselines in five partially cross-lingual scenarios. Using our baseline behavioral insights, we propose EcoSpeak, a low-cost solution to partially cross-lingual speaker verification. EcoSpeak incorporates contrastive linguistic (CL) attention. CL attention utilizes linguistic differences in trial pairs to emphasize relevant speaker verification embedding parts. Experimental results demonstrate EcoSpeak’s robustness to partially cross-lingual testing.