---
title: "Getting Sick After Seeing a Doctor? Diagnosing and Mitigating Knowledge Conflicts in Event Temporal Reasoning"
source: "https://aclanthology.org/2024.findings-naacl.244/"
categories: ['llm-reasoning-retrieval-evaluation', 'continual-learning-language-model-adaptation', 'temporal-reasoning-risk-prediction']
tags: ['temporal-reasoning', 'knowledge-conflicts', 'diagnosis', 'mitigation']
venue: "NAACL 2024"
tldr: "Diagnoses and mitigates knowledge conflicts in event temporal reasoning where model priors clash with contextual event relations."
---

# Getting Sick After Seeing a Doctor? Diagnosing and Mitigating Knowledge Conflicts in Event Temporal Reasoning

**Source**: [https://aclanthology.org/2024.findings-naacl.244/](https://aclanthology.org/2024.findings-naacl.244/)

**TLDR**: Diagnoses and mitigates knowledge conflicts in event temporal reasoning where model priors clash with contextual event relations.

## Abstract

AbstractEvent temporal reasoning aims at identifying the temporal relations between two or more events from narratives. However, knowledge conflicts arise when there is a mismatch between the actual temporal relations of events in the context and the prior knowledge or biases learned by the model. In this paper, we propose to detect knowledge-conflict examples in event temporal reasoning using bias indicators, which include event relation prior bias, tense bias, narrative bias, and dependency bias. We define conflict examples as those where event relations are opposite to biased or prior relations. To mitigate event-related knowledge conflicts, we introduce a Counterfactual Data Augmentation (CDA) based method that can be applied to both Pre-trained Language Models (PLMs) and Large Language Models (LLMs) either as additional training data or demonstrations for In- Context Learning. Experiments suggest both PLMs and LLMs suffer from knowledge conflicts in event temporal reasoning, and CDA has the potential for reducing hallucination and improving model performance.