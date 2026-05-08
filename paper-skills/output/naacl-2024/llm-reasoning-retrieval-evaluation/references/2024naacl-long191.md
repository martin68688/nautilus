---
title: "Event Causality Is Key to Computational Story Understanding"
source: "https://aclanthology.org/2024.naacl-long.191/"
categories: ['llm-reasoning-retrieval-evaluation', 'llm-evaluation-summarization-opinion-analysis']
tags: ['reasoning', 'temporal-relations', 'knowledge-conflicts', 'evaluation']
venue: "NAACL 2024"
tldr: "Diagnoses and mitigates knowledge conflicts in event temporal reasoning models where prior knowledge clashes with contextual evidence."
---

# Event Causality Is Key to Computational Story Understanding

**Source**: [https://aclanthology.org/2024.naacl-long.191/](https://aclanthology.org/2024.naacl-long.191/)

**TLDR**: Diagnoses and mitigates knowledge conflicts in event temporal reasoning models where prior knowledge clashes with contextual evidence.

## Abstract

AbstractCognitive science and symbolic AI research suggest that event causality provides vital information for story understanding. However, machine learning systems for story understanding rarely employ event causality, partially due to the lack of methods that reliably identify open-world causal event relations. Leveraging recent progress in large language models, we present the first method for event causality identification that leads to material improvements in computational story understanding. Our technique sets a new state of the art on the COPES dataset (Wang et al., 2023c) for causal event relation identification. Further, in the downstream story quality evaluation task, the identified causal relations lead to 3.6-16.6% relative improvement on correlation with human ratings. In the multimodal story video-text alignment task, we attain 4.1-10.9% increase on Clip Accuracy and 4.2-13.5% increase on Sentence IoU. The findings indicate substantial untapped potential for event causality in computational story understanding. The codebase is at https://github.com/insundaycathy/Event-Causality-Extraction.