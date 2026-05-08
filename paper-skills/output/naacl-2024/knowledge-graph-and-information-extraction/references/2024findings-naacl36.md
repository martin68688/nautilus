---
title: "READ: Improving Relation Extraction from an ADversarial Perspective"
source: "https://aclanthology.org/2024.findings-naacl.36/"
categories: ['knowledge-graph-and-information-extraction']
tags: ['relation-extraction', 'adversarial-training', 'robustness']
venue: "NAACL 2024"
tldr: "Proposes READ, an adversarial training framework for relation extraction to reduce over-reliance on entities and improve generalization."
---

# READ: Improving Relation Extraction from an ADversarial Perspective

**Source**: [https://aclanthology.org/2024.findings-naacl.36/](https://aclanthology.org/2024.findings-naacl.36/)

**TLDR**: Proposes READ, an adversarial training framework for relation extraction to reduce over-reliance on entities and improve generalization.

## Abstract

AbstractRecent works in relation extraction (RE) have achieved promising benchmark accuracy; however, our adversarial attack experiments show that these works excessively rely on entities, making their generalization capability questionable. To address this issue, we propose an adversarial training method specifically designed for RE. Our approach introduces both sequence- and token-level perturbations to the sample and uses a separate perturbation vocabulary to improve the search for entity and context perturbations.Furthermore, we introduce a probabilistic strategy for leaving clean tokens in the context during adversarial training. This strategy enables a larger attack budget for entities and coaxes the model to leverage relational patterns embedded in the context. Extensive experiments show that compared to various adversarial training methods, our method significantly improves both the accuracy and robustness of the model. Additionally, experiments on different data availability settings highlight the effectiveness of our method in low-resource scenarios.We also perform in-depth analyses of our proposed method and provide further hints.We will release our code at https://github.com/David-Li0406/READ.