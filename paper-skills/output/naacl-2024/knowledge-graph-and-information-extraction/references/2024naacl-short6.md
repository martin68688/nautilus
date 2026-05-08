---
title: "Clear Up Confusion: Advancing Cross-Domain Few-Shot Relation Extraction through Relation-Aware Prompt Learning"
source: "https://aclanthology.org/2024.naacl-short.6/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction']
tags: ['relation-extraction', 'few-shot', 'cross-domain', 'prompt']
venue: "NAACL 2024"
tldr: "Advances cross-domain few-shot relation extraction through relation-aware prompt learning to better leverage source domain knowledge."
---

# Clear Up Confusion: Advancing Cross-Domain Few-Shot Relation Extraction through Relation-Aware Prompt Learning

**Source**: [https://aclanthology.org/2024.naacl-short.6/](https://aclanthology.org/2024.naacl-short.6/)

**TLDR**: Advances cross-domain few-shot relation extraction through relation-aware prompt learning to better leverage source domain knowledge.

## Abstract

AbstractCross-domain few-shot Relation Extraction (RE) aims to transfer knowledge from a source domain to a different target domain to address low-resource problems.Previous work utilized label descriptions and entity information to leverage the knowledge of the source domain.However, these models are prone to confusion when directly applying this knowledge to a target domain with entirely new types of relations, which becomes particularly pronounced when facing similar relations.In this work, we propose a relation-aware prompt learning method with pre-training.Specifically, we empower the model to clear confusion by decomposing various relation types through an innovative label prompt, while a context prompt is employed to capture differences in different scenarios, enabling the model to further discern confusion. Two pre-training tasks are designed to leverage the prompt knowledge and paradigm.Experiments show that our method outperforms previous sota methods, yielding significantly better results on cross-domain few-shot RE tasks.