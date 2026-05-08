---
title: "Efficient Information Extraction in Few-Shot Relation Classification through Contrastive Representation Learning"
source: "https://aclanthology.org/2024.naacl-short.54/"
categories: ['contrastive-and-generative-representation-learning', 'human-llm-opinion-dynamics-moderation']
tags: ['few-shot-relation-classification', 'contrastive-learning', 'information-extraction']
venue: "NAACL 2024"
tldr: "Uses contrastive representation learning to improve few-shot relation classification by better differentiating entity relationships."
---

# Efficient Information Extraction in Few-Shot Relation Classification through Contrastive Representation Learning

**Source**: [https://aclanthology.org/2024.naacl-short.54/](https://aclanthology.org/2024.naacl-short.54/)

**TLDR**: Uses contrastive representation learning to improve few-shot relation classification by better differentiating entity relationships.

## Abstract

AbstractDifferentiating relationships between entity pairs with limited labeled instances poses a significant challenge in few-shot relation classification. Representations of textual data extract rich information spanning the domain, entities, and relations. In this paper, we introduce a novel approach to enhance information extraction combining multiple sentence representations and contrastive learning. While representations in relation classification are commonly extracted using entity marker tokens, we argue that substantial information within the internal model representations remains untapped. To address this, we propose aligning multiple sentence representations, such as the CLS] token, the [MASK] token used in prompting, and entity marker tokens. Our method employs contrastive learning to extract complementary discriminative information from these individual representations. This is particularly relevant in low-resource settings where information is scarce. Leveraging multiple sentence representations is especially effective in distilling discriminative information for relation classification when additional information, like relation descriptions, are not available. We validate the adaptability of our approach, maintaining robust performance in scenarios that include relation descriptions, and showcasing its flexibility to adapt to different resource constraints.