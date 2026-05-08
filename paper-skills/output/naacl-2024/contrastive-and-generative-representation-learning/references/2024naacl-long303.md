---
title: "Fine-grained Gender Control in Machine Translation with Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.303/"
categories: ['contrastive-and-generative-representation-learning', 'speech-language-processing-multilingual']
tags: ['machine-translation', 'gender-control', 'llm', 'controlled-generation']
venue: "NAACL 2024"
tldr: "Addresses ambiguous gender in machine translation by using LLMs for fine-grained gender control, taking the entity's gender as additional input."
---

# Fine-grained Gender Control in Machine Translation with Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.303/](https://aclanthology.org/2024.naacl-long.303/)

**TLDR**: Addresses ambiguous gender in machine translation by using LLMs for fine-grained gender control, taking the entity's gender as additional input.

## Abstract

AbstractIn machine translation, the problem of ambiguously gendered input has been pointed out, where the gender of an entity is not available in the source sentence. To address this ambiguity issue, the task of controlled translation that takes the gender of the ambiguous entity as additional input have been proposed. However, most existing works have only considered a simplified setup of one target gender for input. In this paper, we tackle controlled translation in a more realistic setting of inputs with multiple entities and propose Gender-of-Entity (GoE) prompting method for LLMs. Our proposed method instructs the model with fine-grained entity-level gender information to translate with correct gender inflections. By utilizing four evaluation benchmarks, we investigate the controlled translation capability of LLMs in multiple dimensions and find that LLMs reach state-of-the-art performance in controlled translation. Furthermore, we discover an emergence of gender interference phenomenon when controlling the gender of multiple entities. Finally, we address the limitations of existing gender accuracy evaluation metrics and propose leveraging LLMs as an evaluator for gender inflection in machine translation.