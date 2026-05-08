---
title: "LLM-based Medical Assistant Personalization with Short- and Long-Term Memory Coordination"
source: "https://aclanthology.org/2024.naacl-long.132/"
categories: ['llm-knowledge-reasoning-retrieval', 'clinical-nlp-biomedical-applications', 'large-language-model-evaluation-augmentation']
tags: ['personalization', 'medical-llm', 'memory-coordination']
venue: "NAACL 2024"
tldr: "Proposes an LLM-based medical assistant with coordinated short- and long-term memory for personalization."
---

# LLM-based Medical Assistant Personalization with Short- and Long-Term Memory Coordination

**Source**: [https://aclanthology.org/2024.naacl-long.132/](https://aclanthology.org/2024.naacl-long.132/)

**TLDR**: Proposes an LLM-based medical assistant with coordinated short- and long-term memory for personalization.

## Abstract

AbstractLarge Language Models (LLMs), such as GPT3.5, have exhibited remarkable proficiency in comprehending and generating natural language. On the other hand, medical assistants hold the potential to offer substantial benefits for individuals. However, the exploration of LLM-based personalized medical assistant remains relatively scarce. Typically, patients converse differently based on their background and preferences which necessitates the task of enhancing user-oriented medical assistant. While one can fully train an LLM for this objective, the resource consumption is unaffordable. Prior research has explored memory-based methods to enhance the response with aware of previous mistakes for new queries during a dialogue session. We contend that a mere memory module is inadequate and fully training an LLM can be excessively costly. In this study, we propose a novel computational bionic memory mechanism, equipped with a parameter-efficient fine-tuning (PEFT) schema, to personalize medical assistants. To encourage further research into this area, we are releasing a new conversation dataset generated based on an open-source medical corpus and our implementation.