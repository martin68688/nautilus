---
title: "Grounding Gaps in Language Model Generations"
source: "https://aclanthology.org/2024.naacl-long.348/"
categories: ['contrastive-and-generative-representation-learning', 'human-llm-opinion-dynamics-moderation']
tags: ['common-ground', 'dialogue', 'grounding-gaps']
venue: "NAACL 2024"
tldr: "Analyzes how LLMs fail to establish common ground in conversations, identifying specific grounding gaps in their generations."
---

# Grounding Gaps in Language Model Generations

**Source**: [https://aclanthology.org/2024.naacl-long.348/](https://aclanthology.org/2024.naacl-long.348/)

**TLDR**: Analyzes how LLMs fail to establish common ground in conversations, identifying specific grounding gaps in their generations.

## Abstract

AbstractEffective conversation requires common ground: a shared understanding between the participants. Common ground, however, does not emerge spontaneously in conversation. Speakers and listeners work together to both identify and construct a shared basis while avoiding misunderstanding. To accomplish grounding, humans rely on a range of dialogue acts, like clarification (What do you mean?) and acknowledgment (I understand.). However, it is unclear whether large language models (LLMs) generate text that reflects human grounding. To this end, we curate a set of grounding acts and propose corresponding metrics that quantify attempted grounding. We study whether LLM generations contain grounding acts, simulating turn-taking from several dialogue datasets and comparing results to humans. We find that—compared to humans—LLMs generate language with less conversational grounding, instead generating text that appears to simply presume common ground. To understand the roots of the identified grounding gap, we examine the role of instruction tuning and preference optimization, finding that training on contemporary preference data leads to a reduction in generated grounding acts. Altogether, we highlight the need for more research investigating conversational grounding in human-AI interaction.