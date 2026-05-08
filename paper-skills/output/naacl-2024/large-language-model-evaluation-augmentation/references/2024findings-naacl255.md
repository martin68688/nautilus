---
title: "Interpreting User Requests in the Context of Natural Language Standing Instructions"
source: "https://aclanthology.org/2024.findings-naacl.255/"
categories: ['large-language-model-evaluation-augmentation', 'query-aware-attention-user-modeling']
tags: ['dialogue', 'personalization', 'user-modeling']
venue: "NAACL 2024"
tldr: "Describes an approach for LLM-based dialogue modeling that incorporates persistent user constraints and preferences as standing instructions."
---

# Interpreting User Requests in the Context of Natural Language Standing Instructions

**Source**: [https://aclanthology.org/2024.findings-naacl.255/](https://aclanthology.org/2024.findings-naacl.255/)

**TLDR**: Describes an approach for LLM-based dialogue modeling that incorporates persistent user constraints and preferences as standing instructions.

## Abstract

AbstractUsers of natural language interfaces, frequently powered by Large Language Models (LLMs), must often repeat their full set of preferences each time they make a similar request. We describe an approach to LLM-based dialogue modeling in which persistent user constraints and preferences – collectively termed standing instructions – are provided as additional context for such interfaces. For example, when a user states “I’m hungry”, a previously expressed preference for Persian food can be automatically added to the LLM prompt, influencing the search for relevant restaurants.We develop NLSI, a language-to-program dataset consisting of over 2.4K English dialogues spanning 17 domains, in which each dialogue is paired with a user profile (a set of user-specific standing instructions) and corresponding structured representations (a sequence of API calls). A key challenge in NLSI is to identify which subset of the standing instructions is applicable to a given dialogue. NLSI contains diverse phenomena, from simple preferences to interdependent instructions such as triggering a hotel search whenever the user is booking tickets to an event. We conduct experiments on NLSI using prompting with large language models and various retrieval approaches, achieving a maximum of 46% exact match on API prediction. Our results demonstrate the challenges in identifying the relevant standing instructions and their interpretation into API calls