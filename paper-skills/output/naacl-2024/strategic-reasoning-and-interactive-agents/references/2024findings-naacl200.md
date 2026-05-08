---
title: "MindAgent: Emergent Gaming Interaction"
source: "https://aclanthology.org/2024.findings-naacl.200/"
categories: ['legal-ai-nlp-applications', 'strategic-reasoning-and-interactive-agents']
tags: ['multi-agent', 'gaming', 'benchmark', 'foundation-models']
venue: "NAACL 2024"
tldr: "Introduces MindAgent, a benchmark and framework for evaluating emergent multi-agent gaming interaction capabilities of large foundation models."
---

# MindAgent: Emergent Gaming Interaction

**Source**: [https://aclanthology.org/2024.findings-naacl.200/](https://aclanthology.org/2024.findings-naacl.200/)

**TLDR**: Introduces MindAgent, a benchmark and framework for evaluating emergent multi-agent gaming interaction capabilities of large foundation models.

## Abstract

AbstractLarge Foundation Models (LFMs) can perform complex scheduling in a multi-agent system and can coordinate agents to complete sophisticated tasks that require extensive collaboration.However, despite the introduction of numerous gaming frameworks, the community lacks adequate benchmarks that support the implementation of a general multi-agent infrastructure encompassing collaboration between LFMs and human-NPCs. We propose a novel infrastructure—Mindagent—for evaluating planning and coordination capabilities in the context of gaming interaction. In particular, our infrastructure leverages an existing gaming framework to (i) act as the coordinator for a multi-agent system, (ii) collaborate with human players via instructions, and (iii) enable in-context learning based on few-shot prompting with feedback.Furthermore, we introduce “Cuisineworld”, a new gaming scenario and its related benchmark that supervises multiple agents playing the game simultaneously and measures multi-agent collaboration efficiency. We have conducted comprehensive evaluations with a new auto-metric Collaboration Score: CoS for assessing the collaboration efficiency. Finally, Mindagent can be deployed in real-world gaming scenarios in a customized VR version of Cuisineworld and adapted in the “Minecraft” domain. Our work involving LFMs within our new infrastructure for general-purpose scheduling and coordination can elucidate how such skills may be obtained by learning from large language corpora.