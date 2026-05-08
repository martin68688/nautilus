---
title: "AgentQuest: A Modular Benchmark Framework to Measure Progress and Improve LLM Agents"
source: "https://aclanthology.org/2024.naacl-demo.19/"
categories: ['llm-ranking-benchmarking-adaptation', 'large-language-model-evaluation-augmentation']
tags: ['agent', 'benchmark', 'modular']
venue: "NAACL 2024"
tldr: "Introduces a modular benchmark framework to measure and improve LLM agents."
---

# AgentQuest: A Modular Benchmark Framework to Measure Progress and Improve LLM Agents

**Source**: [https://aclanthology.org/2024.naacl-demo.19/](https://aclanthology.org/2024.naacl-demo.19/)

**TLDR**: Introduces a modular benchmark framework to measure and improve LLM agents.

## Abstract

AbstractThe advances made by Large Language Models (LLMs) have led to the pursuit of LLM agents that can solve intricate, multi-step reasoning tasks. As with any research pursuit, benchmarking and evaluation are key corner stones to efficient and reliable progress. However, existing benchmarks are often narrow and simply compute overall task success. To face these issues, we propose AgentQuest – a framework where (i) both benchmarks and metrics are modular and easily extensible through well documented and easy-to-use APIs; (ii) we offer two new evaluation metrics that can reliably track LLM agent progress while solving a task. We exemplify the utility of the metrics on two use cases wherein we identify common failure points and refine the agent architecture to obtain a significant performance increase. Together with the research community, we hope to extend AgentQuest further and therefore we make it available under https://github.com/nec-research/agentquest.