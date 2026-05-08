---
title: "Language Models can be Deductive Solvers"
source: "https://aclanthology.org/2024.findings-naacl.254/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction']
tags: ['logical-reasoning', 'theorem-proving', 'deduction']
venue: "NAACL 2024"
tldr: "Investigates using language models as deductive solvers for complex logical reasoning."
---

# Language Models can be Deductive Solvers

**Source**: [https://aclanthology.org/2024.findings-naacl.254/](https://aclanthology.org/2024.findings-naacl.254/)

**TLDR**: Investigates using language models as deductive solvers for complex logical reasoning.

## Abstract

AbstractLogical reasoning is a fundamental aspect of human intelligence and a key component of tasks like problem-solving and decision-making. Recent advancements have enabled Large Language Models (LLMs) to potentially exhibit reasoning capabilities, but complex logical reasoning remains a challenge. The state-of-the-art, solver-augmented language models, use LLMs to parse natural language logical questions into symbolic representations first and then adopt external logical solvers to take in the symbolic representations and output the answers. Despite their impressive performance, any parsing errors will inevitably result in the failure of the execution of external logical solvers and no answer to the logical questions. In this paper, we introduce LoGiPT, a novel language model that directly internalizes and emulates the reasoning processes of logical solvers and avoids parsing errors by learning strict adherence to solver syntax and grammar. LoGiPT is fine-tuned on a newly constructed instruction-tuning dataset derived from revealing and refining the invisible reasoning process of deductive solvers. Experimental results on two public deductive reasoning benchmarks show that LoGiPT outperforms state-of-the-art solver-augmented LMs and few-shot prompting methods on competitive LLMs like GPT-4. This project is available in https://github.com/Cyril-JZ/LoGiPT.