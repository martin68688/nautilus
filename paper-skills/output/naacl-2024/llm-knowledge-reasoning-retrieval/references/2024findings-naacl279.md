---
title: "ATG: Benchmarking Automated Theorem Generation for Generative Language Models"
source: "https://aclanthology.org/2024.findings-naacl.279/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-conflict-diagnostic-temporal-reasoning']
tags: ['argumentation', 'causality', 'sufficiency']
venue: "NAACL 2024"
tldr: "A causality-driven method for argument sufficiency assessment that uses LLMs to generate counterfactuals instead of relying on human annotations."
---

# ATG: Benchmarking Automated Theorem Generation for Generative Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.279/](https://aclanthology.org/2024.findings-naacl.279/)

**TLDR**: A causality-driven method for argument sufficiency assessment that uses LLMs to generate counterfactuals instead of relying on human annotations.

## Abstract

AbstractHumans can develop new theorems to explore broader and more complex mathematical results.While current generative language models (LMs) have achieved significant improvement in automatically proving theorems, their ability to generate new or reusable theorems is still under-explored. Without the new theorems, current LMs struggle to prove harder theorems that are distant from the given hypotheses with the exponentially growing search space.More advanced theorem proving is if an agent (for instance, a generative LM) can leverage its creativity to generate new but also reasonable theorems that properly substitute part of a proof and also be saved as reusable knowledge for future theorem proving.Therefore, this paper proposes an Automated Theorem Generation (ATG) benchmark that evaluates whether an agent can automatically generate valuable (and possibly brand new) theorems that are applicable for downstream theorem proving as reusable knowledge. Specifically, we construct the ATG benchmark by splitting the Metamath library into three sets: axioms, library, and problem based on their proving depth.We conduct extensive experiments to investigate whether current LMs can generate theorems in the library and benefit the problem theorems proving. The results demonstrate that high-quality ATG data facilitates models’ performances on downstream ATP. However, there is still room for current LMs to develop better ATG and generate more advanced and human-like theorems. We hope the new ATG challenge can shed some light on advanced complex theorem proving.