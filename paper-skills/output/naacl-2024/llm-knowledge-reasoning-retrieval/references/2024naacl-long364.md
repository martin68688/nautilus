---
title: "PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers"
source: "https://aclanthology.org/2024.naacl-long.364/"
categories: ['llm-knowledge-reasoning-retrieval']
tags: ['decision-making', 'rag', 'planning']
venue: "NAACL 2024"
tldr: "Proposes PlanRAG, a plan-then-retrieval augmented generation framework for using LLMs as decision makers on complex data."
---

# PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers

**Source**: [https://aclanthology.org/2024.naacl-long.364/](https://aclanthology.org/2024.naacl-long.364/)

**TLDR**: Proposes PlanRAG, a plan-then-retrieval augmented generation framework for using LLMs as decision makers on complex data.

## Abstract

AbstractIn this paper, we conduct a study to utilize LLMs as a solution for decision making that requires complex data analysis. We define **Decision QA** as the task of answering the best decision, dbest, for a decision-making question Q, business rules R and a database D. Since there is no benchmark that can examine Decision QA, we propose Decision QA benchmark, **DQA**. It has two scenarios, Locating and Building, constructed from two video games (Europa Universalis IV and Victoria 3) that have almost the same goal as Decision QA. To address Decision QA effectively, we also propose a new RAG technique called the *iterative plan-then-retrieval augmented generation* (**PlanRAG**). Our PlanRAG-based LM generates the plan for decision making as the first step, and the retriever generates the queries for data analysis as the second step. The proposed method outperforms the state-of-the-art iterative RAG method by 15.8% in the Locating scenario and by 7.4% in the Building scenario, respectively. We release our code and benchmark at https://github.com/myeon9h/PlanRAG.