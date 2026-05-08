---
title: "CoMM: Collaborative Multi-Agent, Multi-Reasoning-Path Prompting for Complex Problem Solving"
source: "https://aclanthology.org/2024.findings-naacl.112/"
categories: ['llm-reasoning-retrieval-evaluation']
tags: ['reasoning', 'multi-agent', 'prompting']
venue: "NAACL 2024"
tldr: "Proposes a collaborative multi-agent prompting framework to enhance LLM performance on complex science problems."
---

# CoMM: Collaborative Multi-Agent, Multi-Reasoning-Path Prompting for Complex Problem Solving

**Source**: [https://aclanthology.org/2024.findings-naacl.112/](https://aclanthology.org/2024.findings-naacl.112/)

**TLDR**: Proposes a collaborative multi-agent prompting framework to enhance LLM performance on complex science problems.

## Abstract

AbstractLarge Language Models (LLMs) have shown great ability in solving traditional natural language tasks and elementary reasoning tasks with appropriate prompting techniques. However, their ability is still limited in solving complicated science problems. In this work, we aim to push the upper bound of the reasoning capability of LLMs by proposing a collaborative multi-agent, multi-reasoning-path (CoMM) prompting framework. Specifically, we prompt LLMs to play different roles in a problem-solving team, and encourage different role-play agents to collaboratively solve the target task. In particular, we discover that applying different reasoning paths for different roles is an effective strategy to implement few-shot prompting approaches in the multi-agent scenarios. Empirical results demonstrate the effectiveness of the proposed methods on two college-level science problems over competitive baselines. Our further analysis shows the necessity of prompting LLMs to play different roles or experts independently.