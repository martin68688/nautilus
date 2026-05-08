---
title: "Solving Data-centric Tasks using Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.41/"
categories: ['continual-learning-memory-transfer-llms', 'tabular-data-llm-prompting-generation']
tags: ['data-centric-tasks', 'spreadsheet', 'data-wrangling']
venue: "NAACL 2024"
tldr: "LLMs are applied to solve data-centric tasks like spreadsheet manipulation and data wrangling for non-professional programmers."
---

# Solving Data-centric Tasks using Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.41/](https://aclanthology.org/2024.findings-naacl.41/)

**TLDR**: LLMs are applied to solve data-centric tasks like spreadsheet manipulation and data wrangling for non-professional programmers.

## Abstract

AbstractLarge language models are rapidly replacing help forums like StackOverflow, and are especially helpful to non-professional programmers and end users. These users are often interested in data-centric tasks, like spreadsheet manipulation and data wrangling, which are hard to solve if the intent is only communicated using a natural-language description, without including data. But how do we decide how much data and which data to include in the prompt?This paper makes two contributions towards answering this question. First, we create a dataset of real-world NL-to-code tasks manipulating tabular data, mined from StackOverflow posts. Second, we introduce a novel cluster-then-select prompting technique, which adds the most representative rows from the input data to the LLM prompt. Our experiments show that LLM performance is indeed sensitive to the amount of data passed in the prompt, and that for tasks with a lot of syntactic variation in the input table,our cluster-then-select technique outperforms a random selection baseline.