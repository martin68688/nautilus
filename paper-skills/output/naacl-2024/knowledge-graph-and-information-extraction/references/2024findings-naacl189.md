---
title: "Multi-Review Fusion-in-Context"
source: "https://aclanthology.org/2024.findings-naacl.189/"
categories: ['knowledge-graph-and-information-extraction', 'llm-evaluation-summarization-argument-extraction']
tags: ['fusion-in-context', 'summarization', 'multi-document']
venue: "NAACL 2024"
tldr: "Proposes a method for grounded text generation that fuses information from multiple reviews within the context window for tasks like summarization."
---

# Multi-Review Fusion-in-Context

**Source**: [https://aclanthology.org/2024.findings-naacl.189/](https://aclanthology.org/2024.findings-naacl.189/)

**TLDR**: Proposes a method for grounded text generation that fuses information from multiple reviews within the context window for tasks like summarization.

## Abstract

AbstractGrounded text generation, encompassing tasks such as long-form question-answering and summarization, necessitates both content selection and content consolidation. Current end-to-end methods are difficult to control and interpret due to their opaqueness.Accordingly, recent works have proposed a modular approach, with separate components for each step. Specifically, we focus on the second subtask, of generating coherent text given pre-selected content in a multi-document setting. Concretely, we formalize Fusion-in-Context (FiC) as a standalone task, whose input consists of source texts with highlighted spans of targeted content. A model then needs to generate a coherent passage that includes all and only the target information.Our work includes the development of a curated dataset of 1000 instances in the reviews domain, alongside a novel evaluation framework for assessing the faithfulness and coverage of highlights, which strongly correlate to human judgment. Several baseline models exhibit promising outcomes and provide insightful analyses.This study lays the groundwork for further exploration of modular text generation in the multi-document setting, offering potential improvements in the quality and reliability of generated content. Our benchmark, FuseReviews, including the dataset, evaluation framework, and designated leaderboard, can be found at https://fusereviews.github.io/.