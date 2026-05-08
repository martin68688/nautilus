---
title: "EDC: Effective and Efficient Dialog Comprehension For Dialog State Tracking"
source: "https://aclanthology.org/2024.naacl-long.232/"
categories: ['llm-knowledge-reasoning-retrieval', 'zero-shot-few-shot-multimodal-optimization']
tags: ['dialog-state-tracking', 'comprehension', 'efficiency', 'attention']
venue: "NAACL 2024"
tldr: "An effective and efficient dialog comprehension model for dialog state tracking using attention mechanisms."
---

# EDC: Effective and Efficient Dialog Comprehension For Dialog State Tracking

**Source**: [https://aclanthology.org/2024.naacl-long.232/](https://aclanthology.org/2024.naacl-long.232/)

**TLDR**: An effective and efficient dialog comprehension model for dialog state tracking using attention mechanisms.

## Abstract

AbstractIn Task-Oriented Dialog (TOD) systems, Dialog State Tracking (DST) structurally extracts information from user and system utterances, which can be further used for querying databases and forming responses to users. The two major categories of DST methods, sequential and independent methods, face trade-offs between accuracy and efficiency. To resolve this issue, we propose Effective and Efficient Dialog Comprehension (EDC), an alternative DST approach that leverages the tree structure of the dialog state. EDC predicts domains, slot names and slot values of the dialog state step-by-step for better accuracy, and efficiently encodes dialog contexts with causal attention patterns. We evaluate EDC on several popular TOD datasets and EDC is able to achieve state-of-the-art Joint Goal Accuracy (JGA). We also show theoretically and empirically that EDC is more efficient than model designs used by previous works.