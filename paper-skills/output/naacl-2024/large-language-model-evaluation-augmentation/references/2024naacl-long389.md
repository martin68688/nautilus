---
title: "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"
source: "https://aclanthology.org/2024.naacl-long.389/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation']
tags: ['rag', 'retrieval', 'adaptation', 'question-answering']
venue: "NAACL 2024"
tldr: "Adaptive-RAG learns to adjust retrieval augmentation for LLMs based on question complexity to improve answer accuracy."
---

# Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity

**Source**: [https://aclanthology.org/2024.naacl-long.389/](https://aclanthology.org/2024.naacl-long.389/)

**TLDR**: Adaptive-RAG learns to adjust retrieval augmentation for LLMs based on question complexity to improve answer accuracy.

## Abstract

AbstractRetrieval-Augmented Large Language Models (LLMs), which incorporate the non-parametric knowledge from external knowledge bases into LLMs, have emerged as a promising approach to enhancing response accuracy in several tasks, such as Question-Answering (QA). However, even though there are various approaches dealing with queries of different complexities, they either handle simple queries with unnecessary computational overhead or fail to adequately address complex multi-step queries; yet, not all user requests fall into only one of the simple or complex categories. In this work, we propose a novel adaptive QA framework that can dynamically select the most suitable strategy for (retrieval-augmented) LLMs from the simplest to the most sophisticated ones based on the query complexity. Also, this selection process is operationalized with a classifier, which is a smaller LM trained to predict the complexity level of incoming queries with automatically collected labels, obtained from actual predicted outcomes of models and inherent inductive biases in datasets. This approach offers a balanced strategy, seamlessly adapting between the iterative and single-step retrieval-augmented LLMs, as well as the no-retrieval methods, in response to a range of query complexities. We validate our model on a set of open-domain QA datasets, covering multiple query complexities, and show that ours enhances the overall efficiency and accuracy of QA systems, compared to relevant baselines including the adaptive retrieval approaches. Code is available at: https://github.com/starsuzi/Adaptive-RAG.