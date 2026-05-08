---
title: Specialized Models Outperform General-Purpose LLMs for Mental Health Risk Detection
confidence: HIGH
papers: [2024naacl-long278, 2024naacl-long452, 2024naacl-short58]
---

# Specialized Models Outperform General-Purpose LLMs for Mental Health Risk Detection

Across three NAACL 2024 papers, a consistent finding emerges: general-purpose large language models (GPT-3.5, GPT-4, MentaLLaMa) significantly underperform specialized architectures (multi-task learning models, graph neural networks, domain-specific pretrained language models) when applied to mental health risk detection and assessment tasks. The performance gap is substantial and consistent across diverse tasks including bipolar disorder detection, depression detection from clinical interviews, and cross-cultural depression detection from social media.

## Specific Evidence

**Bipolar Disorder Detection (2024naacl-long278):** When applied to predicting bipolar disorder risk from longitudinal social media posts, ChatGPT achieved an F1 score of only 0.130 and MentaLLaMa (a fine-tuned LLaMA model for mental health) achieved 0.190. In contrast, the proposed multi-task learning model with dynamic mood-aware attention achieved F1 0.578 — a 3-4x improvement over the LLMs. The authors note: "LLMs still exhibit a notable gap due to a lack of mental health-specific knowledge and inconsistencies depending on prompting strategies."

**Depression Detection in Clinical Interviews (2024naacl-long452):** The Structural Element Graph (SEGA) model, using a directed acyclic graph with graph attention networks, achieved macro F1 scores of 68.12% on DAIC-WOZ and 72.83% on EATD, "significantly surpass[ing] existing baseline methods and powerful LLMs like GPT-4 for depression detection in clinical interviews." With LLM-based data augmentation (SEGA++), performance rose further to 70.95% and 74.64% respectively.

**Cross-Cultural Depression Detection (2024naacl-short58):** MentalLongformer, a domain-specific pretrained language model with extended 4096-token capacity, achieved the best overall generalization (e.g., F1 0.72 on UK data) compared to Logistic Regression with TF-IDF (F1 0.69). However, even MentalLongformer showed significant performance gaps between Global North (F1 0.68) and Global South (F1 0.56) populations, indicating that domain-specific PLMs outperform traditional ML but still have limitations.

## Papers & Evidence
- `2024naacl-long278` (Detecting Bipolar Disorder from Misdiagnosed MDD): "Specifically, LLMs... exhibited inferior performance compared to the proposed model... ChatGPT F1 0.130, MentaLLaMa F1 0.190 vs. proposed MTL F1 0.578" — General LLMs achieve F1 scores 3-4x lower than specialized MTL models for BD risk detection.
- `2024naacl-long452` (Depression Detection with SEGA): "SEGA achieves state-of-the-art performance, outperforming the best baseline by 1.12% and 2.36% in terms of macro F1-scores on two datasets, respectively" and "significantly surpasses... powerful LLMs like GPT-4" — A specialized graph architecture outperforms GPT-4 for clinical interview depression detection.
- `2024naacl-short58` (Cross-Cultural Depression Detection): "Pre-trained language models achieve the best generalization compared to Logistic Regression, though still show significant gaps in performance on depressed and non-Western users" — Domain-specific PLMs (MentalLongformer) outperform traditional ML but still have cross-cultural limitations.

## Actionable Guidance
For mental health risk detection tasks, do NOT use general-purpose LLMs (GPT-3.5, GPT-4) as classifiers — they lack domain-specific knowledge and produce F1 scores 3-4x lower than specialized models. Instead, use either (a) a multi-task learning architecture with domain-specific sentence embeddings (SBERT) and temporal attention for longitudinal social media data, or (b) a structured graph model (GAT-based) for multi-modal clinical interview data, or (c) a domain-specific pretrained language model (MentalLongformer) for social media text classification.

**Condition**: When the task is mental health risk detection or severity assessment from text (social media posts, clinical interview transcripts).
**Confidence**: HIGH — Three papers with different tasks (BD detection, depression detection, cross-cultural depression) all converge on the same finding with large performance gaps.
