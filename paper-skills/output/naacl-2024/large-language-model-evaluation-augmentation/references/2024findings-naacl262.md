---
title: "Exploring the Trade-off Between Model Performance and Explanation Plausibility of Text Classifiers Using Human Rationales"
source: "https://aclanthology.org/2024.findings-naacl.262/"
categories: ['llm-evaluation-summarization-argument-extraction', 'large-language-model-evaluation-augmentation']
tags: ['explainability', 'human-rationales', 'model-plausibility', 'trade-off']
venue: "NAACL 2024"
tldr: "Explores the trade-off between classifier performance and explanation plausibility using human rationales as a guide."
---

# Exploring the Trade-off Between Model Performance and Explanation Plausibility of Text Classifiers Using Human Rationales

**Source**: [https://aclanthology.org/2024.findings-naacl.262/](https://aclanthology.org/2024.findings-naacl.262/)

**TLDR**: Explores the trade-off between classifier performance and explanation plausibility using human rationales as a guide.

## Abstract

AbstractSaliency post-hoc explainability methods are important tools for understanding increasingly complex NLP models. While these methods can reflect the model’s reasoning, they may not align with human intuition, making the explanations not plausible. In this work, we present a methodology for incorporating rationales, which are text annotations explaining human decisions, into text classification models. This incorporation enhances the plausibility of post-hoc explanations while preserving their faithfulness. Our approach is agnostic to model architectures and explainability methods. We introduce the rationales during model training by augmenting the standard cross-entropy loss with a novel loss function inspired by contrastive learning. By leveraging a multi-objective optimization algorithm, we explore the trade-off between the two loss functions and generate a Pareto-optimal frontier of models that balance performance and plausibility. Through extensive experiments involving diverse models, datasets, and explainability methods, we demonstrate that our approach significantly enhances the quality of model explanations without causing substantial (sometimes negligible) degradation in the original model’s performance.