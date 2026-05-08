---
title: "Explainable CED: A Dataset for Explainable Critical Error Detection in Machine Translation"
source: "https://aclanthology.org/2024.naacl-srw.4/"
categories: ['language-model-evaluation-benchmarks', 'llm-attribution-verification']
tags: ['machine-translation', 'error-detection', 'explainability', 'dataset']
venue: "NAACL 2024"
tldr: "A dataset for explainable critical error detection in machine translation, providing reasons for catastrophic meaning distortions."
---

# Explainable CED: A Dataset for Explainable Critical Error Detection in Machine Translation

**Source**: [https://aclanthology.org/2024.naacl-srw.4/](https://aclanthology.org/2024.naacl-srw.4/)

**TLDR**: A dataset for explainable critical error detection in machine translation, providing reasons for catastrophic meaning distortions.

## Abstract

AbstractCritical error detection (CED) in machine translation is a task that aims to detect errors that significantly distort the intended meaning. However, the existing study of CED lacks explainability due to the absence of content addressing the reasons for catastrophic errors. To address this limitation, we propose Explainable CED, a dataset that introduces the attributes of error explanation and correction regarding critical errors. Considering the advantage of reducing time costs and mitigating human annotation bias, we leverage a large language model in the data construction process. To improve the quality of the dataset and mitigate hallucination, we compare responses from the model and introduce an additional data filtering method through feedback scoring. The experiment demonstrates that the dataset appropriately reflects a consistent explanation and revision for errors, validating the reliability of the dataset.