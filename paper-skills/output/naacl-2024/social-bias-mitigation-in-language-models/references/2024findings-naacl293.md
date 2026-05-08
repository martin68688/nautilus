---
title: "Contrastive Learning as a Polarizer: Mitigating Gender Bias by Fair and Biased sentences"
source: "https://aclanthology.org/2024.findings-naacl.293/"
categories: ['llm-ranking-benchmarking-adaptation', 'social-bias-mitigation-in-language-models']
tags: ['contrastive-learning', 'debiasing', 'gender-bias']
venue: "NAACL 2024"
tldr: "Mitigates gender bias in language models using contrastive learning with fair and biased sentences."
---

# Contrastive Learning as a Polarizer: Mitigating Gender Bias by Fair and Biased sentences

**Source**: [https://aclanthology.org/2024.findings-naacl.293/](https://aclanthology.org/2024.findings-naacl.293/)

**TLDR**: Mitigates gender bias in language models using contrastive learning with fair and biased sentences.

## Abstract

AbstractRecently, language models have accelerated the improvement in natural language processing. However, recent studies have highlighted a significant issue: social biases inherent in training data can lead models to learn and propagate these biases. In this study, we propose a contrastive learning method for bias mitigation, utilizing anchor points to push further negatives and pull closer positives within the representation space. This approach employs stereotypical data as negatives and stereotype-free data as positives, enhancing debiasing performance. Our model attained state-of-the-art performance in the ICAT score on the StereoSet, a benchmark for measuring bias in models. In addition, we observed that effective debiasing is achieved through an awareness of biases, as evidenced by improved hate speech detection scores. The implementation code and trained models are available at https://github.com/HUFS-NLP/CL_Polarizer.git.