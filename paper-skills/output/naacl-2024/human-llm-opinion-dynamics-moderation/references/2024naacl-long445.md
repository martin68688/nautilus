---
title: "Improving In-context Learning of Multilingual Generative Language Models with Cross-lingual Alignment"
source: "https://aclanthology.org/2024.naacl-long.445/"
categories: ['human-llm-opinion-dynamics-moderation', 'speech-language-processing-multilingual']
tags: ['multilingual', 'in-context-learning', 'cross-lingual-alignment', 'generative-models']
venue: "NAACL 2024"
tldr: "A method to improve cross-lingual in-context learning in multilingual generative models by aligning sentence representations across languages, reducing performance bias toward high-resource languages."
---

# Improving In-context Learning of Multilingual Generative Language Models with Cross-lingual Alignment

**Source**: [https://aclanthology.org/2024.naacl-long.445/](https://aclanthology.org/2024.naacl-long.445/)

**TLDR**: A method to improve cross-lingual in-context learning in multilingual generative models by aligning sentence representations across languages, reducing performance bias toward high-resource languages.

## Abstract

AbstractMultilingual generative models obtain remarkable cross-lingual in-context learning capabilities through pre-training on large-scale corpora. However, they still exhibit a performance bias toward high-resource languages and learn isolated distributions of multilingual sentence representations, which may hinder knowledge transfer across languages. To bridge this gap, we propose a simple yet effective cross-lingual alignment framework exploiting pairs of translation sentences. It aligns the internal sentence representations across different languages via multilingual contrastive learning and aligns outputs by following cross-lingual instructions in the target language. Experimental results show that even with less than 0.1‰ of pre-training tokens, our alignment framework significantly boosts the cross-lingual abilities of generative language models and mitigates the performance gap. Further analyses reveal that it results in a better internal multilingual representation distribution of multilingual models.