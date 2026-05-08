---
title: "XNLIeu: a dataset for cross-lingual NLI in Basque"
source: "https://aclanthology.org/2024.naacl-long.234/"
categories: ['llm-alignment-safety-detoxification', 'speech-language-processing-multilingual', 'language-model-evaluation-benchmarks']
tags: ['basque', 'cross-lingual-nli', 'low-resource-language']
venue: "NAACL 2024"
tldr: "Expands XNLI to include Basque for cross-lingual natural language inference evaluation."
---

# XNLIeu: a dataset for cross-lingual NLI in Basque

**Source**: [https://aclanthology.org/2024.naacl-long.234/](https://aclanthology.org/2024.naacl-long.234/)

**TLDR**: Expands XNLI to include Basque for cross-lingual natural language inference evaluation.

## Abstract

AbstractXNLI is a popular Natural Language Inference (NLI) benchmark widely used to evaluate cross-lingual Natural Language Understanding (NLU) capabilities across languages. In this paper, we expand XNLI to include Basque, a low-resource language that can greatly benefit from transfer-learning approaches. The new dataset, dubbed XNLIeu, has been developed by first machine-translating the English XNLI corpus into Basque, followed by a manual post-edition step. We have conducted a series of experiments using mono- and multilingual LLMs to assess a) the effect of professional post-edition on the MT system; b) the best cross-lingual strategy for NLI in Basque; and c) whether the choice of the best cross-lingual strategy is influenced by the fact that the dataset is built by translation. The results show that post-edition is necessary and that the translate-train cross-lingual strategy obtains better results overall, although the gain is lower when tested in a dataset that has been built natively from scratch. Our code and datasets are publicly available under open licenses.