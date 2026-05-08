---
title: "Fair Abstractive Summarization of Diverse Perspectives"
source: "https://aclanthology.org/2024.naacl-long.187/"
categories: ['llm-evaluation-summarization-argument-extraction', 'sentiment-emotion-analysis-multimodal-llms']
tags: ['summarization', 'fairness', 'diverse-perspectives']
venue: "NAACL 2024"
tldr: "Addresses fair abstractive summarization to comprehensively cover diverse and conflicting perspectives without underrepresentation."
---

# Fair Abstractive Summarization of Diverse Perspectives

**Source**: [https://aclanthology.org/2024.naacl-long.187/](https://aclanthology.org/2024.naacl-long.187/)

**TLDR**: Addresses fair abstractive summarization to comprehensively cover diverse and conflicting perspectives without underrepresentation.

## Abstract

AbstractPeople from different social and demographic groups express diverse perspectives and conflicting opinions on a broad set of topics such as product reviews, healthcare, law, and politics. A fair summary should provide a comprehensive coverage of diverse perspectives without underrepresenting certain groups. However, current work in summarization metrics and Large Language Models (LLMs) evaluation has not explored fair abstractive summarization. In this paper, we systematically investigate fair abstractive summarization for user-generated data. We first formally define fairness in abstractive summarization as not underrepresenting perspectives of any groups of people, and we propose four reference-free automatic metrics by measuring the differences between target and source perspectives. We evaluate nine LLMs, including three GPT models, four LLaMA models, PaLM 2, and Claude, on six datasets collected from social media, online reviews, and recorded transcripts. Experiments show that both the model-generated and the human-written reference summaries suffer from low fairness. We conduct a comprehensive analysis of the common factors influencing fairness and propose three simple but effective methods to alleviate unfair summarization. Our dataset and code are available at https://github.com/psunlpgroup/FairSumm.