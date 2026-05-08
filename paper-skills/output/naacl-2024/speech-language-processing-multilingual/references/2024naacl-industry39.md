---
title: "Search Query Refinement for Japanese Named Entity Recognition in E-commerce Domain"
source: "https://aclanthology.org/2024.naacl-industry.39/"
categories: ['bpe-subword-parsing-evaluation', 'speech-language-processing-multilingual']
tags: ['ner', 'japanese', 'e-commerce', 'query-refinement']
venue: "NAACL 2024"
tldr: "A method for Japanese NER in e-commerce that refines search queries via term splitting and merging to improve canonicalization."
---

# Search Query Refinement for Japanese Named Entity Recognition in E-commerce Domain

**Source**: [https://aclanthology.org/2024.naacl-industry.39/](https://aclanthology.org/2024.naacl-industry.39/)

**TLDR**: A method for Japanese NER in e-commerce that refines search queries via term splitting and merging to improve canonicalization.

## Abstract

AbstractIn the E-Commerce domain, search query refinement reformulates malformed queries into canonicalized forms by preprocessing operations such as “term splitting” and “term merging”. Unfortunately, most relevant research is rather limited to English. In particular, there is a severe lack of study on search query refinement for the Japanese language. Furthermore, no attempt has ever been made to apply refinement methods to data improvement for downstream NLP tasks in real-world scenarios.This paper presents a novel query refinement approach for the Japanese language. Experimental results show that our method achieves significant improvement by 3.5 points through comparison with BERT-CRF as a baseline. Further experiments are also conducted to measure beneficial impact of query refinement on named entity recognition (NER) as the downstream task. Evaluations indicate that the proposed query refinement method contributes to better data quality, leading to performance boost on E-Commerce specific NER tasks by 11.7 points, compared to search query data preprocessed by MeCab, a very popularly adopted Japanese tokenizer.