---
title: "Comparing Explanation Faithfulness between Multilingual and Monolingual Fine-tuned Language Models"
source: "https://aclanthology.org/2024.naacl-long.178/"
categories: ['legal-ai-nlp-applications', 'transformer-language-model-probing']
tags: ['explainability', 'multilingual', 'faithfulness']
venue: "NAACL 2024"
tldr: "Compares explanation faithfulness between multilingual and monolingual fine-tuned language models using feature attribution methods."
---

# Comparing Explanation Faithfulness between Multilingual and Monolingual Fine-tuned Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.178/](https://aclanthology.org/2024.naacl-long.178/)

**TLDR**: Compares explanation faithfulness between multilingual and monolingual fine-tuned language models using feature attribution methods.

## Abstract

AbstractIn many real natural language processing application scenarios, practitioners not only aim to maximize predictive performance but also seek faithful explanations for the model predictions. Rationales and importance distribution given by feature attribution methods (FAs) provide insights into how different parts of the input contribute to a prediction. Previous studies have explored how different factors affect faithfulness, mainly in the context of monolingual English models. On the other hand, the differences in FA faithfulness between multilingual and monolingual models have yet to be explored. Our extensive experiments, covering five languages and five popular FAs, show that FA faithfulness varies between multilingual and monolingual models. We find that the larger the multilingual model, the less faithful the FAs are compared to its counterpart monolingual models. Our further analysis shows that the faithfulness disparity is potentially driven by the differences between model tokenizers. Our code is available: https://github.com/casszhao/multilingual-faith.