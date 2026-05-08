---
title: "AfriMTE and AfriCOMET: Enhancing COMET to Embrace Under-resourced African Languages"
source: "https://aclanthology.org/2024.naacl-long.334/"
categories: ['continual-learning-memory-transfer-llms', 'speech-language-processing-multilingual']
tags: ['african-languages', 'translation-evaluation', 'comet']
venue: "NAACL 2024"
tldr: "Enhances the COMET metric for better evaluation of machine translation for under-resourced African languages."
---

# AfriMTE and AfriCOMET: Enhancing COMET to Embrace Under-resourced African Languages

**Source**: [https://aclanthology.org/2024.naacl-long.334/](https://aclanthology.org/2024.naacl-long.334/)

**TLDR**: Enhances the COMET metric for better evaluation of machine translation for under-resourced African languages.

## Abstract

AbstractDespite the recent progress on scaling multilingual machine translation (MT) to several under-resourced African languages, accurately measuring this progress remains challenging, since evaluation is often performed on n-gram matching metrics such as BLEU, which typically show a weaker correlation with human judgments. Learned metrics such as COMET have higher correlation; however, the lack of evaluation data with human ratings for under-resourced languages, complexity of annotation guidelines like Multidimensional Quality Metrics (MQM), and limited language coverage of multilingual encoders have hampered their applicability to African languages. In this paper, we address these challenges by creating high-quality human evaluation data with simplified MQM guidelines for error detection and direct assessment (DA) scoring for 13 typologically diverse African languages. Furthermore, we develop AfriCOMET: COMET evaluation metrics for African languages by leveraging DA data from well-resourced languages and an African-centric multilingual encoder (AfroXLM-R) to create the state-of-the-art MT evaluation metrics for African languages with respect to Spearman-rank correlation with human judgments (0.441).