---
title: "A Measure for Transparent Comparison of Linguistic Diversity in Multilingual NLP Data Sets"
source: "https://aclanthology.org/2024.findings-naacl.213/"
categories: ['bpe-subword-parsing-evaluation', 'llm-evaluation-summarization-argument-extraction']
tags: ['linguistic-diversity', 'benchmarking', 'multilingual-nlp', 'metrics']
venue: "NAACL 2024"
tldr: "Proposes a measure for transparently comparing the linguistic diversity of multilingual NLP datasets beyond simple language counts."
---

# A Measure for Transparent Comparison of Linguistic Diversity in Multilingual NLP Data Sets

**Source**: [https://aclanthology.org/2024.findings-naacl.213/](https://aclanthology.org/2024.findings-naacl.213/)

**TLDR**: Proposes a measure for transparently comparing the linguistic diversity of multilingual NLP datasets beyond simple language counts.

## Abstract

AbstractTypologically diverse benchmarks are increasingly created to track the progress achieved in multilingual NLP. Linguistic diversity of these data sets is typically measured as the number of languages or language families included in the sample, but such measures do not consider structural properties of the included languages. In this paper, we propose assessing linguistic diversity of a data set against a reference language sample as a means of maximising linguistic diversity in the long run. We represent languages as sets of features and apply a version of the Jaccard index suitable for comparing sets of measures. In addition to the features extracted from typological data bases, we propose an automatic text-based measure, which can be used as a means of overcoming the well-known problem of data sparsity in manually collected features. Our diversity score is interpretable in terms of linguistic features and can identify the types of languages that are not represented in a data set. Using our method, we analyse a range of popular multilingual data sets (UD, Bible100, mBERT, XTREME, XGLUE, XNLI, XCOPA, TyDiQA, XQuAD). In addition to ranking these data sets, we find, for example, that (poly)synthetic languages are missing in almost all of them.