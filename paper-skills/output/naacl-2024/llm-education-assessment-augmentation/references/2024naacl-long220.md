---
title: "Pedagogically Aligned Objectives Create Reliable Automatic Cloze Tests"
source: "https://aclanthology.org/2024.naacl-long.220/"
categories: ['llm-education-assessment-augmentation', 'bpe-subword-parsing-evaluation']
tags: ['education', 'cloze-test', 'distractor-generation']
venue: "NAACL 2024"
tldr: "Shows that training MLMs with pedagogically aligned objectives improves automatic generation of reliable cloze test distractors."
---

# Pedagogically Aligned Objectives Create Reliable Automatic Cloze Tests

**Source**: [https://aclanthology.org/2024.naacl-long.220/](https://aclanthology.org/2024.naacl-long.220/)

**TLDR**: Shows that training MLMs with pedagogically aligned objectives improves automatic generation of reliable cloze test distractors.

## Abstract

AbstractThe cloze training objective of Masked Language Models makes them a natural choice for generating plausible distractors for human cloze questions. However, distractors must also be both distinct and incorrect, neither of which is directly addressed by existing neural methods. Evaluation of recent models has also relied largely on automated metrics, which cannot demonstrate the reliability or validity of human comprehension tests. In this work, we first formulate the pedagogically motivated objectives of plausibility, incorrectness, and distinctiveness in terms of conditional distributions from language models. Second, we present an unsupervised, interpretable method that uses these objectives to jointly optimize sets of distractors. Third, we test the reliability and validity of the resulting cloze tests compared to other methods with human participants. We find our method has stronger correlation with teacher-created comprehension tests than the state-of-the-art neural method and is more internally consistent. Our implementation is freely available and can quickly create a multiple choice cloze test from any given passage.