---
title: "X-Eval: Generalizable Multi-aspect Text Evaluation via Augmented Instruction Tuning with Auxiliary Evaluation Aspects"
source: "https://aclanthology.org/2024.naacl-long.473/"
categories: ['llm-evaluation-summarization-argument-extraction', 'large-language-model-evaluation-augmentation']
tags: ['evaluation', 'instruction-tuning', 'multi-aspect']
venue: "NAACL 2024"
tldr: "Introduces X-Eval, a method for generalizable multi-aspect text evaluation via instruction tuning augmented with auxiliary evaluation aspects."
---

# X-Eval: Generalizable Multi-aspect Text Evaluation via Augmented Instruction Tuning with Auxiliary Evaluation Aspects

**Source**: [https://aclanthology.org/2024.naacl-long.473/](https://aclanthology.org/2024.naacl-long.473/)

**TLDR**: Introduces X-Eval, a method for generalizable multi-aspect text evaluation via instruction tuning augmented with auxiliary evaluation aspects.

## Abstract

AbstractNatural Language Generation (NLG) typically involves evaluating the generated text in various aspects (e.g., consistency and naturalness) to obtain a comprehensive assessment. However, multi-aspect evaluation remains challenging as it may require the evaluator to generalize to any given evaluation aspect even if it’s absent during training. In this paper, we introduce X-Eval, a two-stage instruction tuning framework to evaluate text in both seen and unseen aspects customized by end users. X-Eval consists of two learning stages: the vanilla instruction tuning stage that improves the model’s ability to follow evaluation instructions, and an enhanced instruction tuning stage that exploits the connections between fine-grained evaluation aspects to better assess text quality. To support the training of X-Eval, we collect AspectInstruct, the first instruction tuning dataset tailored for multi-aspect NLG evaluation spanning 27 diverse evaluation aspects with 65 tasks. To enhance task diversity, we devise an augmentation strategy that converts human rating annotations into diverse forms of NLG evaluation tasks, including scoring, comparison, ranking, and Boolean question answering. Extensive experiments across three essential categories of NLG tasks: dialogue generation, summarization, and data-to-text coupled with 21 aspects in meta-evaluation, demonstrate that X-Eval enables even a lightweight language model to achieve a comparable if not higher correlation with human judgments compared to the state-of-the-art NLG evaluators like GPT-4.