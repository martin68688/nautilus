---
title: "Instruction-following Evaluation through Verbalizer Manipulation"
source: "https://aclanthology.org/2024.findings-naacl.233/"
categories: ['llm-evaluation-summarization-argument-extraction', 'zero-shot-few-shot-multimodal-optimization', 'large-language-model-evaluation-augmentation']
tags: ['instruction-following', 'evaluation', 'verbalizer', 'robustness']
venue: "NAACL 2024"
tldr: "Proposes an evaluation method for instruction-following by manipulating verbalizers to test model robustness to unseen instructions."
---

# Instruction-following Evaluation through Verbalizer Manipulation

**Source**: [https://aclanthology.org/2024.findings-naacl.233/](https://aclanthology.org/2024.findings-naacl.233/)

**TLDR**: Proposes an evaluation method for instruction-following by manipulating verbalizers to test model robustness to unseen instructions.

## Abstract

AbstractWhile instruction-tuned models have shown remarkable success in various natural language processing tasks, accurately evaluating their ability to follow instructions remains challenging. Existing benchmarks primarily focus on common instructions that align well with what the model learned during training. However, proficiency in responding to these instructions does not necessarily imply strong ability in instruction following. In this paper, we propose a novel instruction-following evaluation protocol called verbalizer manipulation. It instructs the model to verbalize the task label with words aligning with model priors to different extents, adopting verbalizers from highly aligned (e.g., outputting “positive” for positive sentiment), to minimally aligned (e.g., outputting “negative” for positive sentiment). Verbalizer manipulation can be seamlessly integrated with any classification benchmark to examine the model’s reliance on priors and its ability to override them to accurately follow the instructions. We conduct a comprehensive evaluation of four major model families across nine datasets, employing twelve sets of verbalizers for each of them. We observe that the instruction-following abilities of models, across different families and scales, are significantly distinguished by their performance on less natural verbalizers. Even the strongest GPT-4 model struggles to perform better than random guessing on the most challenging verbalizer, emphasizing the need for continued advancements to improve their instruction-following abilities.