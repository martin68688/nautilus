---
title: "Fumbling in Babel: An Investigation into ChatGPT’s Language Identification Ability"
source: "https://aclanthology.org/2024.findings-naacl.274/"
categories: ['speech-language-processing-multilingual', 'language-model-cultural-linguistic-evaluation']
tags: ['multilingual', 'language-identification', 'chatgpt']
venue: "NAACL 2024"
tldr: "Investigates ChatGPT's ability to identify languages, revealing strengths and weaknesses across a diverse set."
---

# Fumbling in Babel: An Investigation into ChatGPT’s Language Identification Ability

**Source**: [https://aclanthology.org/2024.findings-naacl.274/](https://aclanthology.org/2024.findings-naacl.274/)

**TLDR**: Investigates ChatGPT's ability to identify languages, revealing strengths and weaknesses across a diverse set.

## Abstract

AbstractChatGPT has recently emerged as a powerful NLP tool that can carry out a variety of tasks. However, the range of languages ChatGPT can handle remains largely a mystery. To uncover which languages ChatGPT ‘knows’, we investigate its language identification (LID) abilities. For this purpose, we compile Babel-670, a benchmark comprising 670 languages representing 23 language families spoken in five continents. Languages in Babel-670 run the gamut from the very high-resource to the very low-resource. We then study ChatGPT’s (both GPT-3.5 and GPT-4) ability to (i) identify language names and language codes (ii) under zero- and few-shot conditions (iii) with and without provision of a label set. When compared to smaller finetuned LID tools, we find that ChatGPT lags behind. For example, it has poor performance on African languages. We conclude that current large language models would benefit from further development before they can sufficiently serve diverse communities.