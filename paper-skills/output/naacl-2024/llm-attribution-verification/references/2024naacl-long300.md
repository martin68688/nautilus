---
title: "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer"
source: "https://aclanthology.org/2024.naacl-long.300/"
categories: ['bpe-subword-parsing-evaluation', 'llm-attribution-verification']
tags: ['named-entity-recognition', 'generalist-model']
venue: "NAACL 2024"
tldr: "Presents GLiNER, a generalist bidirectional transformer model for named entity recognition that can extract arbitrary entity types."
---

# GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer

**Source**: [https://aclanthology.org/2024.naacl-long.300/](https://aclanthology.org/2024.naacl-long.300/)

**TLDR**: Presents GLiNER, a generalist bidirectional transformer model for named entity recognition that can extract arbitrary entity types.

## Abstract

AbstractNamed Entity Recognition (NER) is essential in various Natural Language Processing (NLP) applications. Traditional NER models are effective but limited to a set of predefined entity types. In contrast, Large Language Models (LLMs) can extract arbitrary entities through natural language instructions, offering greater flexibility. However, their size and cost, particularly for those accessed via APIs like ChatGPT, make them impractical in resource-limited scenarios. In this paper, we introduce a compact NER model trained to identify any type of entity. Leveraging a bidirectional transformer encoder, our model, GLiNER, facilitates parallel entity extraction, an advantage over the slow sequential token generation of LLMs. Through comprehensive testing, GLiNER demonstrate strong performance, outperforming both ChatGPT and fine-tuned LLMs in zero-shot evaluations on various NER benchmarks.