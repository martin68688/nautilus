---
title: "Know When To Stop: A Study of Semantic Drift in Text Generation"
source: "https://aclanthology.org/2024.naacl-long.202/"
categories: ['large-language-model-evaluation-augmentation', 'code-search-clone-detection']
tags: ['text-generation', 'semantic-drift', 'hallucination']
venue: "NAACL 2024"
tldr: "Studies semantic drift in LLM text generation, developing a score to measure when correct facts transition to incorrect ones."
---

# Know When To Stop: A Study of Semantic Drift in Text Generation

**Source**: [https://aclanthology.org/2024.naacl-long.202/](https://aclanthology.org/2024.naacl-long.202/)

**TLDR**: Studies semantic drift in LLM text generation, developing a score to measure when correct facts transition to incorrect ones.

## Abstract

AbstractIn this work, we explicitly show that modern LLMs tend to generate correct facts first, then “drift away” and generate incorrect facts later: this was occasionally observed but never properly measured. We develop a semantic drift score that measures the degree of separation between correct and incorrect facts in generated texts and confirm our hypothesis when generating Wikipedia-style biographies. This correct-then-incorrect generation pattern suggests that factual accuracy can be improved by knowing when to stop generation. Therefore, we explore the trade-off between information quantity and factual accuracy for several early stopping methods and manage to improve factuality by a large margin. We further show that reranking with semantic similarity can further improve these results, both compared to the baseline and when combined with early stopping. Finally, we try calling external API to bring the model back to the right generation path, but do not get positive results. Overall, our methods generalize and can be applied to any long-form text generation to produce more reliable information, by balancing trade-offs between factual accuracy, information quantity and computational cost.