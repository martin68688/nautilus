---
title: "Is Reference Necessary in the Evaluation of NLG Systems? When and Where?"
source: "https://aclanthology.org/2024.naacl-long.474/"
categories: ['large-language-model-evaluation-augmentation', 'llm-alignment-safety-detoxification']
tags: ['evaluation', 'metrics', 'reference-free']
venue: "NAACL 2024"
tldr: "Investigates when and where reference-free metrics are sufficient for evaluating NLG systems compared to reference-based metrics."
---

# Is Reference Necessary in the Evaluation of NLG Systems? When and Where?

**Source**: [https://aclanthology.org/2024.naacl-long.474/](https://aclanthology.org/2024.naacl-long.474/)

**TLDR**: Investigates when and where reference-free metrics are sufficient for evaluating NLG systems compared to reference-based metrics.

## Abstract

AbstractThe majority of automatic metrics for evaluating NLG systems are reference-based. However, the challenge of collecting human annotation results in a lack of reliable references in numerous application scenarios. Despite recent advancements in reference-free metrics, it has not been well understood when and where they can be used as an alternative to reference-based metrics. In this study, by employing diverse analytical approaches, we comprehensively assess the performance of both metrics across a wide range of NLG tasks, encompassing eight datasets and eight evaluation models. Based on solid experiments, the results show that reference-free metrics exhibit a higher correlation with human judgment and greater sensitivity to deficiencies in language quality. However, their effectiveness varies across tasks and is influenced by the quality of candidate texts. Therefore, it’s important to assess the performance of reference-free metrics before applying them to a new task, especially when inputs are in uncommon form or when the answer space is highly variable. Our study can provide insight into the appropriate application of automatic metrics and the impact of metric choice on evaluation performance.