---
title: "Anonymity at Risk? Assessing Re-Identification Capabilities of Large Language Models in Court Decisions"
source: "https://aclanthology.org/2024.findings-naacl.157/"
categories: ['llm-alignment-safety-detoxification', 'legal-ai-nlp-applications']
tags: ['privacy', 're-identification', 'legal-ai', 'anonymization']
venue: "NAACL 2024"
tldr: "Assesses the re-identification capabilities of LLMs on anonymized court decisions to evaluate privacy risks."
---

# Anonymity at Risk? Assessing Re-Identification Capabilities of Large Language Models in Court Decisions

**Source**: [https://aclanthology.org/2024.findings-naacl.157/](https://aclanthology.org/2024.findings-naacl.157/)

**TLDR**: Assesses the re-identification capabilities of LLMs on anonymized court decisions to evaluate privacy risks.

## Abstract

AbstractAnonymity in court rulings is a critical aspect of privacy protection in the European Union and Switzerland but with the advent of LLMs, concerns about large-scale re-identification of anonymized persons are growing. In accordance with the Federal Supreme Court of Switzerland (FSCS), we study re-identification risks using actual legal data. Following the initial experiment, we constructed an anonymized Wikipedia dataset as a more rigorous testing ground to further investigate the findings. In addition to the datasets, we also introduce new metrics to measure performance. We systematically analyze the factors that influence successful re-identifications, identifying model size, input length, and instruction tuning among the most critical determinants. Despite high re-identification rates on Wikipedia, even the best LLMs struggled with court decisions. We demonstrate that for now, the risk of re-identifications using LLMs is minimal in the vast majority of cases. We hope that our system can help enhance the confidence in the security of anonymized decisions, thus leading the courts to publish more decisions.