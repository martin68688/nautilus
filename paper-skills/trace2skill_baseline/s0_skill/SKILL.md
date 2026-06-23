---
name: small-data-transformer-finetuning
description: Procedural skill for fine-tuning transformer models on small NLP datasets under multi-class log-loss classification. Evolved from mlevolve solver execution traces via the Trace2Skill baseline (creation-from-scratch mode — this skeleton is intentionally minimal; all content below is distilled from traces).
---

# Small-Data Transformer Fine-tuning

Guidance for an automated ML solver fine-tuning large pretrained transformers
(e.g. DeBERTa-v3) on small text-classification datasets where overfitting is the
dominant risk and validation log loss is the objective.
