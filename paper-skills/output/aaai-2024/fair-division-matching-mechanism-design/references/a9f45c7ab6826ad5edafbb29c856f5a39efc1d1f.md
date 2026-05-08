---
title: "LLaVA Needs More Knowledge: Retrieval Augmented Natural Language Generation with Knowledge Graph for Explaining Thoracic Pathologies"
source: "https://www.semanticscholar.org/paper/a9f45c7ab6826ad5edafbb29c856f5a39efc1d1f"
categories: ['explainable-ai-and-human-collaboration', 'fair-division-matching-mechanism-design', 'llm-visual-report-generation-analysis']
tags: ['medical-ai', 'retrieval-augmented-generation', 'knowledge-graph']
venue: "AAAI 2024"
tldr: "Enhances medical image explanation generation by retrieving domain knowledge from a graph to augment an LLM."
---

# LLaVA Needs More Knowledge: Retrieval Augmented Natural Language Generation with Knowledge Graph for Explaining Thoracic Pathologies

**Source**: [https://www.semanticscholar.org/paper/a9f45c7ab6826ad5edafbb29c856f5a39efc1d1f](https://www.semanticscholar.org/paper/a9f45c7ab6826ad5edafbb29c856f5a39efc1d1f)

**TLDR**: Enhances medical image explanation generation by retrieving domain knowledge from a graph to augment an LLM.

## Abstract

Generating Natural Language Explanations (NLEs) for model predictions on medical images, particularly those depicting thoracic pathologies, remains a critical and challenging task. Existing methodologies often struggle due to general models' insufficient domain-specific medical knowledge and privacy concerns associated with retrieval-based augmentation techniques. To address these issues, we propose a novel Vision-Language framework augmented with a Knowledge Graph (KG)-based datastore, which enhances the model's understanding by incorporating additional domain-specific medical knowledge essential for generating accurate and informative NLEs. Our framework employs a KG-based retrieval mechanism that not only improves the precision of the generated explanations but also preserves data privacy by avoiding direct data retrieval. The KG datastore is designed as a plug-and-play module, allowing for seamless integration with various model architectures. We introduce and evaluate three distinct frameworks within this paradigm: KG-LLaVA, which integrates the pre-trained LLaVA model with KG-RAG; Med-XPT, a custom framework combining MedCLIP, a transformer-based projector, and GPT-2; and Bio-LLaVA, which adapts LLaVA by incorporating the Bio-ViT-L vision model. These frameworks are validated on the MIMIC-NLE dataset, where they achieve state-of-the-art results, underscoring the effectiveness of KG augmentation in generating high-quality NLEs for thoracic pathologies.