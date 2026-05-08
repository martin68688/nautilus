---
title: "Normalizing without Modernizing: Keeping Historical Wordforms of Middle French while Reducing Spelling Variants"
source: "https://aclanthology.org/2024.findings-naacl.215/"
categories: ['code-search-clone-detection', 'social-media-linguistic-variation']
tags: ['historical-text', 'normalization', 'spelling-variants', 'digital-humanities']
venue: "NAACL 2024"
tldr: "Proposes a method to normalize spelling variants in Middle French while preserving historical wordforms."
---

# Normalizing without Modernizing: Keeping Historical Wordforms of Middle French while Reducing Spelling Variants

**Source**: [https://aclanthology.org/2024.findings-naacl.215/](https://aclanthology.org/2024.findings-naacl.215/)

**TLDR**: Proposes a method to normalize spelling variants in Middle French while preserving historical wordforms.

## Abstract

AbstractConservation of historical documents benefits from computational methods by alleviating the manual labor related to digitization and modernization of textual content. Languages usually evolve over time and keeping historical wordforms is crucial for diachronic studies and digital humanities. However, spelling conventions did not necessarily exist when texts were originally written and orthographic variations are commonly observed depending on scribes and time periods. In this study, we propose to automatically normalize orthographic wordforms found in historical archives written in Middle French during the 16th century without fully modernizing textual content. We leverage pre-trained models in a low resource setting based on a manually curated parallel corpus and produce additional resources with artificial data generation approaches. Results show that causal language models and knowledge distillation improve over a strong baseline, thus validating the proposed methods.