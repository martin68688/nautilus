---
title: "Lost in Translation? Translation Errors and Challenges for Fair Assessment of Text-to-Image Models on Multilingual Concepts"
source: "https://aclanthology.org/2024.naacl-short.48/"
categories: ['metaphor-analysis-political-framing', 'language-model-cultural-linguistic-evaluation']
tags: ['text-to-image', 'multilingual-evaluation', 'translation-error']
venue: "NAACL 2024"
tldr: "Highlights translation errors as a key challenge for fair multilingual assessment of text-to-image models on conceptual benchmarks."
---

# Lost in Translation? Translation Errors and Challenges for Fair Assessment of Text-to-Image Models on Multilingual Concepts

**Source**: [https://aclanthology.org/2024.naacl-short.48/](https://aclanthology.org/2024.naacl-short.48/)

**TLDR**: Highlights translation errors as a key challenge for fair multilingual assessment of text-to-image models on conceptual benchmarks.

## Abstract

AbstractBenchmarks of the multilingual capabilities of text-to-image (T2I) models compare generated images prompted in a test language to an expected image distribution over a concept set. One such benchmark, “Conceptual Coverage Across Languages” (CoCo-CroLa), assesses the tangible noun inventory of T2I models by prompting them to generate pictures from a concept list translated to seven languages and comparing the output image populations. Unfortunately, we find that this benchmark contains translation errors of varying severity in Spanish, Japanese, and Chinese. We provide corrections for these errors and analyze how impactful they are on the utility and validity of CoCo-CroLa as a benchmark. We reassess multiple baseline T2I models with the revisions, compare the outputs elicited under the new translations to those conditioned on the old, and show that a correction’s impactfulness on the image-domain benchmark results can be predicted in the text domain with similarity scores. Our findings will guide the future development of T2I multilinguality metrics by providing analytical tools for practical translation decisions.