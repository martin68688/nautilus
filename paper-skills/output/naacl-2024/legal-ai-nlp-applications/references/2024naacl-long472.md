---
title: "LegalDiscourse: Interpreting When Laws Apply and To Whom"
source: "https://aclanthology.org/2024.naacl-long.472/"
categories: ['human-llm-opinion-dynamics-moderation', 'legal-ai-nlp-applications']
tags: ['legal-ai', 'discourse-parsing', 'taxonomy']
venue: "NAACL 2024"
tldr: "Presents a discourse parsing taxonomy for legal texts to interpret when laws apply and to whom."
---

# LegalDiscourse: Interpreting When Laws Apply and To Whom

**Source**: [https://aclanthology.org/2024.naacl-long.472/](https://aclanthology.org/2024.naacl-long.472/)

**TLDR**: Presents a discourse parsing taxonomy for legal texts to interpret when laws apply and to whom.

## Abstract

AbstractWhile legal AI has made strides in recent years, it still struggles with basic legal concepts: _when_ does a law apply? _Who_ does it applies to? _What_ does it do? We take a _discourse_ approach to addressing these problems and introduce a novel taxonomy for span-and-relation parsing of legal texts. We create a dataset, _LegalDiscourse_ of 602 state-level law paragraphs consisting of 3,715 discourse spans and 1,671 relations. Our trained annotators have an agreement-rate 𝜅>.8, yet few-shot GPT3.5 performs poorly at span identification and relation classification. Although fine-tuning improves performance, GPT3.5 still lags far below human level. We demonstrate the usefulness of our schema by creating a web application with journalists. We collect over 100,000 laws for 52 U.S. states and territories using 20 scrapers we built, and apply our trained models to 6,000 laws using U.S. Census population numbers. We describe two journalistic outputs stemming from this application: (1) an investigation into the increase in liquor licenses following population growth and (2) a decrease in applicable laws under different under-count projections.