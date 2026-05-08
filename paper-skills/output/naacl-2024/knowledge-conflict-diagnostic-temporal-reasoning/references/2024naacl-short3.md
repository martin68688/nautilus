---
title: "Improving Toponym Resolution by Predicting Attributes to Constrain Geographical Ontology Entries"
source: "https://aclanthology.org/2024.naacl-short.3/"
categories: ['knowledge-graph-and-information-extraction', 'knowledge-conflict-diagnostic-temporal-reasoning']
tags: ['toponym-resolution', 'geocoding', 'ontology', 'prompting']
venue: "NAACL 2024"
tldr: "Proposes a prompt-based geocoding method where a transformer predicts country and other attributes to constrain candidate locations from a geographical ontology."
---

# Improving Toponym Resolution by Predicting Attributes to Constrain Geographical Ontology Entries

**Source**: [https://aclanthology.org/2024.naacl-short.3/](https://aclanthology.org/2024.naacl-short.3/)

**TLDR**: Proposes a prompt-based geocoding method where a transformer predicts country and other attributes to constrain candidate locations from a geographical ontology.

## Abstract

AbstractGeocoding is the task of converting location mentions in text into structured geospatial data.We propose a new prompt-based paradigm for geocoding, where the machine learning algorithm encodes only the location mention and its context.We design a transformer network for predicting the country, state, and feature class of a location mention, and a deterministic algorithm that leverages the country, state, and feature class predictions as constraints in a search for compatible entries in the ontology.Our architecture, GeoPLACE, achieves new state-of-the-art performance on multiple datasets.Code and models are available at https://github.com/clulab/geonorm.