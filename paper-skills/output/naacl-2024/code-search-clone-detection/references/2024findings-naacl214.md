---
title: "Beyond Read-Only: Crafting a Comprehensive Chinese Text-to-SQL Dataset for Database Manipulation and Query"
source: "https://aclanthology.org/2024.findings-naacl.214/"
categories: ['code-search-clone-detection', 'tabular-data-llm-prompting-generation']
tags: ['text-to-sql', 'dataset', 'chinese', 'database-manipulation']
venue: "NAACL 2024"
tldr: "Introduces a comprehensive Chinese Text-to-SQL dataset covering database manipulation beyond read queries."
---

# Beyond Read-Only: Crafting a Comprehensive Chinese Text-to-SQL Dataset for Database Manipulation and Query

**Source**: [https://aclanthology.org/2024.findings-naacl.214/](https://aclanthology.org/2024.findings-naacl.214/)

**TLDR**: Introduces a comprehensive Chinese Text-to-SQL dataset covering database manipulation beyond read queries.

## Abstract

AbstractText-to-SQL aims to convert natural language into structured query language, which is a challenging task. Current research focuses mainly on read operations and ignores other aspects of database operations such as create, update, and delete operations. The benchmark datasets as well as models that have been proposed also fail to cover these operations, limiting the development and practical applications in the field. To bridge this gap, we propose CRUDSQL, a large-scale cross-domain single-table CRUD operations Chinese Text-to-SQL dataset. The dataset contains 10,000 question/SQL pairs involving 625 tables from different domains. To support further research on this dataset, we also propose a baseline method, CRUDParser, which employs a two-phase approach based on BERT and T5 for SQL generation and incorporates two strategies, value matching, and value prompting, for interacting with databases to further improve the performance. The experimental results show that the new operation types bring different challenges for future research, and our approach achieves 67.08% and 83.8% exact set matching accuracy under both read and delete operations in the test set, but only 49.6% and 61.8% under create and update operations. We believe that the proposal of CRUDSQL as well as CRUDParser can provide new directions and possibilities for research and practical applications in the field of Text-to-SQL. The dataset is published at https://github.com/bizard-lab/CRUDSQL.