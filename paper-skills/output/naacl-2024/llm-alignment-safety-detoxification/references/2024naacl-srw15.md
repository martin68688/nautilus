---
title: "Improving Repository-level Code Search with Text Conversion"
source: "https://aclanthology.org/2024.naacl-srw.15/"
categories: ['llm-alignment-safety-detoxification', 'code-search-clone-detection-graph']
tags: ['code-search', 'repository-level', 'retrieval', 'text-conversion']
venue: "NAACL 2024"
tldr: "Improves repository-level code search by converting code structures into text to help LLMs find relevant snippets across files."
---

# Improving Repository-level Code Search with Text Conversion

**Source**: [https://aclanthology.org/2024.naacl-srw.15/](https://aclanthology.org/2024.naacl-srw.15/)

**TLDR**: Improves repository-level code search by converting code structures into text to help LLMs find relevant snippets across files.

## Abstract

AbstractThe ability to generate code using large language models (LLMs) has been increasing year by year. However, studies on code generation at the repository level are not very active. In repository-level code generation, it is necessary to refer to related code snippets among multiple files. By taking the similarity between code snippets, related files are searched and input into an LLM, and generation is performed. This paper proposes a method to search for related files (code search) by taking similarities not between code snippets but between the texts converted from the code snippets by the LLM. We confirmed that converting to text improves the accuracy of code search.