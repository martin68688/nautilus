---
title: "HPipe: Large Language Model Pipeline Parallelism for Long Context on Heterogeneous Cost-effective Devices"
source: "https://aclanthology.org/2024.naacl-industry.1/"
categories: ['efficient-llm-training-optimization', 'efficient-transformer-acceleration-techniques']
tags: ['pipeline_parallelism', 'long_context', 'heterogeneous_devices']
venue: "NAACL 2024"
tldr: "HPipe enables efficient LLM inference for long sequences on cost-effective heterogeneous devices by optimizing pipeline parallelism."
---

# HPipe: Large Language Model Pipeline Parallelism for Long Context on Heterogeneous Cost-effective Devices

**Source**: [https://aclanthology.org/2024.naacl-industry.1/](https://aclanthology.org/2024.naacl-industry.1/)

**TLDR**: HPipe enables efficient LLM inference for long sequences on cost-effective heterogeneous devices by optimizing pipeline parallelism.

## Abstract

AbstractMicro-enterprises and individual developers emerge analysis demands for long sequence with powerful Large Language Models (LLMs). They try to deploy the LLMs at local, but only possess various commodity devices and the unreliable interconnection between devices. Existing parallel techniques do not lead to the same effectiveness in limited environment. The heterogeneity of devices, coupled with their limited capacity and expensive communication, brings challenges to private deployment for maximized utilization of available devices while masking latency. Hence, we introduce HPipe, a pipeline inference framework that successfully mitigates LLMs from high-performance clusters to heterogeneous commodity devices. By ensuring a balanced distribution of workloads, HPipe facilitates the parallel execution of LLMs through pipelining the sequences on the token dimension. The evaluation conducted on LLaMA-7B and GPT3-2B demonstrates that HPipe holds the potential for context analysis on LLM with heterogeneity devices, achieving an impressive speedup in latency and throughput up to 2.28 times.