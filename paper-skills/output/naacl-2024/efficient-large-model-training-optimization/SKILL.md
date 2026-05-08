---
name: efficient-large-model-training-optimization
description: >-
  This skill covers methods for optimizing the training and deployment of large language models (LLMs), focusing on efficiency techniques like low-rank adaptation (LoRA), structured pruning, and memory-augmented architectures for tasks such as dialogue state tracking, summarization, and retrieval-augmented generation. It also involves practical toolkits for fine-tuning, strategies for instruction tuning and distillation to smaller models, and scaling
---

# Efficient Large Model Training Optimization

This skill covers methods for optimizing the training and deployment of large language models (LLMs), focusing on efficiency techniques like low-rank adaptation (LoRA), structured pruning, and memory-augmented architectures for tasks such as dialogue state tracking, summarization, and retrieval-augmented generation. It also involves practical toolkits for fine-tuning, strategies for instruction tuning and distillation to smaller models, and scaling

## Entry Index

| # | Title | Tags | File |
|---|-------|------|------|
| 1 | Adaptive Rank Selections for Low-Rank Approximation of Langu | model-compression, low-rank-approximation, efficiency | 2024naacl-long13.md |
| 2 | Memory Augmented Language Models through Mixture of Word Exp | mixture-of-experts, memory, sparse-models | 2024naacl-long249.md |
| 3 | OrchestraLLM: Efficient Orchestration of Language Models for | dialogue-state-tracking, model-orchestration, efficiency | 2024naacl-long79.md |
| 4 | LMFlow: An Extensible Toolkit for Finetuning and Inference o | toolkit, finetuning, inference | 2024naacl-demo12.md |
| 5 | Self-Regulated Sample Diversity in Large Language Models | sampling, diversity, self-regulation | 2024findings-naacl122.md |
| 6 | Instruction Tuning with Human Curriculum | instruction-tuning, curriculum-learning, model-optimization | 2024findings-naacl82.md |
| 7 | Set-Aligning Framework for Auto-Regressive Event Temporal Gr | efficient-training, auto-regressive, event-graph | 2024naacl-long214.md |
| 8 | Impossible Distillation for Paraphrasing and Summarization:  | distillation, paraphrasing, summarization | 2024naacl-long250.md |
| 9 | SumCSE: Summary as a transformation for Contrastive Learning | contrastive-learning, summarization, data-augmentation | 2024findings-naacl227.md |
| 10 | Effective Long-Context Scaling of Foundation Models | long-context, continual-pretraining, llama | 2024naacl-long260.md |
| 11 | ALoRA: Allocating Low-Rank Adaptation for Fine-tuning Large  | parameter-efficient-finetuning, lora, llm | 2024naacl-long35.md |
| 12 | Structured Pruning for Large Language Models Using Coupled C | model-pruning, llm-efficiency, structured-pruning | 2024findings-naacl1.md |
| 13 | Retrieving Examples from Memory for Retrieval Augmented Neur | retrieval-augmentation, memory, translation | 2024findings-naacl190.md |
| 14 | Efficiently Distilling LLMs for Edge Applications | distillation, edge, supernet | 2024naacl-industry5.md |
| 15 | On Retrieval Augmentation and the Limitations of Language Mo | retrieval-augmentation, kNN, perplexity | 2024naacl-short21.md |
| 16 | MEMORY-VQ: Compression for Tractable Internet-Scale Memory | retrieval-augmentation, compression, memory | 2024naacl-short64.md |
| 17 | Start Simple: Progressive Difficulty Multitask Learning | progressive-learning, curriculum-learning, multitask | 2024naacl-srw7.md |
| 18 | Trusting Your Evidence: Hallucinate Less with Context-aware  | decoding, hallucination, faithfulness | 2024naacl-short69.md |
| 19 | Towards an On-device Agent for Text Rewriting | on-device, text-rewriting, distillation | 2024findings-naacl163.md |
| 20 | The Impact of Depth on Compositional Generalization in Trans | compositional-generalization, transformer, depth | 2024naacl-long402.md |
| 21 | Strings from the Library of Babel: Random Sampling as a Stro | prompt-optimization, baseline, sampling | 2024naacl-long122.md |
| 22 | Extending Input Contexts of Language Models through Training | long-context, training, segmentation | 2024findings-naacl191.md |
| 23 | Accurate Knowledge Distillation via n-best Reranking | knowledge-distillation, model-compression, reranking | 2024naacl-long72.md |
| 24 | CodecLM: Aligning Language Models with Tailored Synthetic Da | instruction-tuning, synthetic-data, alignment | 2024findings-naacl235.md |
| 25 | Fixing Rogue Memorization in Many-to-One Multilingual Transl | low-resource, memorization, indigenous-languages | 2024naacl-long253.md |
| 26 | Improving Machine Translation with Human Feedback: An Explor | reward-model, human-feedback, translation | 2024naacl-long451.md |
| 27 | When Life Gives You Lemons, Make Cherryade: Converting Feedb | dialogue, human-feedback, continuous-improvement | 2024naacl-long169.md |
| 28 | Extremely Weakly-supervised Text Classification with Wordset | weak-supervision, text-classification, denoising | 2024naacl-long397.md |
| 29 | Personalized Federated Learning for Text Classification with | federated-learning, personalization, prompt-tuning | 2024findings-naacl286.md |
| 30 | PELMS: Pre-training for Effective Low-Shot Multi-Document Su | pre-training, multi-document-summarization, low-shot | 2024naacl-long423.md |
| 31 | Reinforcement Learning with Token-level Feedback for Control | reinforcement-learning, controllable-generation, decoding | 2024findings-naacl111.md |
| 32 | VertAttack: Taking Advantage of Text Classifiers’ Horizontal | adversarial-attack, text-classification, robustness | 2024naacl-long41.md |
| 33 | GOLD: Generalized Knowledge Distillation via Out-of-Distribu | knowledge-distillation, data-generation, out-of-distribution | 2024findings-naacl272.md |
| 34 | Mind’s Mirror: Distilling Self-Evaluation Capability and Com | knowledge-distillation, model-evaluation, reasoning | 2024naacl-long376.md |
| 35 | CELI: Simple yet Effective Approach to Enhance Out-of-Domain | cross-encoder, generalization, token-interaction | 2024naacl-short16.md |
| 36 | Neurocache: Efficient Vector Retrieval for Long-range Langua | long-context, retrieval, cache | 2024naacl-long50.md |
| 37 | PaD: Program-aided Distillation Can Teach Small Models Reaso | reasoning, knowledge-distillation, chain-of-thought | 2024naacl-long142.md |
| 38 | Plug-in Language Model: Controlling Text Generation with a S | controlled-generation, language-model, plug-in | 2024findings-naacl139.md |
| 39 | UEGP: Unified Expert-Guided Pre-training for Knowledge Rekin | pre-training, expert-guided, parameter-efficient | 2024findings-naacl170.md |
| 40 | TopicGPT: A Prompt-based Topic Modeling Framework | topic-modeling, llm, prompting | 2024naacl-long164.md |
| 41 | Shears: Unstructured Sparsity with Neural Low-rank Adapter S | sparsity, neural-architecture-search, low-rank-adapters | 2024naacl-industry34.md |
| 42 | Explaining Text Similarity in Transformer Models | transformer, interpretability, similarity | 2024naacl-long435.md |
| 43 | Compensate Quantization Errors: Make Weights Hierarchical to | quantization, model-compression, efficient-inference | 2024findings-naacl173.md |
| 44 | Unveiling the Generalization Power of Fine-Tuned Large Langu | efficient-training, attention-mechanism | 2024naacl-long51.md |
| 45 | HybridBERT - Making BERT Pretraining More Efficient Through  | efficient-pretraining, hybrid-attention | 2024naacl-srw30.md |
| 46 | Contrastive and Consistency Learning for Neural Noisy-Channe | contrastive-learning, noisy-channel, spoken-language-understanding | 2024naacl-long318.md |
| 47 | LM-Infinite: Zero-Shot Extreme Length Generalization for Lar | length-generalization, long-context, transformer | 2024naacl-long222.md |
| 48 | Instructions as Backdoors: Backdoor Vulnerabilities of Instr |  | 2024naacl-long171.md |
| 49 | How does Multi-Task Training Affect Transformer In-Context C | in-context-learning, multi-task-training, transformer-analysis | 2024naacl-short15.md |
| 50 | Unlocking Emergent Modularity in Large Language Models | modularity, emergent-modularity, llm-architecture | 2024naacl-long144.md |
| 51 | Generalizable and Stable Finetuning of Pretrained Language M | fine-tuning, low-resource, generalization | 2024naacl-long277.md |
| 52 | Learning to Compress Prompt in Natural Language Formats | prompt-compression, efficiency, long-context | 2024naacl-long429.md |
| 53 | Human-AI Interaction in the Age of LLMs | in-context-learning, calibration, emergent-abilities | 2024naacl-tutorials5.md |
| 54 | From Quantity to Quality: Boosting LLM Performance with Self | instruction-tuning, data-selection, self-guided | 2024naacl-long421.md |
| 55 | Breaking the Language Barrier: Can Direct Inference Outperfo | efficient-inference, parallel-decoding, decoding-optimization | 2024naacl-short75.md |
| 56 | Lossless Acceleration of Large Language Model via Adaptive N | efficient-inference, parallel-decoding, decoding-optimization | 2024naacl-industry2.md |
| 57 | Anisotropy is Not Inherent to Transformers | embedding-isotropy, representation-degradation, transformers | 2024naacl-long274.md |
| 58 | Teaching Llama a New Language Through Cross-Lingual Knowledg | cross-lingual-transfer, low-resource-language, model-adaptation | 2024findings-naacl210.md |
| 59 | RedCoast: A Lightweight Tool to Automate Distributed Trainin | distributed-training, tool, llm-training | 2024naacl-demo14.md |
| 60 | Efficient Sample-Specific Encoder Perturbations | parameter-efficient, alignment, detoxification | 2024naacl-short57.md |
| 61 | Bridging the Gap between Different Vocabularies for LLM Ense | efficient-training, model-ensemble, vocabulary | 2024naacl-long395.md |
| 62 | PEMA: An Offsite-Tunable Plug-in External Memory Adaptation  | external-memory, parameter-efficient, adaptation | 2024naacl-long336.md |
| 63 | TrojFSP: Trojan Insertion in Few-shot Prompt Tuning | prompt-tuning, security, backdoor-attack | 2024naacl-long64.md |
| 64 | Rehearsal-Free Modular and Compositional Continual Learning  | continual-learning, modular, compositional | 2024naacl-short39.md |
| 65 | DUQGen: Effective Unsupervised Domain Adaptation of Neural R | domain-adaptation, query-generation, ranking | 2024naacl-long413.md |
| 66 | LETI: Learning to Generate from Textual Interactions | fine-tuning, textual-interaction, language-model | 2024findings-naacl16.md |
| 67 | SOLAR 10.7B: Scaling Large Language Models with Simple yet E | model-scaling, depth-upscaling, llm | 2024naacl-industry3.md |
| 68 | MiLe Loss: a New Loss for Mitigating the Bias of Learning Di | loss-function, generative-model, learning-difficulty | 2024findings-naacl18.md |
| 69 | Fine-Tuning Language Models with Reward Learning on Policy | rlhf, alignment, reward-modeling | 2024naacl-long75.md |
| 70 | ARM: Alignment with Residual Energy-Based Model | alignment, energy-based-model, rlhf | 2024naacl-long455.md |
| 71 | Hypernetwork-Assisted Parameter-Efficient Fine-Tuning with M | domain-adaptation, hypernetwork, parameter-efficient | 2024findings-naacl109.md |
| 72 | Tied-LoRA: Enhancing parameter efficiency of LoRA with Weigh | parameter-efficient, lora, weight-tying | 2024naacl-long481.md |
