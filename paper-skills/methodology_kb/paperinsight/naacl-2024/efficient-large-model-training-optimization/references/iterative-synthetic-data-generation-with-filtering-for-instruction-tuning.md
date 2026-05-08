---
title: Iterative Synthetic Data Generation with Quality Filtering for Instruction Tuning
confidence: HIGH
papers: [2024findings-naacl.235, 2024findings-naacl.272, 2024findings-naacl.82, 2024naacl-long.421]
---

# Iterative Synthetic Data Generation with Quality Filtering for Instruction Tuning

Multiple papers demonstrate that generating synthetic instruction-tuning data iteratively with quality filtering mechanisms consistently outperforms single-pass generation or using all available data, often achieving better results with only 5-10% of the original dataset size.

**CodecLM** (2024findings-naacl.235) encodes seed instructions into metadata (use case and skills), uses Self-Rubrics to generate metadata-specific complexity adjustments, and applies Contrastive Filtering to select instruction-response pairs where the target LLM's response quality gap with a strong LLM is large. This achieves +7.8 CRR on Evol-Instruct with LLaMA-7B. Iterative Self-Rubrics with up to 4 iterations further improves results, with more than 70% of high-quality data coming from the first iteration.

**GOLD** (2024findings-naacl.272) uses an iterative OOD-guided feedback mechanism where failure modes of the small language model are identified via energy-based OOD scores and fed back to the LLM to generate more diverse training data. With 375 iterations producing 3K samples, GOLD achieves +4% average accuracy over ZeroGen and ProGen on classification tasks. A Symmetric Cross-Entropy (SCE) loss handles noisy labels, contributing +3% accuracy on RTE and MNLI.

**CORGI** (2024findings-naacl.82) uses Contriever-based data filtering to remove low-quality instruction-response pairs, removing 40-50% of instances and yielding significant benefits (ΔMMLU +1.7: 107K→66K). Combined with curriculum ordering, CORGI achieves +4.76 on TruthfulQA and +2.98 on MMLU compared to random shuffling.

**IFD Score** (2024naacl-long.421) uses the Instruction-Following Difficulty score (ratio of Direct Answer Score to Conditioned Answer Score) to select the top 5-10% of samples with highest IFD scores. This outperforms full-data models with only 5-10% of data. A pre-experienced model trained on 1000 diverse samples (via K-means clustering) improves IFD computation. The method also filters out misaligned samples where IFD > 1.

## Papers & Evidence
- `2024findings-naacl.235` (CodecLM): "CodecLM outperforms comparing methods consistently on all benchmarks... The consistently superior performance of CodecLM highlights its generalizability to different downstream instruction distributions and target LLMs." — Metadata encoding + Self-Rubrics + Contrastive Filtering.
- `2024findings-naacl.272` (GOLD): "Our method (with an average accuracy of 67.1%) improves the performance of the pre-trained SLM... Our method demonstrates a 4% improvement over ZeroGen and ProGen." — Iterative OOD-guided feedback with energy-based scoring.
- `2024findings-naacl.82` (CORGI): "Filtering out poor-quality data points yields significant benefits across different data sizes in LLaMA 1 (e.g., ΔMMLU +1.7: 107K → filter 66K)." — Contriever-based relevance filtering.
- `2024naacl-long.421` (IFD Score): "Through the application of IFD, cherry samples can be pinpointed, leading to a marked uptick in model training efficiency. Empirical validations... with a mere 10% of original data input, our strategy showcases improved results." — IFD-based cherry-picking.

## Actionable Guidance
Use an iterative data generation pipeline with quality filtering: (1) generate synthetic data from a strong LLM, (2) score quality using either contrastive filtering (CodecLM), energy-based OOD detection (GOLD), relevance filtering (CORGI), or IFD scoring (IFD), (3) keep only the top 5-10% of data, (4) optionally iterate by feeding back failure modes. Use SCE loss (GOLD) when label noise is expected.

**Condition**: When generating synthetic instruction-tuning data from LLMs where data quality varies and computational budget for training is limited.
**Confidence**: HIGH — Four independent papers with different filtering mechanisms all converge on the same conclusion with quantitative evidence.
