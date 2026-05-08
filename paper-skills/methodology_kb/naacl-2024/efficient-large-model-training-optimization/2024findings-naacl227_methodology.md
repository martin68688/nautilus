# SumCSE: Summary as a transformation for Contrastive Learning

**Source**: https://aclanthology.org/2024.findings-naacl.227/

## [POSITIVE] Summary Transformation for Positives
Using a generative LM to summarize an anchor sentence (in ~7 words) to create a positive example, filtering out irrelevant information while retaining core meaning.

**Delta**: Outperforms all other single transformations (Entailment, Paraphrase, etc.) on STS-B validation (86.97 vs. 85.13 for Paraphrase).
**Condition**: When generating positive pairs for contrastive learning on sentence similarity tasks (STS).

**Evidence**: "Summary outperforms all other transformations in Table 1."

## [POSITIVE] Compositional Transformations (Summary + Paraphrase/Contradiction)
Applying a second transformation (e.g., summary) on top of a first transformation (e.g., paraphrase or contradiction) to generate positives and negatives.

**Delta**: Random summary compositions outperform all other transformations (Table 2: 87.3 vs. 86.88 for Entailment as second transformation).
**Condition**: When generating both positive and negative pairs; composition of summary with diverse paraphrasing/contradiction prompts.

**Evidence**: "Random summary compositions outperform all other transformations in last row Table 2."

## [POSITIVE] Summary Composition for Negatives
First generating a contradiction of the anchor, then summarizing that contradiction to create a negative example. This shortens negatives and improves anisotropy.

**Delta**: Improves STS-B validation from 85.6 (without summary) to 88.13 (with summary) and reduces anisotropy from 0.132 to 0.105 (Table 8).
**Condition**: When generating negative pairs; especially effective when positives are short.

**Evidence**: "Summarizing negatives makes a big difference in validation performance and anisotropy (Table 8, rows 3 vs 5)."

## [POSITIVE] Random Selection Among Diverse Summary Compositions
For each anchor, randomly selecting one of four summary compositions (over Entailment, Sentence Structure Change, Paraphrase, Concise Paraphrase) as the positive, and one of four summary compositions over contradiction prompts as the negative.

**Delta**: Achieves best STS-B validation (87.3) compared to any single composition (Table 2). Final SumCSE outperforms SynCSE by +1.8 and SimCSE by +0.3 on STS benchmark (Table 3).
**Condition**: Final dataset generation for training sentence embedding models on STS.

**Evidence**: "Random summary compositions outperform all other transformations (Table 2). SumCSE significantly outperforms SynCSE and SimCSE (Table 3)."

## [POSITIVE] Using Vicuna-13B Instead of ChatGPT for Generation
Using the open-source Vicuna-13B-v1.3 model (instead of ChatGPT) to generate the synthetic contrastive learning data.

**Delta**: SumCSE (Vicuna-13B) outperforms SynCSE (ChatGPT) by +1.8 on STS average (Table 3: 84.07 vs. 82.25).
**Condition**: When generating synthetic CL data; Vicuna-13B was used for all SumCSE experiments.

**Evidence**: "SumCSE (uses Vicuna-13B) improves over SynCSE (uses ChatGPT) (+1.8) on STS (Abstract)."

## [POSITIVE] Scaling Dataset Size (4x)
Using all four summary compositions as positives (instead of random one) and one random summary composition as negative, creating a 4x larger dataset (1.1M examples).

**Delta**: SumCSE_large achieves 84.63 on STS vs. 84.07 for SumCSE (Table 6, row 2).
**Condition**: When computational budget allows generating larger synthetic datasets; improves STS performance.

**Evidence**: "SumCSE_large achieves the best result in this paper, significantly improving over SumCSE on STS (Table 6)."

## [POSITIVE] Applying Summary Transformation to Existing Datasets
Taking an existing CL dataset (SynCSE+) and applying the summary transformation to both positives and negatives to create a new dataset.

**Delta**: Improves SynCSE+ from 83.28 to 84.18 on STS (Table 6, row 3).
**Condition**: When augmenting existing contrastive learning datasets; works with SynCSE+ data.

**Evidence**: "Summary transformation when applied to other datasets works significantly improves the overall performance (Table 6)."

## [POSITIVE] Positive-Only Training (Eq. 1) with Summary Compositions
Training the embedding model using only positive pairs (no negatives) with the InfoNCE loss variant that uses only positives.

**Delta**: SumCSE positive-only achieves 81.73 on STS, already outperforming most unsupervised methods (Table 6, row 1).
**Condition**: When evaluating the quality of positives independently; also useful when negatives are unavailable.

**Evidence**: "Positive only performance of SumCSE already outperforms most unsupervised methods in Table 3 (Section 4.6)."

## [POSITIVE] Shorter Positives and Negatives (InfoMin Principle)
Generating shorter positives and negatives (average length ~7 words) to minimize mutual information with the anchor while retaining core meaning.

**Delta**: Shorter negatives work best with shorter positives (Table 8: STSB 88.13 with short negatives vs. 85.6 with long negatives).
**Condition**: When designing transformations for sentence similarity; follows InfoMin principle from CV.

**Evidence**: "Shorter negatives generally work better with shorter positives - models with higher STSB validation and better anisotropy (Table 8)."

## [NEGATIVE] Higher Anisotropy (Side Effect)
SumCSE embeddings exhibit higher anisotropy (0.123) compared to SimCSE (0.094) and SynCSE (0.112), meaning embeddings are more clustered in a narrow cone.

**Delta**: SumCSE underperforms SimCSE on MTEB overall (52.10 vs. 52.31) and specifically on clustering and retrieval tasks.
**Condition**: When evaluating on diverse MTEB tasks beyond STS; summary transformation is less suitable for clustering/retrieval.

**Evidence**: "Higher anisotropy indicates that the sentence embeddings are all in a close space and explains lower performance in clustering, retrieval tasks of MTEB (Section 4.9)."

## [NEUTRAL] Optimizing Transformations for STS (Task-Specific)
Selecting and tuning transformations (summary composition) specifically for the sentence similarity (STS) downstream task.

**Delta**: Gains on STS (+1.8 vs SynCSE) but lags on MTEB overall (52.10 vs. 52.31 for SimCSE).
**Condition**: When the downstream task is known (e.g., STS); may not generalize to all embedding tasks.

**Evidence**: "We optimised our transformations for sentence similarity. Hence, SumCSE shows gains on STS while lagging on some other tasks (Section 6)."
