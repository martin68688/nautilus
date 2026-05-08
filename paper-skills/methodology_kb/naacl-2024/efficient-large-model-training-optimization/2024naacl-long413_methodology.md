# DUQGen: Effective Unsupervised Domain Adaptation of Neural Rankers by Diversifying Synthetic Query Generation

**Source**: https://aclanthology.org/2024.naacl-long.413/

## [POSITIVE] Document Clustering for Domain Representation
Representing the target document collection as clusters using K-Means on Contriever embeddings to capture topical diversity and domain structure.

**Delta**: Outperforms no-clustering baseline; optimal K=1000 across all datasets
**Condition**: Applied to target document collection before query generation; K=1000 optimal via Elbow method.

**Evidence**: "Table 5 confirms that performances without clustering (K=1) often fall below zero-shot in both datasets, especially NQ exhibiting the poorest performances."

## [POSITIVE] Probabilistic Document Sampling from Clusters
Sampling documents from each cluster with probability proportional to similarity to cluster centroid, using softmax with temperature T=1.

**Delta**: Enables representative sampling; contributes to overall 4% average relative improvement
**Condition**: Used within each cluster to select documents for query generation; temperature T=1.

**Evidence**: "We define an exponential value e^di as the representative of how close Di is to its cluster centroid. Therefore, Pr(Di|clusterk) becomes the normalized softmax."

## [POSITIVE] Diversified Document Selection via MMR
Applying Maximal Marginal Relevance (MMR) on pooled samples from multiple sampling iterations to select diverse documents per cluster.

**Delta**: Improves selection robustness; contributes to overall improvements
**Condition**: Applied after iterative sampling (m=5); lambda=1.0.

**Evidence**: "To improve selection robustness in the sampling process, we apply a diversity measure, namely Maximal Marginal Relevance (MMR)."

## [POSITIVE] In-Domain Few-Shot Prompting for Query Generation
Using 3 in-domain human-generated query-document examples in the prompt to Llama2-7B-Chat, instead of out-of-domain MS-MARCO examples.

**Delta**: Notable improvements over ms-marco prompt on Llama2-7B
**Condition**: Applied during synthetic query generation; 3-shot prompting optimal.

**Evidence**: "Our second main contribution involves using in-domain 3-shot prompts to generate queries over the ms-marco prompt, showcasing notable improvements on LLAMA-2 7B model."

## [POSITIVE] Small-Scale Synthetic Training Data (1k-5k examples)
Using only 1,000 to 5,000 synthetic query-document pairs for fine-tuning, instead of hundreds of thousands or millions.

**Delta**: Outperforms Promptagator++ which uses 1M pairs; 1k examples often better than 5k on 13/18 datasets
**Condition**: N=1000 for ColBERT; N=1000 or 5000 for MonoT5-3B; effective across all BEIR datasets.

**Evidence**: "DUQGen surpasses Promptagator++ in performance, utilizing merely 1000 LLM calls and fine-tuning with only 1000 training pairs, in contrast to Promptagator++'s requirement of generating 8 million queries."

## [POSITIVE] Llama2-7B-Chat as Query Generator
Using Llama2-7B-Chat with greedy decoding (temperature 0.0) for synthetic query generation, compared to larger models like GPT-3.5 or BLOOM.

**Delta**: Outperforms GPT-3.5-turbo and BLOOM variants on dev datasets
**Condition**: Used for query generation; greedy decoding; in-domain 3-shot prompts.

**Evidence**: "LLAMA-2 7B with 3-shot in-domain prompts exhibits higher improvements on both dev datasets, surpassing gpt-3.5-turbo."

## [POSITIVE] Hard Negative Mining with BM25/ColBERT/Contriever
Using first-stage retrievers to get top-x documents for each query, then selecting bottom-numneg documents as hard negatives.

**Delta**: Standard practice; contributes to effective fine-tuning
**Condition**: x=100 hits; numneg=4 negatives per positive pair.

**Evidence**: "We parse the synthetic queries to any first-stage retrievers, such as BM25, ColBERT, or Contriever, to get top-x documents. Then we pick the bottom-numneg documents."

## [POSITIVE] Sequential Full Fine-Tuning on MS-MARCO Pretrained Model
Starting from a ranker already pretrained on MS-MARCO, then fully fine-tuning on synthetic data with same hyperparameters.

**Delta**: Consistently outperforms zero-shot baselines across all BEIR datasets
**Condition**: Applied to MonoT5-3B and ColBERT; same hyperparameters as MS-MARCO pretraining.

**Evidence**: "We leverage the task pre-trained model (on MS-MARCO), and sequentially fully fine-tune with our own generated synthetic data."

## [POSITIVE] Filtering Short Documents (<300 characters)
Discarding documents with character length less than 300 to remove noisy documents before clustering.

**Delta**: Reduces noise; contributes to overall effectiveness
**Condition**: Applied during preprocessing; threshold can vary across datasets.

**Evidence**: "We start with the full collection of documents and apply a preprocessing step, where we discard short span documents, filtering out noisy documents."

## [NEUTRAL] Using Contriever for Document Embeddings
Encoding documents with Contriever (unsupervised dense retriever) to obtain embeddings for clustering.

**Delta**: Not compared against other embedding methods
**Condition**: Used for clustering; alternative embeddings not evaluated.

**Evidence**: "We use a SOTA text encoder, viz. Contriever, to encode each of the documents."

## [POSITIVE] K-Means with Elbow Method for Optimal K
Using the Elbow method on Sum of Squared Error to determine optimal number of clusters K=1000.

**Delta**: Optimal K=1000 consistently across all datasets
**Condition**: Applied to determine K for clustering; K=1000 optimal.

**Evidence**: "The optimal K consistently aligns at a fixed point of 1000 across all evaluation datasets, irrespective of variations in corpus size, domain properties, or domain-divergence from MS-MARCO."

## [POSITIVE] Simple InPars-Style Prompt Template
Using a simple prompt template (similar to InPars) for query generation, found to consistently yield superior retrieval performance.

**Delta**: Consistently yields superior retrieval performance across datasets
**Condition**: Used for query generation prompt; 3-shot in-domain examples.

**Evidence**: "Through tuning different prompt templates, we discovered that a simple InPars-style template... consistently yields superior retrieval performance across datasets."
