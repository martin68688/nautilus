# Neurocache: Efficient Vector Retrieval for Long-range Language Modeling

**Source**: https://aclanthology.org/2024.naacl-long.50/

## [POSITIVE] Compressed State Storage
Storing compressed hidden states (projected to lower dimension d) instead of full key-value pairs to reduce cache size.

**Delta**: Reduces cache entry size from 2*a*f (Memorizing Transformer) or e (Unlimiformer) to d, where d=256 vs h=1024 (4x compression).
**Condition**: Used in all Neurocache configurations; compression factor of 4 (h=1024 to d=256) in pre-training experiments.

**Evidence**: "Neurocache improves upon previous methods by (1) storing compressed states, which reduces cache size"

## [POSITIVE] Single Retrieval Operation Per Token
Performing only one kNN cache query per token, unlike prior methods that query per attention head or per layer.

**Delta**: 1 query/token vs a queries/token (Memorizing Transformer) or l*a queries/token (Unlimiformer). Inference speed advantage shown in Figure 1.
**Condition**: Applies to all Neurocache models; key advantage over Memorizing Transformers and Unlimiformer.

**Evidence**: "Neurocache improves upon previous methods by (2) performing a single retrieval operation per token which increases inference speed"

## [POSITIVE] Contextual Retrieval Window
When retrieving top-k cached states, also including neighboring states (within window w) around each retrieved state to provide richer context.

**Delta**: Perplexity improved from 14.117 (w=1) to 13.720 (w=2) on PG-19 16K; from 14.118 to 13.578 on PG-19 64K.
**Condition**: Optimal window size w=2 determined via hyperparameter search; used in all final experiments.

**Evidence**: "extending the retrieval window to neighboring states, which improves both language modeling and downstream task accuracy"

## [POSITIVE] Extended Cache-Attention (Context Size c)
For each token, cache-attention includes not only its own retrieved states but also the retrieved states of preceding c-1 tokens.

**Delta**: Perplexity improved from 13.720 (c=1) to 13.704 (c=2) on PG-19 16K; from 13.578 to 13.564 on PG-19 64K.
**Condition**: Optimal context size c=2 determined via hyperparameter search; used in all final experiments.

**Evidence**: "We enhance the contextual awareness of each token during the cache-attention operation by granting access to the retrievals of preceding tokens."

## [POSITIVE] FIFO Cache Update Strategy
Cache is updated with compressed states after retrieval, discarding oldest n states to maintain fixed size m (First-In-First-Out).

**Delta**: Enables scaling from 16K training cache to 128K evaluation cache with improved perplexity.
**Condition**: Used in all Neurocache experiments; cache size m=16K during training, up to 128K during evaluation.

**Evidence**: "The cache Ccache is updated with the compressed states C, maintaining a fixed size of m entries. This is achieved by discarding the oldest n states, adhering to a First-In-First-Out strategy."

## [POSITIVE] Mid-Layer State Compression (Layer r)
Compressing hidden states from the r-th layer (r = 3*n_layers/4) rather than from the final layer, using a learned projection matrix.

**Delta**: Outperforms Memorizing Transformer baseline (13.352 vs 14.818 perplexity on PG-19 128K).
**Condition**: Used in all Neurocache configurations; r=9 for 12-layer models.

**Evidence**: "In Neurocache, we set the augmented layer threshold r at 3 * nlayers/4, leading to the compression of outputs from the 9th layer of a 12-layer model."

## [POSITIVE] LoRA Adaptation for Pre-trained Models
Integrating Low-Rank Adapters (LoRA) into feed-forward networks of cache-augmented layers, freezing original parameters and training only new weights.

**Delta**: Reduced perplexity from 7.359 to 7.078 (Llama2-7B on PG-19 128K); from 9.075 to 8.308 (Llama2-7B on LongPile 128K).
**Condition**: Applied to pre-trained models (OPT-1.3B, Llama2-7B, Mistral-7B); rank r=16, scale α=32, bias off.

**Evidence**: "We integrate Low-Rank Adapters (LoRA) into the feed-forward networks of the cache augmented layers. LoRA, introducing a minimal number of parameters, plays a key role in adapting the models to cache attention without compromising their original strengths."

## [POSITIVE] Weight Initialization from Pre-trained Self-Attention
Initializing cache-attention weight matrices (Wk, Wv, Wq, Wo) by duplicating weights from corresponding self-attention layers of pre-trained models.

**Delta**: Enables effective adaptation without training from scratch; achieves strong downstream performance.
**Condition**: Used when adapting pre-trained models (Llama2-7B, Mistral-7B, OPT-1.3B).

**Evidence**: "the adaptation involves initializing cache-attention weight matrices (Wk[j], Wv[j], Wq[j], Wo[j]) by duplicating weights from the corresponding self-attention layers of the pre-trained models."

## [POSITIVE] kNN with L2 Distance
Using L2-distance (Euclidean distance) to measure similarity between compressed query states and cached states for retrieval.

**Delta**: Enables effective retrieval; outperforms Memorizing Transformer across all cache sizes.
**Condition**: Used in all Neurocache configurations; k=64 in final experiments.

**Evidence**: "For each compressed state c ∈ R^d within C, we identify the top-k most similar states Cret ∈ R^(k×d) from the cache Ccache based on the L2-distance"

## [POSITIVE] Cache Size Scaling to 128K Tokens
Training with 16K cache size and evaluating with up to 128K cache size, demonstrating generalization to larger storage.

**Delta**: Perplexity consistently decreases when scaling from 16K to 128K cache (e.g., 13.511→13.352 for Neurocache on PG-19).
**Condition**: Training cache size=16K, evaluation cache size up to 128K; applies to both pre-trained and from-scratch models.

**Evidence**: "Neurocache enhances the maximum context length of models like Llama2-7B and Mistral-7B to 128K tokens."

## [POSITIVE] Segment-wise Document Processing
Dividing long documents into smaller segments of n tokens (n=1024 or 2048) and processing them sequentially through the transformer decoder.

**Delta**: Enables processing of arbitrarily long documents; outperforms truncation baselines significantly.
**Condition**: Used in all Neurocache experiments; segment size n=1024 for pre-training, n=2048 for adaptation.

**Evidence**: "The process begins by segmenting the long text sequences into smaller segments, each containing n tokens, fitting the model's attention window size."

## [POSITIVE] Cache-Augmented Layers with Residual Connection
Adding cache attention output to self-attention output via residual connection before feeding to feed-forward network.

**Delta**: Enables integration of retrieved information without disrupting original model capabilities.
**Condition**: Applied to layers j > r (top 3 layers in 12-layer model).

**Evidence**: "The output of cache attention is processed by an output matrix Wo[j] before being combined with self-attention outputs through a residual connection."
