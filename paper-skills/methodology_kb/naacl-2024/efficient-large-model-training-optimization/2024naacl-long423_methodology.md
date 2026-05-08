# PELMS: Pre-training for Effective Low-Shot Multi-Document Summarization

**Source**: https://aclanthology.org/2024.naacl-long.423/

## [POSITIVE] Semantic sentence clustering for topic detection
Uses continuous semantic representations (Sentence Transformers) to cluster input sentences into topics, replacing brittle lexical overlap methods like ROUGE or entity linking.

**Delta**: outperforms baselines in ROUGE-G and BertScore across all datasets
**Condition**: Zero-shot and low-shot multi-document summarization across diverse domains (news, reviews, scientific papers)

**Evidence**: "Methods like Primera use ROUGE similarity or align sentences using entity mentions. However, these are brittle and generalize poorly, motivating the need for a more refined selection mechanism. We propose to use continuous semantic representations when performing the sentence similarity comparison."

## [POSITIVE] Entailment-aware target sentence selection
Ranks candidate sentences within each topic cluster by both distance to cluster centroid and average NLI entailment with other cluster elements, then aggregates rankings via Borda count.

**Delta**: improves faithfulness (MDSummaC) and informativeness (ROUGE, BertScore)
**Condition**: Pre-training target formation; particularly effective for maintaining input-consistent summaries in low-shot settings

**Evidence**: "To improve the consistency of our selected sentence with the rest of the cluster, we use a combination of cluster centrality and intra-cluster entailment to score the candidate sentences."

## [POSITIVE] Minimum set cover for target sentence selection
Selects one sentence per topic cluster using minimum set cover to source from as few documents as possible, improving target coherence.

**Delta**: improves coherence (DiscoScore) and faithfulness
**Condition**: Pre-training target formulation; especially beneficial for multi-document inputs with arbitrary document ordering

**Evidence**: "Considering only the c highest-ranked sentences from each topic cluster, we select target sentences, one per topic, using minimum set cover to source from as few documents as possible to improve target coherence."

## [POSITIVE] Topic-position ordering of target sentences
Orders selected target sentences by average topic position within input documents (lead topics first), with sentences from the same document maintaining original relative ordering.

**Delta**: improves coherence (DiscoScore) and informativeness
**Condition**: Pre-training target formulation; addresses positional biases from arbitrary document ordering in multi-document inputs

**Evidence**: "After selecting target sentences, we order them subject to the following constraints in this order of precedence: 1) sentences selected from the same document should maintain their original relative ordering, and 2) sentences should be ordered by average topic position – e.g., ‘lead’ topics should appear early within the target."

## [POSITIVE] Removing selected sentences from pre-training input
Instead of masking target sentences (as in GSG), the selected sentences are removed from the input to encourage abstractiveness and reduce positional cues from mask tokens.

**Delta**: improves abstractiveness (N-gram Novelty) by 19.2% overall vs 3.3% for Primera
**Condition**: Pre-training; particularly effective for generating abstractive summaries in zero-shot settings

**Evidence**: "Finally we remove the selected sentences from the pre-training input (Fig. 3-[3c])."

## [POSITIVE] MultiPT pre-training corpus
A large-scale multi-document dataset with over 3 million topic-centric document clusters from 93 million documents, covering news, product reviews, business reviews, and Wikipedia.

**Delta**: Primera trained on MultiPT improves ROUGE-G from 15.0 to 15.4 (+0.4) and MDSummaC from 18.0 to 20.7 (+2.7)
**Condition**: Pre-training; benefits both PELMS and existing models (e.g., Primera) across all domains

**Evidence**: "We find Primera benefits from pre-training on our diverse large-scale dataset (versus on NewSHead)."

## [POSITIVE] Length-controlled pre-training with variable k
Varies the number of target sentences (k) during pre-training and uses length prefixes to enable controllable summary length at inference.

**Delta**: average ROUGE-G improvement of 0.6 points (from 16.7 to 17.3)
**Condition**: Zero-shot summarization; best prefixes vary by dataset (e.g., 'long' for MultiNews, 'medium-long' for MetaTomatoes)

**Evidence**: "We briefly explore length control during pre-training, varying the k value which sets the number of target sentences and training with a corresponding a length-prefix. We achieve an average ROUGE-G improvement of 0.6 points."

## [POSITIVE] Adapter-based fine-tuning (PEFT)
Parameter-efficient fine-tuning using adapters that update only 5% of model parameters while freezing the rest.

**Delta**: PELMS with adapters outperforms all baselines with full fine-tuning at 16-shot (RG 17.0 vs 15.8 for Centrum)
**Condition**: Low-shot supervised settings; particularly effective for resource-constrained environments

**Evidence**: "PELMS’s performance is especially pronounced in adapter fine-tuning scenarios. It significantly surpasses others in ROUGE-G across all data scenarios, from 16-shot to full-shot."

## [POSITIVE] Global <doc-sep> tokens for cross-document attention
Inserts a global <doc-sep> token after each input document with full attention for improved cross-document communication.

**Delta**: outperforms baselines without this mechanism
**Condition**: All settings (zero-shot and supervised); standard for LED-based MDS models

**Evidence**: "We follow Primera in inserting a global <doc-sep> token after each input document for improved cross-document communication."

## [POSITIVE] Using LED-large as base architecture
Initializes from the 464M Longformer Encoder-Decoder (LED-large) with sparse attention for long inputs.

**Delta**: outperforms Pegasus-X architecture on faithfulness (MDSummaC 19.7 vs 17.6)
**Condition**: All settings; LED architecture provides better faithfulness than Pegasus-X

**Evidence**: "We also see our method extends well to an alternate architecture (Pegasus-X), outperforming the LED architecture model on most metrics, although faithfulness is decreased."

## [POSITIVE] NLI-based intra-cluster entailment scoring
Uses an NLI model (albert-base-vitaminc) to compute positive entailment probability between candidate sentences and the 5 most central cluster elements.

**Delta**: improves faithfulness and consistency
**Condition**: Pre-training target selection; computationally constrained to top 5 central elements per cluster

**Evidence**: "We cap the NLI calculation to the 5 most central cluster elements due to the quadratic computational complexity of the intra-cluster entailment score."

## [POSITIVE] Borda count aggregation of rankings
Aggregates distance-to-centroid and entailment rankings using an unweighted Borda count to produce a final ranking of cluster elements.

**Delta**: improves overall summary quality
**Condition**: Pre-training target selection; simple and effective combination of two ranking signals

**Evidence**: "We aggregate these two rankings using a simple unweighted Borda count (Emerson, 2013) to generate a ranking of all cluster elements."
