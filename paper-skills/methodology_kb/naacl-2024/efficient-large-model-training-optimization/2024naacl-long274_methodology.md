# Anisotropy is Not Inherent to Transformers

**Source**: https://aclanthology.org/2024.naacl-long.274/

## [POSITIVE] Untied embedding and unembedding matrices
Using separate weight matrices for the embedding (input) and unembedding (output) layers, rather than tying them together.

**Delta**: Pythia models with untied weights achieve isotropy estimates of 0.73-0.82 vs. previous best of 0.52
**Condition**: Large models (>=410M parameters); untied weights increase isotropy for large models but may hurt small models

**Evidence**: "We suspect having untied embedding and unembedding matrices leads to higher isotropy... Our results are also in line with previous work which showed that tying weights in small models... improves performance even though the Pythia-70M and Pythia-160M models have the worst isotropy across all models."

## [POSITIVE] Final Layer Norm optimization for isotropy
The final Layer Norm layer is optimized during training such that its bias vector b has low norm relative to the gain vector g, minimizing anisotropy introduced by the Layer Norm.

**Delta**: Isotropic Pythia models have smallest ||b||_2 and smallest ||b||_2/||g||_2 ratios
**Condition**: Large Pythia models (>=410M); the optimization happens during training and correlates with isotropy

**Evidence**: "We see that all isotropic Pythia models minimize the norm of b generally and with respect to g, and that the anisotropic Pythia models fail to do either... the final Layer Norm for said models... actually increased isotropy compared to the previous layer."

## [POSITIVE] Large model scale (>=410M parameters)
Using larger model sizes (410M, 1.4B, 6.9B, 12B parameters) which exhibit isotropic embedding spaces, unlike smaller models (70M, 160M).

**Delta**: Large Pythia models have Inter-Sim of 0.14 (81.6° angle) and isotropy estimates 0.73-0.82
**Condition**: Pythia models with >=410M parameters; smaller models (70M, 160M) remain anisotropic

**Evidence**: "We identify the most globally isotropic models to date, the Pythia models with at least 410M parameters... The 70M and 160M Pythia models have relatively low Intra-Sim... The 1.4B and 6.9B models, contrastingly, have high Inter-Sim... followed by a sharp drop in the last transformer layer and Layer Norm."

## [POSITIVE] Reformulated average cosine similarity for efficiency
Reformulating average cosine similarity computation from O(n²) to O(n) using linearity of inner product, enabling analysis on 425M sentences.

**Delta**: Reduced computational complexity from O(n²) to O(n)
**Condition**: When computing average cosine similarity on large token sets (millions of tokens)

**Evidence**: "Thus, we can compute U using O(n) operations rather than O(n²). This allows us to compute U efficiently for large sets."

## [POSITIVE] Analysis on full training dataset (The Pile) instead of random samples
Using 425M sentences from the actual training dataset (The Pile) rather than randomly sampled text sources, including rare words with frequency <5.

**Delta**: No significant differences found across text sources
**Condition**: When evaluating isotropy across different domains; rare words are included

**Evidence**: "Contrary to previous work, which use token frequencies in the 1000s, we perform cosine analysis on 425M sentences from the actual training dataset, The Pile... We find no significant differences and thus only report the results on all text sources combined."

## [POSITIVE] Monitoring isotropy as early stopping criterion
Using a sudden and steady transition from isotropy to anisotropy as an indicator that downstream task performance is degrading.

**Delta**: Correlation of 0.99 (Pearson) between isotropy and downstream performance for Pythia-70M
**Condition**: When a model shows a clear transition from isotropic to anisotropic state during training

**Evidence**: "We see a clear correlation between the isotropy of the Pythia 70M model and its downstream task performance across all correlation metrics and across all tasks... a sudden and steady transition from isotropy to anisotropy does imply that downstream task performance is also degrading. This suggests that monitoring isotropy may be useful as an early stopping criterion."

## [POSITIVE] Using multiple isotropy metrics (cosine, partition function, Layer Norm analysis)
Employing three different isotropy measures (average cosine similarity, partition function I(W), and final Layer Norm parameter analysis) to ensure robust conclusions.

**Delta**: All metrics agree on isotropic status of large Pythia models
**Condition**: General evaluation of isotropy; provides cross-validation

**Evidence**: "Using multiple metrics allows us to present a more confident conclusion when all of our isotropy measures agree."

## [NEUTRAL] Autoregressive language modeling with cross-entropy loss
Training using standard autoregressive language modeling objective with cross-entropy loss, which was previously assumed to inherently cause anisotropy.

**Delta**: Large Pythia models achieve isotropy despite using this standard training objective
**Condition**: Standard training setup; does not inherently cause anisotropy as previously assumed

**Evidence**: "These models are trained using cross-entropy loss, using autoregressive language modeling... The identification of a set of isotropic Transformer models calls previous assumptions into question."

## [NEUTRAL] Flash Attention and rotary position embeddings
Using Flash Attention for efficient attention computation and rotary position embeddings (RoPE) for positional encoding.

**Delta**: No direct quantitative effect on isotropy reported
**Condition**: Architecture choices; not identified as primary cause of isotropy

**Evidence**: "Pythia models use Flash Attention... rotary position embeddings... parallelized attention and feed-forward... We see GPT-NeoX has the next best isotropy estimates, but surprisingly, due to its similar architecture and training, is clearly worse than large Pythia models."

## [NEUTRAL] Parallelized attention and feed-forward layers
Computing attention and feed-forward network in parallel rather than sequentially, as used in GPT-NeoX architecture.

**Delta**: No direct quantitative effect on isotropy reported
**Condition**: Architecture choice; not sufficient alone to produce isotropy

**Evidence**: "Pythia models use... parallelized attention and feed-forward... GPT-NeoX-20B... uses parallelized attention and feedforward and Flash Attention. While it has the next best results after the Pythia models, those results are not in line with the Pythia models."
