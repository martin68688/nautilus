# Depression Detection in Clinical Interviews with LLM-Empowered Structural Element Graph

**Source**: https://aclanthology.org/2024.naacl-long.452/

## [POSITIVE] Structural Element Graph (SEGA)
Transforms the clinical interview into a directed acyclic graph (DAG) that models all four elements (transcript, audio, video, question) based on human expertise. Within each round, the transcript is the central node, audio/video are auxiliary nodes with a proxy node for noise suppression, and the question is a supplementary node. Across rounds, central nodes are linked for temporal propagation, and a summary node aggregates information for prediction.

**Delta**: Outperforms best baseline by 1.12% and 2.36% macro F1 on DAIC-WOZ and EATD respectively
**Condition**: When modeling clinical interviews with multiple modalities (text, audio, video, questions) under limited data.

**Evidence**: "SEGA achieves state-of-the-art performance, outperforming the best baseline by 1.12% and 2.36% in terms of macro F1-scores on two datasets, respectively."

## [POSITIVE] Principle-Guided Data Augmentation with LLMs
Uses GPT-3.5-turbo to rephrase participant answer transcripts guided by five principles: integrity, authenticity, respectfulness, consistency, and informality. Rephrasing is done in a sliding window of three rounds with context awareness and manual demonstrations. Only the first rephrased result is used, yielding 2x training samples.

**Delta**: SEGA++ (with augmentation) gains +2.83% and +1.81% macro F1 over SEGA on DAIC-WOZ and EATD respectively
**Condition**: When training data is scarce (e.g., ~100 training samples). The principles ensure semantic preservation and quality.

**Evidence**: "After LLM-empowered data augmentation, SEGA++ obtains further gains of 2.83% and 1.81%, surpassing baselines by 3.95% and 4.17%."

## [POSITIVE] Graph Contrastive Learning (InfoNCE)
Self-supervised contrastive learning with InfoNCE loss applied on the summary node representations. Positive samples are those with the same label (depressed/control), negative samples are those with opposite labels. Combined with cross-entropy loss as the final training objective.

**Delta**: Removing contrastive learning ('w/o CL') results in a performance decline compared to SEGA++
**Condition**: When augmented synthetic data is available; encourages distinct representations for depressed vs. control participants.

**Evidence**: "The absence of contrastive learning ('w/o CL') results in a performance decline."

## [POSITIVE] Proxy Node for Noise Suppression
Within each question-answer round, a proxy node is inserted between audio/video auxiliary nodes and the central transcript node. The proxy node first distills evidence from audio/video, transforms it into the text vector space, and then the transcript node selectively absorbs useful information. The proxy node is initialized with the transcript representation.

**Delta**: Removing proxy nodes ('w/o Proxy') causes evident performance drop
**Condition**: When audio and video modalities contain background noise that could interfere with multimodal interaction.

**Evidence**: "The evident performance drop in these structural variants underscores the effectiveness of SEGA based on human experience."

## [POSITIVE] Directed Acyclic Graph (DAG) Structure with Unidirectional Edges
Information flows unidirectionally from auxiliary nodes (audio, video, question) to the central transcript node within each round, and from central nodes to the summary node across rounds. Self-loop edges preserve each node's own information.

**Delta**: Bidirectional edges ('Bi Edge') cause evident performance drop compared to unidirectional
**Condition**: When modeling structured multi-round clinical interviews with predefined information flow based on human expertise.

**Evidence**: "The evident performance drop in these structural variants underscores the effectiveness of SEGA based on human experience."

## [POSITIVE] Element Feature Extraction with Temporal Alignment
For each question-answer round, text utterances are embedded via pre-trained word embeddings (GloVe or BERT) and averaged. Audio and video frames are averaged over the duration of the utterance based on transcript timestamps, ensuring strict alignment at the round granularity.

**Delta**: Enables comprehensive modeling of all four elements
**Condition**: When multimodal data (text, audio, video) is available with timestamps for alignment.

**Evidence**: "SEGA comprehensively captures all key elements - interview questions, answer transcripts, audios, and videos - within a simple yet effective directed acyclic graph."

## [POSITIVE] Graph Attention Network (GAT) for Node Update
Uses multi-head graph attention to iteratively update node representations by aggregating neighborhood information. The summary node representation after L=2 GAT layers is used as the final interview feature for depression prediction via a feed-forward layer.

**Delta**: Outperforms baselines including GPT-3.5/4
**Condition**: When modeling graph-structured interview data with multiple node types and predefined edges.

**Evidence**: "SEGA significantly surpasses existing baseline methods and powerful LLMs like GPT-4 for depression detection in clinical interviews."

## [NEUTRAL] Random Frame-Swapping for Audio/Video Augmentation
For audio and video modalities (which cannot be synthesized by LLMs), augmentation is done by randomly swapping frames from the same interview (a_j = a_i, v_j = v_i) to capture feasible variance.

**Delta**: Not explicitly ablated; part of the full SEGA++ pipeline
**Condition**: When only text can be augmented via LLMs; audio/video augmentation is a fallback method.

**Evidence**: "For the accompanying audios and videos, due to the untouchability of raw recordings and the lack of LLM-level synthesis tools, we adopt a simple random frame-swapping method."

## [POSITIVE] Manual Quality Verification of Augmented Data
After LLM-based rephrasing, all synthetic samples are manually inspected to verify semantic consistency, tone/sentiment preservation, absence of offensive language, and no alteration of personal details. Less than 1% of augmented data is filtered out and regenerated.

**Delta**: Only very few (less than 1%) of the augmented data is filtered out and regenerated
**Condition**: When using LLM-generated data in sensitive domains like mental healthcare to ensure quality and safety.

**Evidence**: "Ultimately, steered by the guiding principles, only very few (less than 1%) of the augmented data is filtered out and regenerated."

## [POSITIVE] Mixing Real and Synthetic Data for Training
The synthetic data from LLM augmentation is mixed with the original corpus to form the augmented training set. This is used for both supervised cross-entropy loss and contrastive learning.

**Delta**: Removing principle-based augmentation ('w/o DA') leads to a performance drop to the level of the original SEGA
**Condition**: When original training data is limited; synthetic data helps alleviate data scarcity.

**Evidence**: "Removing principle-based augmentation ('w/o DA') leads to a performance drop to the level of the original SEGA, highlighting the effectiveness of synthetic data."
