# Instruction Tuning with Human Curriculum

**Source**: https://aclanthology.org/2024.findings-naacl.82/

## [POSITIVE] Curriculum Instruction Tuning (Interleaved Curriculum)
Organizing instruction data in a sequence that progresses from simple to complex, interleaving different subjects while globally increasing cognitive difficulty based on Bloom's Taxonomy.

**Delta**: +4.76 on TruthfulQA, +2.98 on MMLU, +2.8 on OpenbookQA, +1.28 on ARChard compared to random shuffling
**Condition**: When applied to instruction tuning of LLMs (specifically LLaMA 2 13B) with the CORGI dataset.

**Evidence**: "The findings of our study reveal that substantial improvements in performance can be achieved through the mere application of curriculum ordering to instruction data—achieving gains of +4.76 on TruthfulQA, +2.98 on MMLU, +2.8 on OpenbookQA, and +1.28 on ARChard—compared to random shuffling."

## [POSITIVE] Interleaving vs. Blocking Curriculum
Interleaving cyclically revisits each subject with increasing cognitive difficulty, while blocking stacks hierarchical blocks per subject sequentially.

**Delta**: Interleaving outperforms blocking: ΔMMLU +2.75, ΔARC +2.39, ΔPIQA +1.14, ΔOpenbookQA +3.8 compared to blocking
**Condition**: When training multi-domain knowledge with LLaMA 2 13B on the CORGI dataset.

**Evidence**: "CORGI demonstrated notable improvements when subjected to our interleaved curriculum training (ΔMMLU +0.64 → +2.75, ΔARC +0.26 → +2.39, ΔPIQA -0.65 → +1.14, ΔOpenbookQA +0.60 → +3.8) compared to naive stacking of concepts."

## [POSITIVE] Synthetic Instruction-Response Generation with Educational Curricula
Using real-world educational curricula (e.g., Cambridge IGCSE, university catalogs) and a teacher model (ChatGPT) to generate instruction-response pairs covering secondary school to graduate levels.

**Delta**: Outperforms other instruction datasets (WizardLM, Vicuna) with smaller dataset size (66K vs 125K/250K)
**Condition**: When generating instruction-tuning data for LLMs, especially when combined with curriculum ordering.

**Evidence**: "When CORGI is subjected to random training, its performance is comparable to other instruction datasets such as WizardLM and Vicuna. However, by simply optimizing the sequence of learning data, we observed a roughly 3 points improvement in the knowledge benchmark (i.e., MMLU), surpassing both WizardLM and Vicuna with a considerably smaller dataset size (66K)."

## [POSITIVE] Bloom's Taxonomy-based Question Generation
Generating 19 question templates per concept based on three lower cognitive levels of Bloom's Taxonomy: Remember, Understand, and Apply.

**Delta**: Enables structured cognitive progression; contributes to overall performance gains
**Condition**: When creating instruction data with varying cognitive difficulty for curriculum learning.

**Evidence**: "Exploiting this concept, we produce diverse data for a single concept by giving a detailed object from each cognitive level as instructions to a teacher language model during data generation."

## [POSITIVE] Contriever-based Data Filtering
Using Contriever to retrieve relevant Wikipedia passages and assess relevance to filter out low-quality instruction-response pairs.

**Delta**: Removes 40-50% of instances; yields significant benefits: ΔMMLU +1.7 (107K→66K), +1.9 (60K→37K), +1.7 (30K→15K)
**Condition**: When dealing with synthetic instruction data from teacher models to improve data quality and efficiency.

**Evidence**: "Filtering out poor-quality data points yields significant benefits across different data sizes in LLaMA 1 (e.g., ΔMMLU +1.7: 107K → filter 66K; ΔMMLU +1.9: 60K → filter 37K; ΔMMLU +1.7: 30K → filter 15K)."

## [POSITIVE] Rule-based String Matching Filtering
Removing refusal data containing specific text sequences like 'As an AI...', 'sorry', 'I cannot', etc.

**Delta**: Removes 1-2% of samples containing illegal or unhelpful text
**Condition**: When filtering synthetic instruction data to remove unhelpful refusal responses.

**Evidence**: "String-matching accounted for a significantly small percentage, removing 1∼2% of samples containing illegal or unhelpful text."

## [NEUTRAL] Semantic Deduplication of Concepts
Using cosine similarity threshold of 0.67 with sentence-transformers to remove redundant concepts extracted from curricula.

**Delta**: Reduced 5.6K fine-grained concepts from initial extraction
**Condition**: When extracting concepts from educational curricula to ensure diversity and avoid redundancy.

**Evidence**: "To achieve maximal diversity and distinction among the selected concepts, we harvested an extensive array of fine-grained concepts and subsequently eliminated any redundancies. Specifically, we employed semantic deduplication utilizing a cosine similarity threshold of 0.67 using the sentence-transformers library model all-MiniLM-L12-v2."

## [POSITIVE FOR GLOBAL; NEGATIVE FOR LOCAL] Global vs. Local Curriculum Strategies
Global curriculum maintains structure across larger batch sizes by covering all subjects in each batch; local curriculum (e.g., clustering, spiral) focuses on intra-subject or intra-concept progression.

**Delta**: Global curriculum (interleaving) outperforms random shuffling; local curricula (clustering, spiral) show diminished or negative effects
**Condition**: When training multi-domain knowledge with curriculum learning; global curriculum is effective, local curriculum may be detrimental.

**Evidence**: "Global curricula give large benefits, while local curricula can mislead. ... most local progressions or structures are destroyed when employing a larger global batch size. This results in a biased training batch."

## [POSITIVE] Interleaved Training for Stability
Interleaving subjects ensures all subjects are covered in every training batch, maintaining structure under larger batch sizes.

**Delta**: More stable than random shuffling; improves MMLU subject group scores
**Condition**: When training with multi-domain data and larger batch sizes to prevent catastrophic forgetting and biased batches.

**Evidence**: "Figure 6 shows how a global curriculum, which maintains structure under most larger batch sizes while ensuring that all subjects are covered in every training batch, successfully pushes performance above the random shuffling baseline."

## [POSITIVE] Curriculum Learning for Noisy Data Robustness
Using curriculum ordering to mitigate negative impacts of noisy synthetic data from teacher models.

**Delta**: Turns performance decreases into increases: ΔMMLU -0.31 → +2.75, ΔPIQA -0.55 → +1.14, ΔHellaSwag -1.49 → +2.18
**Condition**: When training with noisy synthetic datasets, especially when the teacher and student model performance gap narrows.

**Evidence**: "We observe that several benchmarks, which initially show decreased performance after random shuffled instruction tuning, exhibit substantial performance improvements after curriculum-based instruction tuning (ΔMMLU -0.31 → +2.75, ΔPIQA -0.55 → +1.14, ΔHellaSwag -1.49 → +2.18)."

## [POSITIVE] Using Real-World Educational Curricula as Knowledge Source
Leveraging Cambridge IGCSE curriculum and university catalogs covering 45 subjects to ensure broad knowledge coverage.

**Delta**: Enables broad knowledge coverage; contributes to outperforming baselines with smaller dataset
**Condition**: When constructing a comprehensive knowledge instruction dataset for LLMs.

**Evidence**: "These curricula cover 45 distinct subjects and provide rich metadata, including educational stage, subject, course, and syllabus, ensuring a broad spectrum of knowledge coverage as well."
