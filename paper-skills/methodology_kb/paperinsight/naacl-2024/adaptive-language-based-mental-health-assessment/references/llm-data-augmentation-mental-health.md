---
title: LLM-Based Data Augmentation Mitigates Data Scarcity in Mental Health Assessment
confidence: MEDIUM
papers: [2024naacl-long452, 2024naacl-long285]
---

# LLM-Based Data Augmentation Mitigates Data Scarcity in Mental Health Assessment

Two NAACL 2024 papers demonstrate that LLMs can be effectively used to generate synthetic mental health data — either by rephrasing existing clinical interview transcripts or by generating therapy transcripts from scratch — to address the chronic data scarcity problem in mental health NLP. Both papers use principled approaches (guiding principles or genetic algorithm optimization) and validate outputs with human experts.

## Specific Evidence

**Principle-Guided Rephrasing for Clinical Interviews (2024naacl-long452):** GPT-3.5-turbo was used to rephrase participant answer transcripts from clinical interviews, guided by five principles: integrity, authenticity, respectfulness, consistency, and informality. Rephrasing was done in a sliding window of three rounds with context awareness and manual demonstrations. This yielded 2x training samples. The augmented data was mixed with original data for both supervised cross-entropy loss and contrastive learning (InfoNCE). Results: SEGA++ (with augmentation) gained +2.83% macro F1 on DAIC-WOZ and +1.81% on EATD over the base SEGA model. Manual quality verification found less than 1% of augmented data needed filtering. The authors note: "Removing principle-based augmentation ('w/o DA') leads to a performance drop to the level of the original SEGA, highlighting the effectiveness of synthetic data."

**Genetic Algorithm for Therapy Transcript Generation (2024naacl-long285):** SAPE (Spanish Adaptive Prompt Engineering) used a genetic algorithm with RLHF-based fitness to evolve prompts for generating Spanish therapy transcripts from scratch. The fitness function used a reward model trained on 15,000 paired transcript comparisons by annotators. Eight clinical psychologists evaluated 180 generated transcripts across three interaction types (mood check, change methods, set goals). Results: SAPE-generated text was found by mental health professionals to "resemble authentic therapy transcripts more closely than texts generated with other prompt engineering techniques" (Reflexion, Chain-of-Thought).

## Papers & Evidence
- `2024naacl-long452` (Depression Detection with SEGA): "After LLM-empowered data augmentation, SEGA++ obtains further gains of 2.83% and 1.81%, surpassing baselines by 3.95% and 4.17%" — GPT-3.5-turbo rephrasing with 5 guiding principles yields 2x training data; <1% filtered out.
- `2024naacl-long285` (SAPE for Spanish Therapy Transcripts): "Our results indicate that overall mental health professionals find SAPE-generated text to resemble authentic therapy transcripts more closely than texts generated with other prompt engineering techniques" — Genetic algorithm with RLHF fitness generates therapy transcripts validated by 8 psychologists.

## Actionable Guidance
When training data is scarce (e.g., <200 samples as in clinical interview datasets), use LLM-based data augmentation with the following recipe: (1) Use GPT-3.5-turbo (not GPT-4, for cost efficiency) to rephrase existing text; (2) Define 4-5 guiding principles (integrity, authenticity, respectfulness, consistency, informality) to constrain generation quality; (3) Use a sliding window of 3 rounds for context preservation; (4) Include manual demonstrations in the prompt; (5) Always perform manual quality verification — expect <1% rejection rate with proper principles; (6) Mix synthetic and real data for training; (7) Apply contrastive learning (InfoNCE) on the augmented data for additional gains.

**Condition**: When training data for mental health assessment is limited (e.g., <500 samples) and LLM-based text generation is feasible and ethically appropriate.
**Confidence**: MEDIUM — Only two papers support this insight, and they use different augmentation strategies (rephrasing vs. generation from scratch). The principle-guided approach (Paper 452) has stronger quantitative evidence (+2.83% macro F1) than the genetic algorithm approach (Paper 285), which relies on qualitative psychologist evaluation.
