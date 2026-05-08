# The Colorful Future of LLMs: Evaluating and Improving LLMs as Emotional Supporters for Queer Youth

**Source**: https://aclanthology.org/2024.naacl-long.113/

## [POSITIVE] Novel Ten-Question Rating Scale
A questionnaire developed with psychologists and psychiatrists to evaluate written responses to queer youth, covering ten traits like inclusiveness, sensitivity, emotional validation, mental status, personal circumstances, support networks, accuracy, safety, authenticity, and completeness.

**Delta**: Enables structured evaluation; human evaluators showed high inter-annotator agreement (e.g., 99% pairwise agreement for Q1, 78% overall).
**Condition**: Used for evaluating LLM and human responses to Reddit posts from queer youth.

**Evidence**: "We develop a novel ten-question scale that is inspired by psychological standards and expert input."

## [POSITIVE] Guided Supporter Prompt
A prompt that provides a list of dos and don'ts corresponding to the questionnaire traits, guiding the LLM to improve emotional support responses.

**Delta**: Significantly improves responses across most attributes; e.g., ChatGPT+Guided scores 0.95 (Q1), 0.94 (Q2), 0.93 (Q3) vs. ChatGPT's 0.93, 0.86, 0.83 respectively.
**Condition**: Applied to LLMs (e.g., ChatGPT, GPT3.5, GPT4) when generating responses to queer youth posts.

**Evidence**: "Our research contributes to enhancing LLMs performance through the ‘Guided Supporter’ prompt, significantly improving responses across most attributes."

## [POSITIVE] Automatic Evaluation with GPT-4
Using GPT-4 to automatically evaluate LLM responses based on the ten-question scale, enabling broader model comparison.

**Delta**: Agrees with human rankings over 80% of the time for pairwise comparisons; Spearman correlation significant for most questions.
**Condition**: Used to evaluate 11 combinations of LLMs and prompts when human annotation is too costly or time-consuming.

**Evidence**: "For almost all questions, there is an agreement between human and automatic rankings over 80% of the time."

## [POSITIVE] LGBTeen Dataset Construction
A dataset of 1,000 Reddit posts from r/LGBTeens, with human comments and LLM responses (11,320 responses from 15 model-prompt combinations), annotated with the ten-question scale.

**Delta**: Enables comprehensive quantitative analysis; provides over 5,000 human labels.
**Condition**: Used for evaluating and comparing LLM and human emotional support responses.

**Evidence**: "We constructed the LGBTeen dataset, comprising hundreds of posts, thousands of LLMs’ responses, and human annotations."

## [POSITIVE] Qualitative Case Study Analysis
Reviewing ChatGPT interactions with queer-related queries, including specific cultural and personal contexts, to identify strengths and weaknesses.

**Delta**: Identified key issues: generic responses, cultural ignorance, lack of personalization, and harmful advice (e.g., ignoring death penalty in Afghanistan).
**Condition**: Applied to initial exploration of LLM behavior before quantitative analysis.

**Evidence**: "ChatGPT does not mention the death penalty for LGBTQ+ which exists in this country."

## [POSITIVE] Comparison to Human Reddit Comments
Comparing LLM responses to the most upvoted human comments on the same Reddit posts, using the ten-question scale.

**Delta**: LLMs outscore humans on most dimensions (e.g., ChatGPT: 0.93 Q1 vs. Reddit: 0.98; but LLMs score lower on authenticity Q9: 0.61 vs. 0.97).
**Condition**: Used to benchmark LLM performance against anonymous human support.

**Evidence**: "LLM responses score better than human comments across most dimensions of our questionnaire."

## [NEGATIVE] Diversity Analysis via Embedding Cosine Similarity
Computing cosine similarity of RoBERTa embeddings to measure diversity of LLM responses vs. human comments and posts.

**Delta**: LLM responses are much more similar to each other (higher cosine similarity) than human comments or posts, indicating lack of diversity.
**Condition**: Applied to all responses in the dataset to quantify genericness.

**Evidence**: "LLMs responses are much similar to each other and lack diversity (lower scores are better) compared to the posts and human comments."

## [NEUTRAL] Prompt Engineering: Supporter, Redditor, Therapist Prompts
Testing different system prompts (empathetic AI, Redditor, therapist) to see their effect on response quality.

**Delta**: Supporter and Therapist prompts improve performance but less than Guided prompt; Redditor prompt shows no significant effect.
**Condition**: Used in automatic evaluation of GPT3.5 and GPT4.

**Evidence**: "For GPT3.5, the ‘Supporter’ prompt enhances performance, and this improvement is elevated by the ‘Guided’ prompt... The ‘Redditor’ prompt shows no significant effect."

## [POSITIVE] Blueprint for AI Queer Supporter
A proposed system with four components: aligned LLM, queer-dedicated textual collection, Identification component, and Assertion component, to improve reliability, empathy, and personalization.

**Delta**: Conceptual framework; no quantitative delta, but addresses identified weaknesses (reliability, empathy, personalization).
**Condition**: Proposed for future development; not implemented in this study.

**Evidence**: "We propose a blueprint of an LLM-supporter that actively (but sensitively) seeks user context to provide personalized, empathetic, and reliable responses."

## [POSITIVE] Human Evaluation with Queer Annotators
Using three queer-identifying evaluators with academic degrees to annotate responses using the ten-question scale, after training.

**Delta**: High inter-annotator agreement (e.g., 99% for Q1, 78% overall); provides reliable ground truth.
**Condition**: Used for the main human evaluation of UI LLMs and Reddit comments.

**Evidence**: "Our evaluators, comprising two females and one male, all identifying as queers and holding academic degrees... show high IAA."

## [NEGATIVE] Automatic Evaluation with GPT-3.5
Using GPT-3.5 to evaluate responses, compared to human majority vote.

**Delta**: Lower agreement with humans than GPT-4; e.g., Q9 authenticity: 29% agreement vs. 40% for GPT-4.
**Condition**: Used as a cheaper alternative to human evaluation, but found insufficient for nuanced emotional tasks.

**Evidence**: "GPTs struggle in Q5, Q6, Q7, and Q9... LLMs currently cannot replace human evaluators."

## [NEUTRAL] Multi-Language Testing of ChatGPT
Testing ChatGPT responses in Hebrew, Russian, and Arabic to see if cultural biases affect output.

**Delta**: No notable difference from English responses; all remained inclusive and supportive.
**Condition**: Applied to case studies from qualitative analysis translated into conservative languages.

**Evidence**: "Contrary to our expectations, we observed no notable difference from the English responses. In all languages, the responses remained inclusive, supportive, and positive."
