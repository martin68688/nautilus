# Human-AI Interaction in the Age of LLMs

**Source**: https://aclanthology.org/2024.naacl-tutorials.5/

## [POSITIVE] Mixed-Initiative Interaction
A flexible interaction strategy where each agent (human or AI) contributes what it is best suited at the most appropriate time, allowing both to take leading roles interchangeably.

**Delta**: outperforms baseline
**Condition**: When designing human-AI collaboration systems where complementary strengths are needed.

**Evidence**: "a flexible interaction strategy in which each agent contributes what it is best suited at the most appropriate time"

## [POSITIVE] Chain-of-Thought Prompting
A prompting technique that demonstrates desired actions step-by-step to elicit reasoning in large language models.

**Delta**: outperforms baseline
**Condition**: When performing complex reasoning tasks with LLMs.

**Evidence**: "chain of thought prompting (Wei et al., 2022) to demonstrate desired actions"

## [POSITIVE] Few-Shot Learning
A technique to tailor generation by providing a small number of examples in the prompt.

**Delta**: outperforms baseline
**Condition**: When adapting LLMs to specific tasks with limited labeled data.

**Evidence**: "few-shot learning techniques to tailor the generation"

## [POSITIVE] Human-in-the-Loop Planning
Incorporating human feedback and oversight into the LLM workflow through structured planning, conditional generation, and memory mechanisms.

**Delta**: outperforms baseline
**Condition**: When tasks require transparency, collaboration, or end-user feedback that pure prompting cannot easily leverage.

**Evidence**: "planning and human-in-the-loop (Zhang et al., 2023a) can help boost the workflow via techniques like structured planning, conditional generation (Hsu et al., 2023), and memory mechanisms (Park et al., 2023), for more transparent and collaborative human-LLM collaboration"

## [NEGATIVE] Purely Relying on Prompting
Using only standard prompt engineering without planning, human-in-the-loop, or structured techniques to generate outcomes from LLMs.

**Delta**: requires substantial expertise and time
**Condition**: When tasks are complex or require iterative refinement and user feedback.

**Evidence**: "Purely relying on prompting requires substantial expertise and time for design and implementation, and makes it difficult to leverage end-user feedback"

## [POSITIVE] Complementary Performance Evaluation
Evaluating human-AI interaction success based on achieving better outcomes than either could accomplish alone by leveraging strengths of both.

**Delta**: outperforms baseline
**Condition**: When assessing the effectiveness of human-AI collaborative systems.

**Evidence**: "To achieve better outcomes than either could accomplish alone, by leveraging the strengths of both AI and humans"

## [POSITIVE] User-Centered Evaluation
Evaluation of human-LLM interaction using quantitative measures and user-centered metrics including usability, satisfaction, engagement, and long-term effects.

**Delta**: outperforms baseline
**Condition**: When comprehensively assessing human-LLM interaction beyond task accuracy.

**Evidence**: "evaluation of human-LLM interaction (Lee et al., 2022b), ranging from quantitative measures to user-centered evaluation... not only cover task-level performances, but also interaction dimensions such as usability, satisfaction, and engagement, as well as long-term effects on users"

## [POSITIVE] Human Initiation as Fallback Option
Allowing humans to take initiative as a fallback when the model does not behave as expected.

**Delta**: outperforms baseline
**Condition**: When model behavior is unreliable or unpredictable, requiring human oversight.

**Evidence**: "human initiations may be used as not only a driving force on achieving human goals (Oh et al., 2018), but also a fallback option when the model does not behave as expected (Lee et al., 2022a)"

## [POSITIVE] Model Initiation Impact on Perceived Usefulness
The AI system taking the lead in interactions, which affects how useful users perceive the model to be.

**Delta**: outperforms baseline
**Condition**: When designing conversational or collaborative AI systems where system proactivity is needed.

**Evidence**: "discuss how model initiations impact the perceived model usefulness (Avula et al., 2022; Santy et al., 2019)"

## [NEGATIVE] Introducing Human Insights (Cognitive Load Risk)
Operationalizing domain knowledge and skills from human-human interaction into human-AI workflows, which may trigger cognitive load for users.

**Delta**: triggers cognitive load
**Condition**: When integrating human expertise into AI workflows without managing user cognitive burden.

**Evidence**: "the introduction of human insights is very likely to trigger cognitive load for users"
