# Generating Mental Health Transcripts with SAPE (Spanish Adaptive Prompt Engineering)

**Source**: https://aclanthology.org/2024.naacl-long.285/

## [POSITIVE] SAPE (Spanish Adaptive Prompt Engineering) with Genetic Algorithm
An automated prompt engineering method using genetic algorithms for prompt generation and selection, designed for Spanish-language therapy transcript generation. It evolves both instruction-prompts and mutation-prompts through biological-inspired mutation mechanisms (sexual/asexual reproduction, selective breeding, environmental adaptation, taught behaviors, hyper-mutation).

**Delta**: outperforms Reflexion and CoT baselines
**Condition**: When generating Spanish therapy transcripts for mood check, change methods, and set goals interactions; evaluated by 8 psychologists on 180 transcripts.

**Evidence**: "Our results indicate that overall mental health professionals find SAPE-generated text to resemble authentic therapy transcripts more closely than texts generated with other prompt engineering techniques."

## [POSITIVE] Fitness-based Probabilistic Selection (non-greedy)
Instead of always selecting the prompt with the highest fitness (greedy), SAPE normalizes collective fitness of all prompts and selects winners via a probability function based on normalized values, allowing under-performing individuals a chance to survive.

**Delta**: avoids local maxima, improves exploration
**Condition**: When evolving prompts over generations; particularly important for open-ended tasks without a single correct answer.

**Evidence**: "To prevent convergence to a local maximum, our genetic algorithm employs a strategy that extends beyond selecting prompts solely based on their highest fitness. Instead, it normalizes the collective fitness of all the prompts. The algorithm then determines the winner through a probability function, utilizing values derived from this normalization process."

## [POSITIVE] Reinforcement Learning from Human Feedback (RLHF) for Fitness Function
Instead of using accuracy (as in closed-ended tasks), SAPE uses a reward model trained on human preferences from 15,000 paired transcript comparisons by annotators, to evaluate prompt fitness for open-ended therapy transcript generation.

**Delta**: enables optimization for open-ended tasks
**Condition**: When optimizing prompts for open-ended generative tasks where objective accuracy metrics are unavailable.

**Evidence**: "Unlike arithmetic problems with clear correct answers, therapy transcripts are more open ended and do not have a single correct way to be written. Hence, to optimize our prompts our fitness function relied on reinforcement learning with human feedback facilitated by domain experts – in this case, clinical psychologists."

## [POSITIVE] Multiple Mutation Mechanisms (5 types + hyper-mutation)
SAPE uses five biologically-inspired mutation categories: sexual reproduction, asexual reproduction, selective breeding, environmental adaptation, and taught behaviors, plus hyper-mutation for evolving mutation prompts themselves. Each has equal base probability but mutation categories with good track records get increased chance.

**Delta**: maintains diversity and improves search
**Condition**: During prompt evolution across generations; all five mechanisms used.

**Evidence**: "To achieve the balance of breadth and depth required for a healthy evolutionary search process, every mechanism of mutation has an equal base probability of being the acquired mutation, but mutation categories that have track record of yielding good fitnesses have an increased chance of being acquired on top of the base chance."

## [NEGATIVE] Sexual Reproduction Mutation
Combines genetic information from two individuals (prompts) by selecting a partner based on fitness similarity, then applying a mutation prompt to both parent prompts to generate a child prompt with characteristics from both.

**Delta**: lowest fitness among mutation types
**Condition**: When used as the mutation mechanism during prompt evolution; authors suggest future work could change or remove it.

**Evidence**: "Notably, for all three prompts, sexual reproduction emerged as the mutation prompt generating the lowest fitness."

## [NEUTRAL] Zero-Shot Chain-of-Thought (CoT) Prompting
A prompt engineering technique where the model is instructed to reason step-by-step before answering. Used as a baseline comparison method, with prompts crafted by psychologists.

**Delta**: outperformed by SAPE in cumulative rankings; no significant difference vs Reflexion
**Condition**: When generating Spanish therapy transcripts; compared against SAPE and Reflexion.

**Evidence**: "For the cumulative score rankings... subsequent post hoc analysis... identified a significant difference between the SAPE group and the CoT group (p = 1.0 × 10⁻³). However, no statistically significant difference was found between the Reflexion and CoT groups (p = 0.90)."

## [NEGATIVE] Reflexion Prompting
A prompt engineering technique where the model reflects on its own outputs to improve. Used as a baseline comparison method, with prompts crafted by psychologists.

**Delta**: outperformed by SAPE in all significant comparisons
**Condition**: When generating Spanish therapy transcripts; compared against SAPE and CoT.

**Evidence**: "For the cumulative score rankings... subsequent post hoc analysis... identified a significant difference between the SAPE group and the Reflexion group (p = 1.0 × 10⁻³)."

## [NEGATIVE] Prompt with Preamble (explicit instruction marker)
Including a preamble explicitly indicating the text is an instruction, as opposed to prompts that solely specify expected actions. Used in the 'Set goals' SAPE prompt.

**Delta**: SAPE did not achieve highest ranking for Set goals
**Condition**: When generating 'Set goals' therapy transcripts; may be related to model's handling of Spanish instructions or budget constraints preventing use of latest models.

**Evidence**: "The only therapeutic interaction prompt in which SAPE did not get the highest ranking was for the 'Set goals' prompt. A notable characteristic distinguishing this prompt is the inclusion of a preamble explicitly indicating it as an instruction, unlike the other prompts that solely specify the actions expected from the model."

## [NEUTRAL] Using text-davinci-003 (older model) due to budget constraints
The study used OpenAI's text-davinci-003 model instead of newer, more expensive models, acknowledging potential performance decrease for automatic prompt engineering.

**Delta**: potential performance decrease acknowledged
**Condition**: When running SAPE algorithm; authors note this may have affected results, especially for Spanish-language tasks.

**Evidence**: "We made a deliberate decision not to employ the latest and most expensive models from OpenAI due to budget constraints, acknowledging the potential decrease in performance of our automatic prompt engineering technique."

## [NEUTRAL] Separate LLM calls for mutation and formatting (alternative approach)
Instead of one call for both evolving and formatting a prompt, using separate calls: one to mutate the prompt and another to format it. Suggested as an alternative for handling preamble issues.

**Delta**: not implemented; suggested for future work
**Condition**: When dealing with prompt formatting issues; suggested for future work to address preamble problems.

**Evidence**: "An alternative approach could have involved separate calls to the LLM: one to mutate the prompt and another to format it. However, this approach might not always be feasible when using models behind paid APIs due to associated additional costs."
