# Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation

**Source**: https://aclanthology.org/2024.naacl-long.122/

## [POSITIVE] Random Vocabulary Sampling
Randomly sampling tokens from the model vocabulary as separators, without any language model or context.

**Delta**: +10% relative improvement over human-curated separators; less than 1% difference from self-optimisation methods
**Condition**: Applicable to prompt-style text classification tasks across various language models (both pretrained and instruction-tuned).

**Evidence**: "Table 6 shows that the Random Vocabulary method yields, on average, a 10% relative improvement across nine benchmark datasets compared to human-curated separators. Random Vocabulary shows only marginal differences (less than 1% difference) compared to self-optimisation style methods."

## [POSITIVE] Random Sampling without Context (from LM prior)
Drawing samples from the language model's prior distribution to generate natural language phrases as separators, without task context.

**Delta**: +12% relative improvement over human baselines; less than 1% difference from previous state-of-the-art prompt optimisation methods
**Condition**: Applicable to text classification tasks; does not require instruction-tuned models.

**Evidence**: "Experimental results show that the Random w/o Context is significantly better than human baselines (12% relative improvement) and nearly on par with previous state-of-the-art prompt optimisation methods (less than a 1% difference)."

## [POSITIVE] Random Sampling with Context
Sampling from a language model conditioned on a few training examples to generate task-relevant separators.

**Delta**: +0.3% relative improvement over Random w/o Context; +0.9% over Random Vocabulary
**Condition**: Applicable when task-relevant context is available; provides marginal improvements over context-free random methods.

**Evidence**: "The Random with Context method achieves a relative improvement of 0.3% over Random w/o Context and 0.9% over Random Vocabulary."

## [POSITIVE] Random Separator Selection via Evaluation and Selection
Evaluating randomly generated separators on a small training set and selecting the best-performing one for test evaluation.

**Delta**: Outperforms human-curated separators by 12% on average; competitive with OPRO and other optimisation methods
**Condition**: Applicable when a small labelled training set (e.g., 64 samples) is available for evaluation.

**Evidence**: "Our random strategies offer considerable flexibility in discovering task-wide performant separators. Random separators attain a 12% average relative improvement across nine classification tasks on eight language models, compared to human-curated separators."

## [POSITIVE] Using Random Separators for Generative Reasoning Tasks (GSM8K)
Applying random vocabulary sampling to mathematical reasoning tasks, using 5-shot prompting and majority voting.

**Delta**: +23% relative increase in accuracy over Chain-of-Thought baseline (47.3 vs 38.3)
**Condition**: Applicable to generative reasoning tasks like GSM8K; shows better robustness than CoT across different demonstration sets.

**Evidence**: "Our best random separators reach an average accuracy of 47.3, a 23% relative increase in accuracy over the CoT baseline."

## [POSITIVE] OPRO (Optimization by Prompting) with Meta-Prompt
Using a meta-prompt with natural language instructions and historical solutions to generate new separators iteratively.

**Delta**: Top average performance for 4 out of 7 models; 0.1% average performance difference compared to random methods
**Condition**: Applicable when an instruction-tuned or pretrained language model is available; requires a meta-prompt with task description and historical solutions.

**Evidence**: "OPRO and its in-context learning variant achieve the top average performance for four out of seven models. However, the advantage is minimal, with a 0.1% average performance difference compared to random methods."

## [POSITIVE] OPRO-ICL (In-Context Learning Variant)
Removing all instructions from OPRO's meta-prompt to create an in-context learning variant that only uses historical solutions and context.

**Delta**: Comparable to OPRO; achieves top average performance for 4 out of 7 models
**Condition**: Applicable when instruction-tuned models are not available; relies on in-context learning from examples.

**Evidence**: "OPRO-ICL achieves the top average performance for four out of seven models, with minimal advantage over random methods."

## [NEUTRAL] Human-Curated Separator Baseline (Answer:)
Using the widely used human-curated separator 'Answer:' as a baseline for comparison.

**Delta**: Baseline performance; outperformed by random methods by 12% on average
**Condition**: Serves as a baseline; not recommended as an optimal choice given the effectiveness of random separators.

**Evidence**: "Random separators attain a 12% average relative improvement across nine classification tasks on eight language models, compared to human-curated separators."

## [NEUTRAL] Zero-Shot Chain-of-Thought (Let's think step by step)
Using the thinking-style phrase 'Let's think step by step' as a separator for reasoning tasks.

**Delta**: Only slightly better than average random separators on GSM8K (38.3 vs 37.8); high variance across demonstrations
**Condition**: Applicable to reasoning tasks; shows high sensitivity to demonstration selection.

**Evidence**: "For the few-shot mathematical reasoning task, thinking-style phrases could only be slightly better than random separators. On average, CoT attains an accuracy of 38.3 and the average of random separators is 37.8."

## [NEGATIVE] Cross-Task Transferability of Random Separators
Testing whether the best random separator from one task transfers to other tasks.

**Delta**: Near random-guessing performance on different tasks; average score 59.4% vs 59.0% for human baseline
**Condition**: Random separators are task-specific and do not transfer well across different tasks.

**Evidence**: "According to Table 10, we do not observe high transferability across tasks; for example, a performant prompt for SST2 has almost random-guessing performance on SST5, and vice versa."

## [POSITIVE] Cross-Context Transferability within Same Task
Testing whether the best random separator from one context (demonstration set) transfers to other contexts within the same task.

**Delta**: 73.4% accuracy vs 50.6% for human-curated separator on AGNews
**Condition**: Applicable within the same task; random separators show robustness across different demonstration sets.

**Evidence**: "Random separators exhibit a degree of transferability and are significantly better than human-curated ones (73.4% versus 50.6%)."
