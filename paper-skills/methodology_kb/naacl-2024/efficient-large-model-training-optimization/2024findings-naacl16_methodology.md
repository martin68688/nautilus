# LETI: Learning to Generate from Textual Interactions

**Source**: https://aclanthology.org/2024.findings-naacl.16/

## [POSITIVE] Feedback-conditioned Fine-tuning (FCFT)
Fine-tuning the LM on a concatenation of natural language instruction, LM-generated program, and textual feedback (error messages/stack traces), with a binary reward token prepended to differentiate correct and buggy solutions.

**Delta**: 63.2% more syntactically correct and executable code (on 2B LM); Pass@1 improves from 4.50% to 28.00% on MBPP test set (2B, w/o post-processing); outperforms supervised fine-tuned baseline on 2B model.
**Condition**: Code generation tasks (MBPP, HumanEval) and NLP tasks formulated as code generation (Event Argument Extraction).

**Evidence**: "LETI improves test set Pass@1 with increasing iterations and outperforms a supervised fine-tuned baseline (for the 2B model)."

## [POSITIVE] Textual Feedback from Python Interpreter
Using stack traces and error messages from a Python interpreter as textual feedback when generated code fails, instead of only binary feedback (correct/incorrect).

**Delta**: 2.24x faster improvement per iteration for 2B model compared to binary-only feedback; achieves same performance with fewer than half of gradient steps.
**Condition**: Code generation tasks where automatic textual feedback is available from a Python interpreter.

**Evidence**: "LMs trained with textual feedback obtain better final performance and improve faster (up to 2.2x for 2B)."

## [POSITIVE] Binary Reward Token Conditioning
Prepending <|good|> or <|bad|> reward tokens to fine-tuning sequences to differentiate correct and buggy solutions, enabling conditional generation of better solutions.

**Delta**: <|good|> generally outperforms <|bad|> and both outperform no conditioning on in-domain MBPP dataset.
**Condition**: In-domain code generation tasks (MBPP); less effective on out-of-domain datasets like HumanEval.

**Evidence**: "We find that <|good|> generally outperforms <|bad|> (i.e., positive ∆<|good|>−<|bad|>) and both reward tokens outperform none on in-domain dataset MBPP."

## [POSITIVE] Continued Pre-training Regularization
Interleaving FCFT optimization with LM objective optimization on pre-training data (Python subset of TheStack v1.1) to alleviate distribution shifts.

**Delta**: Maintains or improves out-of-domain reasoning performance; removing regularization degrades GSM8K PaL performance from 16.68% to 7.88% (350M) and BBH CoT performance.
**Condition**: When maintaining out-of-domain reasoning capabilities (GSM8K, Big-Bench-Hard) is important.

**Evidence**: "Removing regularization significantly degrades LM's capability on PaL-prompted GSM-8K... it also degrades BBH's chain-of-thought performance."

## [POSITIVE] Iterative Multi-round Fine-tuning
Repeating the generate-evaluate-fine-tune cycle for multiple rounds (iterations), allowing the model to iteratively improve from past mistakes.

**Delta**: Pass@1 improves from 4.50% to 28.00% over 6 iterations (2B, w/o post-processing); NameError reduced from 10% to 1% in 2 iterations.
**Condition**: When iterative improvement is feasible; most effective in early iterations (diminishing returns later).

**Evidence**: "LETI can iteratively improve the success rate of the LMs' generated solutions on training set problems."

## [POSITIVE] Rule-based Solution Evaluator for NLP Tasks
Implementing a custom rule-based evaluator for Event Argument Extraction that checks argument constraints, matches predictions to ground truths, and checks completeness.

**Delta**: Arg-I F1 increases by 12.3% (21.2% → 33.5%), Arg-C F1 increases 2.6% (8% → 10.6%) over 3 iterations.
**Condition**: NLP tasks that can be formulated as code generation; evaluator design influences which metrics are optimized (precision vs. recall).

**Evidence**: "LETI's performance on EAE task... LETI is capable of improving the train and test pass rate of generated solutions."

## [POSITIVE] Sampling Multiple Solutions per Problem (n=128)
Generating 128 candidate solutions per problem to increase exploration and chance of finding correct solutions.

**Delta**: Consistent benefit from larger n; n=128 outperforms n=16 and n=64.
**Condition**: When computational budget allows; more important for smaller models or harder problems.

**Evidence**: "LETI consistently benefits from larger n for each problem (i.e., more exploration)."

## [POSITIVE] No Ground-truth Code Required
Training without any ground-truth solutions, learning solely from evaluator feedback (binary + textual).

**Delta**: Outperforms supervised fine-tuned baseline on 2B model despite not using ground-truth code.
**Condition**: When ground-truth solutions are unavailable or expensive to obtain.

**Evidence**: "LETI requires no ground-truth code for training and even outperforms a fine-tuned baseline that does."

## [POSITIVE] Post-processing Heuristics (Stop-word-based)
Commonly used heuristics to remove irrelevant code (e.g., keep only first code block) before evaluation.

**Delta**: 2B LM improves from 26.89% to 29.53% Pass@1 within two iterations with post-processing.
**Condition**: When post-processing is feasible; reduces certain error types (e.g., NameError) but may not scale to all tasks.

**Evidence**: "LETI is also helpful when the post-processing heuristic is applied to the LM's output: 2B LM improves from 26.89% to 29.53% within two iterations."

## [POSITIVE] Larger Model Scale Benefits
Larger LMs (2B vs 350M) benefit more from LETI, showing larger improvements per iteration and better ability to leverage textual feedback.

**Delta**: 2B model improves 2.24x faster with textual feedback vs binary; 350M only 1.57x faster. 2B obtains ~8x more improvement per iteration than 350M.
**Condition**: When using larger LMs (2B+); smaller models (350M) may lack capacity to fully leverage textual feedback.

**Evidence**: "A larger model is more effective in learning from textual feedback and can obtain a larger (average) improvement per iteration than a baseline that only uses binary feedback."

## [POSITIVE] Larger Training Problem Set Size
Using more training problems (|P|=374 full dataset vs smaller subsets) leads to faster and more significant improvements.

**Delta**: Larger |P| leads to faster and more significant improvements; 350M model needs |P|≥128 for net positive test improvement.
**Condition**: When sufficient training problems are available; smaller models require more diverse problems to benefit.

**Evidence**: "The number of training problems impacts the performance of LMs on test sets: larger |P| generally leads to faster and more significant improvements."
