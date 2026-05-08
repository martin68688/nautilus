# PaD: Program-aided Distillation Can Teach Small Models Reasoning Better than Chain-of-thought Fine-tuning

**Source**: https://aclanthology.org/2024.naacl-long.142/

## [POSITIVE] Program-aided Distillation (PaD)
Uses reasoning programs (Python code) instead of natural language chain-of-thought to distill reasoning ability from LLMs to smaller models. Programs can be automatically checked by a Python interpreter.

**Delta**: outperforms baselines; achieves 10% improvement while using 35% of baseline data size
**Condition**: Arithmetic reasoning tasks (GSM8K, ASDiv, SVAMP, MultiArith) and symbolic reasoning tasks.

**Evidence**: "PaD achieves a 10% improvement while utilizing just 35% of the baseline model's data size. And it accomplishes comparable performance utilizing merely 4.5% of the baseline model's data size."

## [POSITIVE] Data Filtering via Python Interpreter
Automatically filters out synthetic data with faulty reasoning by executing the generated Python code and checking for execution errors or incorrect return values.

**Delta**: enables higher quality dataset; key to distillation quality
**Condition**: During data synthesis from LLMs; applicable to any reasoning task where programs can be executed.

**Evidence**: "We can filter out incorrect samples to refine our fine-tune dataset. We regard this as a crucial step in our distillation process. Intuitively, higher quality data can improve performance while incorrect reasoning steps may confound models."

## [POSITIVE] Self-Refinement via Error Injection
Injects errors into reasoning programs (e.g., NameError, SyntaxError) and trains the small model to correct these errors using error messages, enabling iterative self-improvement.

**Delta**: further improvement over PaD alone (see ablation Figure 4)
**Condition**: During fine-tuning; requires constructing error datasets from AST manipulation.

**Evidence**: "Employing self-refinement and step-by-step verification can bring further improvement."

## [POSITIVE] Step-by-Step Verification (Step-wise Beam Search)
During decoding, generates multiple candidate reasoning steps, scores them using a pre-trained reasoning scorer (cosine similarity), and selects the most faithful steps to guide generation.

**Delta**: further improvement over PaD alone (see ablation Figure 4)
**Condition**: During inference/decoding; applicable to any stepwise reasoning generation.

**Evidence**: "Employing self-refinement and step-by-step verification can bring further improvement."

## [POSITIVE] Data Augmentation via Multiple Contexts
Uses different context examples for the same question to synthesize diverse reasoning programs, increasing dataset diversity.

**Delta**: PaD with augmentation achieves 30.6% on GSM8K vs 13.0% without augmentation (Table 3)
**Condition**: During data synthesis; applied to GSM8K train set with 8 rounds of synthesis.

**Evidence**: "Since one question can correspond to multiple solutions and diverse reasoning data could improve performance, we use different contexts for the same question to synthesize different reasoning programs. This augmentation enhanced the diversity of data."

## [POSITIVE] Using Program Format Instead of Natural Language CoT
Replaces chain-of-thought natural language reasoning with Python code, which has simpler syntax and narrower output space.

**Delta**: consistently lower training and evaluation losses than CoT fine-tuning (Figure 5)
**Condition**: All reasoning tasks; particularly effective for mathematical and symbolic reasoning.

**Evidence**: "PaD effectively narrows the prediction space, focusing on specific central points. ... PaD primarily adheres to Python syntax, instead of extensive sampling across the entire language representation space. This, in essence, reduces the complexity of the task."

## [NEGATIVE] Specialization Trade-off (Decline in General Ability)
As the small model specializes in reasoning tasks, its performance on general ability benchmarks (BBH) declines.

**Delta**: BBH drops significantly (e.g., CodeT5_large: 28.1 to 1.1 on BBH, Table 1)
**Condition**: When fine-tuning small models (0.06B-0.77B) on reasoning tasks; general ability evaluated on BBH.

**Evidence**: "While improving the reasoning capability of smaller models, it leads to a decline in general abilities. ... We speculate that with restricted parameters, a small model can only precisely master certain abilities."

## [NEUTRAL] Self-Distillation (Appendix B)
Uses the small model itself as a teacher to align prediction distributions, adding a regularization term to cross-entropy loss.

**Delta**: minor boost (Figure 7); only effective for some models
**Condition**: After initial fine-tuning; applied iteratively on small models.

**Evidence**: "Self-distillation is only effective for some models, which is why we did not include this method in the main body of our text."
