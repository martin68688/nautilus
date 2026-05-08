# Self-Regulated Sample Diversity in Large Language Models

**Source**: https://aclanthology.org/2024.findings-naacl.122/

## [POSITIVE] Self-Regulated Sample Diversity via Prompting
A method that dynamically adjusts LLM sampling parameters (temperature, top-p, frequency penalty, presence penalty) based on the input prompt, using a guidance prompt to generate parameter values before task completion.

**Delta**: GPT-3.5: +4.4%, GPT-4: +1.5% overall accuracy on MMLU
**Condition**: Applicable to large foundation models (GPT-3.5, GPT-4) across diverse tasks in MMLU benchmark; requires model capable of few-shot guidance task.

**Evidence**: "our method demonstrates significant improvement in all supercategories of the MMLU multitask benchmark (GPT-3.5: +4.4%, GPT-4: +1.5%)"

## [POSITIVE] Guidance Prompt Design
A human-generated prompt instructing the LLM to output suitable sampling parameters based on the task, with examples (e.g., 'maths should have more correct non-diverse answers, whereas prompts about fiction should be more creative').

**Delta**: Outperforms static parameter baselines across all MMLU supercategories
**Condition**: Used as the guidance prompt g in the methodology; effective for GPT-3.5 and GPT-4.

**Evidence**: "the method demonstrates consistent improvement in average accuracy across all MMLU task supercategories, shown in Table 1"

## [POSITIVE] Dynamic Parameter Extraction (Ψ function)
Extracting updated diversity parameters from the LLM's output string (e.g., 'temperature=0.0 top_p=1 presence_penalty=0 frequency_penalty=0') by parsing the generated text into a real vector.

**Delta**: Enables the overall accuracy improvements reported
**Condition**: Used in the inference pipeline; requires successful parsing of model output.

**Evidence**: "We then extract the updated parameter values w′ ∈ Rn from this output string s by the function Ψ"

## [NEUTRAL] Continual Diversity Updates (Equation 4)
A variant where the LLM is prompted to output syntax during generation that triggers dynamic parameter updates mid-generation, allowing mixed diversity requirements within a single prompt.

**Delta**: Not quantitatively evaluated; described as sufficient for general use with current models
**Condition**: Applicable when task prompt has mixed diversity requirements (e.g., 'solve y=100×100, then write a poem about it').

**Evidence**: "In practice, we find the approach in equations 1–3 sufficient for general use with current models"

## [POSITIVE] Integration with Chain-of-Thought (CoT) and Few-Shot Learning
Combining the self-regulated diversity method with CoT reasoning and 5-shot examples to further improve performance.

**Delta**: GPT-3.5 CoT+5shot: +4.4% (0.648 to 0.692); GPT-4 CoT+5shot: +1.5% (0.830 to 0.845)
**Condition**: Applied on top of the self-regulated method; tested on GPT-3.5 and GPT-4 with MMLU.

**Evidence**: "With the integration of Chain-of-Thought (CoT) and 5-shot learning, the accuracy improved from 0.648 to 0.692, yielding an increase of 4.4%"

## [NEUTRAL] Error Handling with Restart
If parameter extraction fails (incorrect data, inference failure, or incorrect ranges), the query is simply restarted; defaults used after n restarts if efficiency is prioritized.

**Delta**: No quantitative impact reported; described as practical safeguard
**Condition**: Used during inference to handle edge cases; no infinite loops or significant delays observed.

**Evidence**: "If the parameter extraction fails (incorrect parameter data, inference failure or incorrect parameter ranges) we simply restart the query"
