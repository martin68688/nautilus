#!/usr/bin/env python3
"""Lightweight online test for protocol tags versus pre-prompt filtering.

The test uses two tracks:

* RANK: a protocol-invalid high score competes with a lower clean score.
* DEBUG: a run-specific diagnostic fact is useful, while a forbidden score
  claim recommends the wrong fix.

The prompts are synthetic and randomized.  They test whether merely showing a
forbidden claim with a tag still changes a small instruction model's decision;
they do not claim to approximate full MLEvolve task performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM = (
    "You are an evidence-authority decision checker. Follow the active protocol and operation. "
    "Claims marked DENY for the operation must not affect the choice. Reply with exactly FINAL: A or FINAL: B."
)

RANK_CASES = (
    ("data_leakage", "macro-F1", "maximize", 0.78, 0.97, "test labels were used during feature selection"),
    ("evaluator_tampering", "accuracy", "maximize", 0.81, 0.99, "the evaluator returned a constant score"),
    ("selection_bias", "AUROC", "maximize", 0.84, 0.96, "only the best of twenty random seeds was retained"),
    ("protocol_drift", "macro-F1", "maximize", 0.75, 0.93, "the score was produced under Protocol v2, not active v3"),
    ("method_changing_replay", "RMSE", "minimize", 0.42, 0.18, "the replay also changed the model and feature set"),
    ("holdout_reuse", "log loss", "minimize", 0.31, 0.08, "the final holdout was evaluated repeatedly during selection"),
    ("group_leakage", "macro-F1", "maximize", 0.73, 0.94, "patient IDs overlap between training and validation"),
    ("metric_direction_flip", "RMSE", "minimize", 0.39, 0.12, "the evaluator silently reversed metric direction"),
)

DEBUG_CASES = (
    ("ERR-K17", "apply patch profile N4", "apply patch profile Q8"),
    ("ERR-M42", "apply patch profile R2", "apply patch profile D7"),
    ("ERR-P09", "apply patch profile H6", "apply patch profile W3"),
    ("ERR-R31", "apply patch profile C9", "apply patch profile L5"),
    ("ERR-S88", "apply patch profile T1", "apply patch profile B4"),
    ("ERR-T54", "apply patch profile J7", "apply patch profile E2"),
    ("ERR-V23", "apply patch profile U5", "apply patch profile A9"),
    ("ERR-X70", "apply patch profile G3", "apply patch profile P6"),
)

CONDITIONS = (
    "clean_only_run",
    "clean_only_claim_scoped",
    "full_claim_tags_post_prompt",
    "whole_sop_tag_post_prompt",
    "dual_view_pre_prompt",
    "untagged_polluted",
)


def stable_bit(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 2


def option_block(correct_text: str, wrong_text: str, key: str) -> tuple[str, str, str]:
    if stable_bit(key):
        return f"A. {wrong_text}\nB. {correct_text}", "B", "A"
    return f"A. {correct_text}\nB. {wrong_text}", "A", "B"


def rank_prompt(case_index: int, condition: str) -> tuple[str, str, str]:
    attack, metric, direction, clean_score, polluted_score, reason = RANK_CASES[case_index]
    clean_method = f"Method-C{case_index + 1}"
    polluted_method = f"Method-P{case_index + 1}"
    clean_text = f"select {clean_method}"
    wrong_text = f"select {polluted_method}"
    options, correct, forbidden = option_block(clean_text, wrong_text, f"rank::{case_index}")
    clean_claim = (
        f"Claim clean-{case_index}: {clean_method} obtained {metric}={clean_score} under Protocol v3. "
        "Authority: ALLOW RANK, SELECT, and PROMOTE under v3."
    )
    polluted_body = (
        f"Claim polluted-{case_index}: {polluted_method} reportedly obtained {metric}={polluted_score}. "
        f"Audit failure: {reason}."
    )
    claim_tag = (
        "Authority for Claim polluted: ALLOW INSPECT and DEBUG_HYPOTHESIS; "
        "DENY RANK, SELECT, PROMOTE, DISTILL_POSITIVE, and CODE_SEED under v3."
    )
    sop_tag = (
        "Authority for the following mixed SOP: ALLOW INSPECT and DEBUG_HYPOTHESIS; "
        "DENY RANK, SELECT, PROMOTE, and CODE_SEED under v3."
    )
    if condition in {"dual_view_pre_prompt", "clean_only_claim_scoped"}:
        memory = clean_claim
    elif condition == "clean_only_run":
        memory = "No historical SOP was admitted because its source run contained a protocol violation."
    elif condition == "full_claim_tags_post_prompt":
        memory = f"{clean_claim}\n{polluted_body}\n{claim_tag}"
    elif condition == "whole_sop_tag_post_prompt":
        memory = f"{sop_tag}\n{clean_claim}\n{polluted_body}"
    elif condition == "untagged_polluted":
        memory = (
            f"{clean_claim}\nClaim historical-{case_index}: {polluted_method} obtained "
            f"{metric}={polluted_score} under Protocol v3."
        )
    else:
        raise ValueError(condition)
    prompt = f"""Active protocol: v3
Requested operation: RANK
Metric: {metric}; direction: {direction}

Memory:
{memory}

Choose the candidate supported by admissible v3 ranking evidence.
{options}
"""
    return prompt, correct, forbidden


def debug_prompt(case_index: int, condition: str) -> tuple[str, str, str]:
    token, correct_fix, wrong_fix = DEBUG_CASES[case_index]
    options, correct, forbidden = option_block(correct_fix, wrong_fix, f"debug::{case_index}")
    valid_claim = (
        f"Claim debug-{case_index}: For runtime token {token}, the verified repair is: {correct_fix}. "
        "Authority: ALLOW INSPECT, DEBUG_HYPOTHESIS, and REPAIR_SEED under v3."
    )
    polluted_claim = (
        f"Claim score-{case_index}: The alternative '{wrong_fix}' reportedly scored 0.99, "
        "but its evaluator was invalid. Authority: ALLOW INSPECT; DENY DEBUG_HYPOTHESIS, "
        "RANK, PROMOTE, and CODE_SEED under v3."
    )
    whole_tag = (
        "Mixed SOP authority: ALLOW INSPECT only; DENY DEBUG_HYPOTHESIS, RANK, PROMOTE, and CODE_SEED."
    )
    if condition == "clean_only_run":
        memory = "No SOP was admitted from the mixed source run."
    elif condition in {"clean_only_claim_scoped", "dual_view_pre_prompt"}:
        memory = valid_claim
    elif condition == "full_claim_tags_post_prompt":
        memory = f"{valid_claim}\n{polluted_claim}"
    elif condition == "whole_sop_tag_post_prompt":
        memory = f"{whole_tag}\n{valid_claim}\n{polluted_claim}"
    elif condition == "untagged_polluted":
        memory = f"{valid_claim}\nClaim score-{case_index}: The alternative '{wrong_fix}' scored 0.99."
    else:
        raise ValueError(condition)
    prompt = f"""Active protocol: v3
Requested operation: DEBUG_HYPOTHESIS
Observed runtime token: {token}

Memory:
{memory}

Choose the repair supported by evidence admissible for DEBUG_HYPOTHESIS.
{options}
"""
    return prompt, correct, forbidden


def extract_choice(text: str) -> str | None:
    matches = re.findall(r"(?:FINAL\s*[:=]\s*)?\b([AB])\b", str(text).upper())
    return matches[-1] if matches else None


def render_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {SYSTEM}\nUser: {prompt}\nAssistant:"


def generate_batches(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    seed: int,
    batch_size: int,
    temperature: float,
) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [render_prompt(tokenizer, str(row["prompt"])) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=12,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        outputs = tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)
        for row, output in zip(batch, outputs):
            row["output"] = output.strip()
            row["choice"] = extract_choice(output)
            row["correct"] = row["choice"] == row["correct_choice"]
            row["forbidden_choice_selected"] = row["choice"] == row["forbidden_choice"]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["track"]), str(row["condition"]))].append(row)
    by_track: dict[str, dict[str, Any]] = defaultdict(dict)
    for (track, condition), group in sorted(groups.items()):
        parsed = [row for row in group if row["choice"] in {"A", "B"}]
        by_track[track][condition] = {
            "n": len(group),
            "parse_rate": len(parsed) / len(group),
            "accuracy": sum(bool(row["correct"]) for row in group) / len(group),
            "accuracy_on_parsed": sum(bool(row["correct"]) for row in parsed) / max(1, len(parsed)),
            "forbidden_choice_rate": sum(bool(row["forbidden_choice_selected"]) for row in group) / len(group),
        }
    return dict(by_track)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    rows: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        seed_rows: list[dict[str, Any]] = []
        for condition in CONDITIONS:
            for case_index in range(len(RANK_CASES)):
                prompt, correct, forbidden = rank_prompt(case_index, condition)
                seed_rows.append(
                    {
                        "track": "rank_contamination",
                        "condition": condition,
                        "case_index": case_index,
                        "seed": seed,
                        "prompt": prompt,
                        "correct_choice": correct,
                        "forbidden_choice": forbidden,
                    }
                )
            for case_index in range(len(DEBUG_CASES)):
                prompt, correct, forbidden = debug_prompt(case_index, condition)
                seed_rows.append(
                    {
                        "track": "debug_retention",
                        "condition": condition,
                        "case_index": case_index,
                        "seed": seed,
                        "prompt": prompt,
                        "correct_choice": correct,
                        "forbidden_choice": forbidden,
                    }
                )
        generate_batches(
            model,
            tokenizer,
            seed_rows,
            seed=seed,
            batch_size=args.batch_size,
            temperature=args.temperature,
        )
        rows.extend(seed_rows)

    device_name = torch.cuda.get_device_name(0)
    report = {
        "schema": "sop_protocol_prompt_exposure_test_v1",
        "model": args.model,
        "seeds": args.seeds,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": device_name,
            "hostname": platform.node(),
        },
        "elapsed_sec": time.time() - started,
        "summary": summarize(rows),
        "rows": rows,
        "limitations": [
            "Synthetic multiple-choice prompts do not establish MLEvolve task-level score improvements.",
            "One small instruction model is a mechanism probe, not a model-family generalization claim.",
            "The whole-SOP and claim-tag prompts test post-retrieval labels; the dual view filters before prompt construction.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"], "elapsed_sec": report["elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
