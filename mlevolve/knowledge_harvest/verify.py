"""Check knowledge entries on held-out training eras before they reach the agent.

For each selected entry the LLM writes ``variant(ctx)``: the entry's method
applied to a fixed LightGBM baseline (see verify_remote.py for ``ctx``). The
code runs on a compute host over ssh; the held-out labels never reach it.
The result is a ``check`` on the entry: helps / no_effect / hurts (paired
per-era CORR difference vs the baseline, Newey-West t with 12 lags), failed
(no working implementation after one repair), or untestable.

Usage (from mlevolve/, LLM credentials in OPENAI_API_KEY / OPENAI_BASE_URL):
  python -m knowledge_harvest.verify --knowledge DIR/knowledge_set.json \\
      --host wsl --remote-dir experiments/ks_verify --remote-python venvs/mlevolve/bin/python \\
      --remote-train data/numerai-v6/ks_eval/train.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

from .llm import JsonLLM

PROMPT_VERSION = "verify-v1"
MODELLING = ("targets", "features", "tree_models", "neural_networks", "ensembling",
             "neutralization_and_risk", "data_and_eras")
FORBIDDEN = re.compile(r"read_parquet|read_csv|np\.load|open\(|heldout|_scoring|glob|listdir|os\.walk|pathlib|subprocess")

CTX_DOC = """ctx fields (read-only):
- ctx.X_train: float32 array [n_train, n_features] (memory-mapped; rows of every 4th training era; NaN possible in float features)
- ctx.X_val: float32 array [n_val, n_features] for the held-out eras (features only, no labels)
- ctx.train_era, ctx.val_era: era label per row (str, weekly dates)
- ctx.targets: dict target name -> float32 array [n_train] (NaN where missing); ctx.main_target = "target" is scored
- ctx.feature_names: list of column names; ctx.feature_sets: dict set name -> list of names (may be empty)
- ctx.baseline_params: LightGBM params of the baseline; ctx.baseline_val_pred: the baseline's held-out predictions
- ctx.fit_lgbm(params, X, y, **fit_kwargs) -> fitted LGBMRegressor (drops NaN y, uses 24 threads)
- ctx.gauss_rank(pred, era) -> per-era gaussianized ranks; ctx.neutralize(pred, feats, era, proportion) -> per-era
  neutralized and standardized predictions against the columns of feats
- ctx.n_jobs = 24
The baseline is: model = ctx.fit_lgbm(ctx.baseline_params, ctx.X_train, ctx.targets["target"]); return model.predict(ctx.X_val)"""

SYSTEM = f"""You turn one Numerai modelling method into a controlled experiment against a fixed baseline.
Write Python defining `def variant(ctx):` that returns a 1-D array of predictions for ctx.X_val rows (only the
per-era ranking matters). Apply ONLY the given method on top of the baseline; keep everything else as in the baseline.
{CTX_DOC}
Limits: at most 20 minutes on 24 cores and 40 GB of RAM (each 2000-tree baseline fit takes about 4 minutes; use at most
5 such fits). Use numpy, pandas, scipy, sklearn, lightgbm only. Do not read or write any file, and do not use the
held-out eras' labels (there are none in ctx). Use the method's concrete parameters when it gives them.
Return a JSON object {{"testable": bool, "reason": "one sentence", "code": "python source"}}. Set testable to false
(with empty code) when the method cannot be expressed with this data and harness (e.g. needs live data, other
datasets, staking or a different metric)."""


def candidates(data: dict, grades: str, limit: int) -> list[dict]:
    order = {g: n for n, g in enumerate("ABCD")}
    pool = [e for e in data["entries"]
            if e["grade"] in grades and e.get("stance") == "recommend" and e["category"] in MODELLING]
    pool.sort(key=lambda e: (order[e["grade"]], -e["n_sources"]))
    return pool[:limit]


def _entry_text(entry: dict) -> str:
    parts = [f"Method: {entry['title']}", entry["principle"]]
    if entry.get("condition"):
        parts.append(f"When: {entry['condition']}")
    if entry.get("caveats"):
        parts.append("Caveats: " + "; ".join(entry["caveats"]))
    return "\n".join(parts)


class Remote:
    def __init__(self, host: str, root: str, python: str):
        self.host, self.root, self.python = host, root.rstrip("/"), python

    def sh(self, command: str, timeout: int | None = None) -> str:
        done = subprocess.run(["ssh", self.host, command], capture_output=True, text=True, timeout=timeout)
        if done.returncode != 0:
            raise RuntimeError(done.stderr[-2000:] or done.stdout[-2000:])
        return done.stdout

    def put(self, local: Path, remote: str) -> None:
        subprocess.run(["scp", "-q", str(local), f"{self.host}:{remote}"], check=True)

    def harness(self, args: str, timeout: int | None = None) -> dict:
        out = self.sh(f"cd {self.root} && {self.python} verify_remote.py {args}", timeout=timeout)
        return json.loads(out.strip().splitlines()[-1])


def verdict(result: dict) -> str:
    if result["diff"] > 0 and result["t"] >= 2:
        return "helps"
    if result["diff"] < 0 and result["t"] <= -2:
        return "hurts"
    return "no_effect"


def check_entry(entry: dict, llm: JsonLLM, remote: Remote, work: Path, timeout: int) -> dict:
    prompt = _entry_text(entry)
    answer = llm(SYSTEM, prompt)
    if not answer.get("testable", True) or not answer.get("code"):
        return {"verdict": "untestable", "reason": answer.get("reason", "")}
    error = ""
    for attempt in range(2):
        code = answer.get("code", "")
        if FORBIDDEN.search(code):
            error = "code touches files or held-out data: " + FORBIDDEN.search(code).group(0)
        else:
            path = work / f"{entry['id']}_{attempt}.py"
            path.write_text(code, encoding="utf-8")
            remote.put(path, f"{remote.root}/variants/{path.name}")
            result = remote.harness(
                f"run --cache cache --variant variants/{path.name} --name {shlex.quote(entry['id'])} --timeout {timeout}",
                timeout=timeout + 600,
            )
            if result["status"] == "ok":
                return {"verdict": verdict(result), "reason": answer.get("reason", ""), "attempts": attempt + 1,
                        **{k: result[k] for k in ("corr", "sharpe", "diff", "t", "win_rate", "eras", "seconds")}}
            error = result.get("error") or result["status"]
        if attempt == 0:
            answer = llm(SYSTEM, f"{prompt}\n\nYour previous code failed:\n```python\n{code}\n```\nError:\n{error[-2500:]}\n"
                                 "Return a corrected JSON object.")
    return {"verdict": "failed", "error": error[-500:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knowledge", type=Path, required=True)
    ap.add_argument("--host", default="wsl")
    ap.add_argument("--remote-dir", default="experiments/ks_verify")
    ap.add_argument("--remote-python", default="venvs/mlevolve/bin/python")
    ap.add_argument("--remote-train", default="data/numerai-v6/ks_eval/train.parquet")
    ap.add_argument("--grades", default="AB")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--model", default="deepseek-v4-pro")
    args = ap.parse_args()

    out = args.knowledge.parent
    work = out / "verify"
    work.mkdir(exist_ok=True)
    llm = JsonLLM(args.model, out / "cache" / "llm_verify", max_tokens=12000)
    remote = Remote(args.host, args.remote_dir, args.remote_python)
    remote.sh(f"mkdir -p {remote.root}/variants")
    remote.put(Path(__file__).with_name("verify_remote.py"), f"{remote.root}/verify_remote.py")
    if "ok" not in remote.sh(f"test -f {remote.root}/cache/meta.json && echo ok || true"):
        print("prepare:", remote.harness(f"prepare --train ~/{args.remote_train} --cache cache"), flush=True)
    if "ok" not in remote.sh(f"test -f {remote.root}/cache/baseline_scores.json && echo ok || true"):
        print("baseline:", remote.harness("baseline --cache cache"), flush=True)

    data = json.loads(args.knowledge.read_text(encoding="utf-8"))
    checks_path = out / "checks.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    for entry in candidates(data, args.grades, args.limit):
        if entry["id"] in checks:
            continue
        try:
            checks[entry["id"]] = check_entry(entry, llm, remote, work, args.timeout)
        except Exception as error:  # ssh drop or a harness crash: record it and continue
            checks[entry["id"]] = {"verdict": "failed", "error": str(error)[-500:]}
        checks[entry["id"]]["title"] = entry["title"]
        print(entry["id"], json.dumps(checks[entry["id"]])[:300], flush=True)
        checks_path.write_text(json.dumps(checks, indent=1))

    for entry in data["entries"]:
        entry["check"] = checks.get(entry["id"], {"verdict": "untested"})
    data["manifest"]["check"] = {"prompt": PROMPT_VERSION, "model": args.model, "grades": args.grades,
                                 "baseline": "LightGBM 2000 trees, lr 0.01, depth 5, 31 leaves, colsample 0.1",
                                 "heldout": "last 156 training eras after a 16-era purge; train rows every 4th era"}
    (out / "knowledge_verified.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out / 'knowledge_verified.json'}")


if __name__ == "__main__":
    main()
