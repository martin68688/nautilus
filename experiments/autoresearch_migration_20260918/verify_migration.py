"""Reproduce upstream failure modes and verify real-pack lossless transport."""
import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from memory_bridge import MemoryBridge

ROOT = Path(__file__).resolve().parent


def main():
    inputs = json.loads((ROOT / "AUDIT_INPUTS.json").read_text())
    source = (ROOT / "upstream/main.py").read_text()
    tree = ast.parse(source)
    # Execute the actual upstream functions, without importing/installing SDK.
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name in {"parse_metrics", "run_training"}]
    ns = {"re": re, "os": os, "asyncio": asyncio, "Path": Path, "PYTHON": sys.executable}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "upstream/main.py", "exec"), ns)
    findings = {}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        ns.update(REPO=folder, RUN_LOG=folder / "run.log")
        (folder / "train.py").write_text("print('val_bpb: 0.100000', flush=True)\nraise SystemExit(17)\n")
        result = asyncio.run(ns["run_training"](None, 10, 1))
        assert result == (0.1, 0.0, True), result
        findings["nonzero_exit_accepted"] = {"reproduced": True, "exit_code": 17, "upstream_result": result}
        ns["RUN_LOG"].write_text("val_bpb: 1.2e-3\n")
        value = ns["parse_metrics"](ns["RUN_LOG"])[0]
        assert value == 1.2
        findings["scientific_notation_misparsed"] = {"reproduced": True, "expected": 0.0012, "actual": value}
        ns["RUN_LOG"].write_text("val_bpb: 9" + "0" * 400 + "\n")
        assert ns["parse_metrics"](ns["RUN_LOG"])[0] == float("inf")
        findings["nonfinite_not_rejected"] = {"reproduced": True}
    checks = {}
    for task, names in [("spooky-author-identification", ["spooky_search", "spooky_inner"]),
                        ("leaf-classification", ["leaf_search"])]:
        specs = {name: inputs["candidate_memory_packs"][name] for name in names}
        data_hash = specs[names[0]]["data_manifest_sha256"]
        bridge = MemoryBridge(specs, task, data_hash)
        keys = list(bridge.cards)
        transported = json.loads(bridge.prompt(keys))
        assert {v["key"]: v["body"] for v in transported["cards"]} == bridge.cards
        for channel, spec in specs.items():
            original = json.loads(Path(spec["path"]).read_text())
            for card in original["cards"]:
                assert bridge.cards[channel + ":" + card["id"]] == card
        assert bridge.grep([".*THIS_IS_LITERAL.*"]) == []
        for name in names:
            altered = {k: dict(v) for k, v in specs.items()}
            altered[name]["sha256"] = "0" * 64
            try:
                MemoryBridge(altered, task, data_hash)
            except ValueError:
                pass
            else:
                raise AssertionError("tampered hash accepted")
        try:
            MemoryBridge(specs, task, "0" * 64)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong split accepted")
        checks[task] = {"cards": len(keys), "full_json_roundtrip": True,
                       "all_source_fields_preserved": True, "bad_hash_rejected": True,
                       "wrong_data_manifest_rejected": True, "literal_query_check": True}
    result = {"status": "OFFLINE_AUDIT_AND_TRANSPORT_ONLY", "upstream_commit": inputs["upstream_commit"],
              "upstream_findings": findings, "transport_checks": checks,
              "actual_agent_probe": False, "training_launched": False, "memory_benefit_measured": False}
    (ROOT / "VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
