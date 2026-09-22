#!/usr/bin/env python3
"""Lossless trial evidence export, independent of the frozen training process."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "mlevolve_trial_memory_evidence_v1"

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def configuration_diff(old,new,path=""):
    if isinstance(old,dict) and isinstance(new,dict):
        result=[]
        for key in sorted(set(old)|set(new)):
            where=path+"/"+key.replace("~","~0").replace("/","~1")
            if key not in old:result.append({"path":where,"operation":"add","after":new[key]})
            elif key not in new:result.append({"path":where,"operation":"remove","before":old[key]})
            else:result.extend(configuration_diff(old[key],new[key],where))
        return result
    return [] if digest(old)==digest(new) else [{"path":path or "/","operation":"replace","before":old,"after":new}]

def export(run_dir, *, task="spooky-author-identification", launch=None):
    run = Path(run_dir).resolve(strict=True)
    sources = []
    def read(path):
        path = Path(path)
        raw = path.read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)
    state = read(run / "logs/incumbent_search_state.json")
    journal = read(run / "logs/journal.json")
    identity = read(run / "logs/run_identity.json")
    launch = read(launch) if launch else {}
    nodes = {n["id"]: n for n in journal["nodes"]}
    rows = state["iterations"]
    by_id = {n["node_id"]: n for n in rows}
    assert len(by_id) == len(rows), "duplicate trial identity"
    records = []
    graph_nodes = []
    for row in rows:
        node_id = row["node_id"]
        node = nodes[node_id]
        code_hash = hashlib.sha256((node.get("code") or "").encode()).hexdigest()
        assert code_hash == row["code_sha256"], "journal/state snapshot mismatch; retry export"
        config = row.get("configuration") or {}
        if config: assert digest(config) == row["configuration_sha256"]
        parent = by_id.get(row.get("proposal_base_id"))
        if row.get("proposal_base_id"): assert parent is not None
        lineage_file = run / "logs/node_configurations" / (node_id + ".json")
        lineage = read(lineage_file) if lineage_file.exists() else None
        if lineage:
            assert lineage["source_sha256"] == code_hash
            assert lineage["configuration_sha256"] == row["configuration_sha256"]
        checkpoint_root = run / "workspace/working/budget_checkpoints" / node_id
        receipt = None
        checkpoint_verified = False
        current = checkpoint_root / "CURRENT.json"
        if current.exists():
            name = read(current)["bundle"]
            assert Path(name).name == name
            bundle = checkpoint_root / name
            receipt = read(bundle / "receipt.json")
            for key, expected in (("node_id", node_id), ("code_sha256", code_hash),
                                  ("configuration_sha256", row.get("configuration_sha256"))):
                assert receipt[key] == expected, "checkpoint identity mismatch"
            for name, expected in receipt["files"].items():
                relative = Path(name)
                assert not relative.is_absolute() and ".." not in relative.parts
                file = bundle / relative
                assert not file.is_symlink() and file_hash(file) == expected, "checkpoint file mismatch"
            checkpoint_verified = True
        events_file = checkpoint_root / "training_events.jsonl"
        events = []
        if events_file.exists():
            sources.append({"path": str(events_file), "sha256": file_hash(events_file)})
            events = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
        parent_valid = bool(parent and parent.get("is_valid") and finite(parent.get("metric")))
        valid = bool(row.get("is_valid") and finite(row.get("metric")))
        delta = row["metric"] - parent["metric"] if valid and parent_valid else None
        changes = row.get("configuration_changes", [])
        if parent and config and parent.get("configuration"):
            assert changes == configuration_diff(parent["configuration"],config), "configuration diff does not match before/after recipes"
        architecture_changed = bool(parent and config.get("architecture") != parent.get("configuration", {}).get("architecture"))
        kind = ("debug_repair" if row.get("stage") == "debug" else
                "initial_recipe" if not parent else "architecture_switch" if architecture_changed else
                "parameter_trial" if changes else "implementation_change_or_replication")
        intervention = row.get("intervention")
        repairs = row.get("review_repairs", [])
        if intervention:
            assert intervention["source_sha256"] == code_hash
            parent_hash = parent["code_sha256"] if parent else hashlib.sha256(b"").hexdigest()
            assert intervention["parent_source_sha256"] == parent_hash
            assert intervention["changes"] == changes
            kind = intervention["kind"]
            architecture_changed = kind == "architecture_switch"
        for repair in repairs:
            assert hashlib.sha256(repair["before_source"].encode()).hexdigest() == repair["before_source_sha256"]
            assert hashlib.sha256(repair["after_source"].encode()).hexdigest() == repair["after_source_sha256"]
        if repairs:
            assert repairs[-1]["after_source_sha256"] == code_hash, "repair chain does not end at executed source"
            assert all(a["after_source_sha256"]==b["before_source_sha256"] for a,b in zip(repairs,repairs[1:])), "broken repair chain"
        comparison = "unscored" if delta is None else "improved" if delta < 0 else "regressed" if delta > 0 else "tied"
        audit = node.get("leakage_audit") or {}
        verified_score = bool(valid and checkpoint_verified and audit.get("status") == "clean"
                              and audit.get("rank_eligible") is True and audit.get("metric_disposition") == "accept"
                              and audit.get("code_sha256") == code_hash
                              and audit.get("contract_sha256") == receipt.get("contract_sha256"))
        record = {
            "trial_id": run.name + "/" + node_id, "task_id": task, "run_id": run.name,
            "node_id": node_id, "iteration": row["iteration"], "kind": kind,
            "stage": row.get("stage"), "source_parent_id": row.get("proposal_base_id"),
            "previous_attempt_id": row.get("previous_attempt_id"), "branch_id": row.get("branch_id"),
            "intervention": intervention, "review_repairs": repairs,
            "execution_status": row.get("execution_status", "evaluated" if valid else "unscored"),
            "operational_changes": intervention.get("operational_changes", []) if intervention else changes,
            "declaration_corrections": intervention.get("declaration_corrections", []) if intervention else [],
            "configuration_before": parent.get("configuration") if parent else None,
            "configuration_after": config or None, "configuration_changes": changes,
            "configuration_sha256": row.get("configuration_sha256"), "code_sha256": code_hash,
            "metric_name": "fixed_search_log_loss", "metric_direction": "minimize", "official_score": False,
            "metric_before": parent["metric"] if parent_valid else None,
            "metric_after": row["metric"] if valid else None, "metric_delta": delta,
            "comparison": comparison, "decision": row["decision"], "best_so_far": row.get("best_so_far"),
            "is_valid": valid, "is_buggy": row.get("is_buggy"), "verified_score": verified_score,
            "checkpoint_files_verified": checkpoint_verified, "checkpoint_receipt": receipt,
            "data_manifest_sha256": (receipt or launch).get("data_manifest_sha256"),
            "contract_sha256": (receipt or launch).get("contract_sha256"),
            "runtime_manifest_sha256": launch.get("runtime_manifest_sha256"),
            "node_wall_seconds": row.get("node_wall_seconds"), "exec_time_seconds": row.get("exec_time_seconds"),
            "proposal_reason": row.get("proposal_reason"), "feedback": row.get("feedback"),
            "exception_type": row.get("exception_type"), "terminal_output": node.get("_term_out"),
            "leakage_audit": audit, "configuration_receipt": lineage, "training_events": events,
            "attribution": "joint_change_observation" if delta is not None and intervention and (intervention.get("implementation_repair") or architecture_changed)
                           else "no_parameter_effect_claim" if intervention and not intervention.get("operational_changes")
                           else "observed_single_field_difference" if delta is not None and len(intervention.get("operational_changes", [] ) if intervention else changes) == 1 and not architecture_changed
                           else "joint_change_observation" if delta is not None and changes else "no_parameter_effect_claim",
            "limitations": ["single fixed holdout; no general causal or cross-task claim",
                            "engineering failure or missing checkpoint does not establish a bad hyperparameter"],
        }
        records.append(record)
        graph_nodes.append({"id": node_id, "type": "RunNode", "task": task, "run_id": run.name,
                            "stage": row.get("stage"), "step": row["iteration"], "metric": record["metric_after"],
                            "is_valid": valid, "is_buggy": row.get("is_buggy"), "leakage_audit": audit,
                            "code_sha256": code_hash, "code": node.get("code"), "plan": row.get("proposal_reason"),
                            "analysis": node.get("analysis"), "terminal_excerpt": node.get("_term_out"),
                            "trial_id": record["trial_id"]})
        if parent:
            graph_nodes.append({"id": run.name + "/transition/" + node_id, "type": "Transition", "task": task,
                                "run_id": run.name, "parent_node_id": parent["node_id"], "child_node_id": node_id,
                                "parent_buggy": parent.get("is_buggy"), "child_buggy": row.get("is_buggy"),
                                "stage_pair": str(parent.get("stage")) + "->" + str(row.get("stage")),
                                "outcome": "debug_fixed" if kind == "debug_repair" and parent.get("is_buggy") and valid
                                           else "metric_improved" if comparison == "improved" else comparison,
                                "text": row.get("proposal_reason"), "trial_id": record["trial_id"],
                                "configuration_changes": changes, "metric_delta": delta})
    payload = {"schema": SCHEMA, "run_identity": identity, "launch": launch,
               "source_artifacts": sources, "source_state_updated_at": state.get("updated_at_unix"),
               "generation_failures_without_trial": state.get("proposal_failures_without_child", 0),
               "records": records, "trial_count": len(records), "graph": {"nodes": graph_nodes}}
    payload["payload_sha256"] = digest(payload)
    return payload

def validate_package(package):
    assert package.get("schema") == SCHEMA
    assert package.get("payload_sha256") == digest({k: v for k, v in package.items() if k != "payload_sha256"})
    records = package["records"]
    assert len(records) == package["trial_count"] == len({r["trial_id"] for r in records})
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = export(args.run_dir, launch=args.launch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(args.output)
    print(json.dumps({"trial_count": payload["trial_count"], "sha256": payload["payload_sha256"]}))

if __name__ == "__main__": main()
