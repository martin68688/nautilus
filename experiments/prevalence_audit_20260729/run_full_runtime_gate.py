#!/usr/bin/env python3
"""Run deterministic online Host full-runtime gates for the two image tasks.

This is a pre-formal infrastructure gate, not an experimental condition.  It
uses the production Interpreter, Host SDK, signed Collector journal, receipt
bridge, and AuthorityEngine while avoiding stochastic LLM architecture errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import textwrap
import traceback
from types import SimpleNamespace
from typing import Any
import uuid


EXPECTED_TASKS = {
    "denoising-dirty-documents": ("rmse", "minimize"),
    "aerial-cactus-identification": ("roc_auc", "maximize"),
}

REQUIRED_RECEIPTS = {
    "code_execution",
    "method_identity",
    "split_lineage",
    "fit_scope",
    "prediction_scope",
    "evaluator",
    "selection_freeze",
}

METRIC_PREFIX = "HOST_GATE_METRIC_JSON="


def _denoising_source() -> str:
    return textwrap.dedent(
        r'''
        import json
        import numpy as np
        from PIL import Image
        import torch
        from protocol_runtime import current_session

        def _training_pixels(rows, image_limit):
            noisy_values = []
            target_values = []
            for row in rows[:image_limit]:
                noisy = np.asarray(Image.open(row["assets"]["noisy"]), dtype=np.float32) / 255.0
                target = np.asarray(Image.open(row["assets"]["target"]), dtype=np.float32) / 255.0
                if noisy.shape != target.shape:
                    raise ValueError("Host train noisy/target shapes differ")
                noisy = noisy.reshape(-1)
                target = target.reshape(-1)
                stride = max(1, noisy.size // 32768)
                noisy_values.append(noisy[::stride])
                target_values.append(target[::stride])
            return np.concatenate(noisy_values), np.concatenate(target_values)

        def _fit_affine(rows, steps, image_limit):
            torch.manual_seed(20260729)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            noisy, target = _training_pixels(rows, image_limit)
            x = torch.from_numpy(noisy).to(device=device).reshape(-1, 1)
            y = torch.from_numpy(target).to(device=device).reshape(-1, 1)
            model = torch.nn.Linear(1, 1, bias=True).to(device)
            with torch.no_grad():
                model.weight.fill_(1.0)
                model.bias.zero_()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(x), y)
                loss.backward()
                optimizer.step()
            return model, device

        def _predict_images(model, device, rows):
            predictions = []
            model.eval()
            with torch.no_grad():
                for row in rows:
                    noisy = np.asarray(Image.open(row["assets"]["noisy"]), dtype=np.float32) / 255.0
                    shape = noisy.shape
                    values = torch.from_numpy(noisy.reshape(-1, 1)).to(device=device)
                    prediction = model(values).clamp(0.0, 1.0).cpu().numpy().reshape(shape)
                    predictions.append((row["sample_id"], prediction))
            return predictions

        def candidate(session):
            views = session.get_split()
            with session.fit_scope(component="deterministic_affine_preflight", data_view=views.train) as train_rows:
                model, device = _fit_affine(train_rows, 2, 4)
            with session.prediction_scope(component="deterministic_affine_preflight", data_view=views.validation) as validation_rows:
                predictions = _predict_images(model, device, validation_rows)
            session.evaluate_internal(views.validation, predictions, label_key="target")
            session.freeze_selection("denoising-preflight", based_on=views.validation, artifact_hash="a" * 64)
            with session.inference_scope(component="denoising-preflight-submission", data_view=views.inference) as inference_rows:
                _predict_images(model, device, inference_rows)

        def main():
            session = current_session()
            views = session.get_split()
            with session.fit_scope(component="deterministic_affine_online", data_view=views.train) as train_rows:
                model, device = _fit_affine(train_rows, 24, 24)
            with session.prediction_scope(component="deterministic_affine_online", data_view=views.validation) as validation_rows:
                predictions = _predict_images(model, device, validation_rows)
            score = session.evaluate_internal(views.validation, predictions, label_key="target")
            checkpoint = "denoising-full-runtime-gate.pt"
            torch.save(model.state_dict(), checkpoint)
            session.freeze_selection(checkpoint, based_on=views.validation, artifact_hash=checkpoint)
            with session.inference_scope(component="denoising-final-submission", data_view=views.inference) as inference_rows:
                _predict_images(model, device, inference_rows)
            print("HOST_GATE_METRIC_JSON=" + json.dumps({
                "metric_name": "rmse",
                "metric_value": float(score),
                "device": str(device),
                "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
            }, sort_keys=True))

        if __name__ == "__main__":
            main()
        '''
    ).lstrip()


def _aerial_source() -> str:
    return textwrap.dedent(
        r'''
        import json
        import numpy as np
        from PIL import Image
        import torch
        from protocol_runtime import current_session

        def _image_features(rows, row_limit=None):
            selected = rows if row_limit is None else rows[:row_limit]
            features = []
            labels = []
            for row in selected:
                image = np.asarray(Image.open(row["assets"]["image"]).convert("RGB"), dtype=np.float32) / 255.0
                flat = image.reshape(-1, 3)
                means = flat.mean(axis=0)
                stds = flat.std(axis=0)
                green_excess = np.asarray([means[1] - 0.5 * (means[0] + means[2])], dtype=np.float32)
                brightness = np.asarray([means.mean()], dtype=np.float32)
                features.append(np.concatenate([means, stds, green_excess, brightness]))
                if "label" in row:
                    labels.append(float(row["label"]))
            return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)

        def _fit_logistic(rows, steps, row_limit):
            torch.manual_seed(20260729)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            features, labels = _image_features(rows, row_limit)
            x = torch.from_numpy(features).to(device=device)
            y = torch.from_numpy(labels).to(device=device).reshape(-1, 1)
            model = torch.nn.Linear(x.shape[1], 1).to(device)
            positives = y.sum().clamp(min=1.0)
            negatives = (y.numel() - y.sum()).clamp(min=1.0)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=(negatives / positives))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            return model, device

        def _predict_probabilities(model, device, rows):
            features, _ = _image_features(rows)
            x = torch.from_numpy(features).to(device=device)
            model.eval()
            with torch.no_grad():
                return torch.sigmoid(model(x)).reshape(-1).cpu().numpy()

        def candidate(session):
            views = session.get_split()
            with session.fit_scope(component="deterministic_aerial_preflight", data_view=views.train) as train_rows:
                model, device = _fit_logistic(train_rows, 8, 1024)
            with session.prediction_scope(component="deterministic_aerial_preflight", data_view=views.validation) as validation_rows:
                predictions = _predict_probabilities(model, device, validation_rows)
            session.evaluate_internal(views.validation, predictions, label_key="label")
            session.freeze_selection("aerial-preflight", based_on=views.validation, artifact_hash="b" * 64)
            with session.inference_scope(component="aerial-preflight-submission", data_view=views.inference) as inference_rows:
                _predict_probabilities(model, device, inference_rows)

        def main():
            session = current_session()
            views = session.get_split()
            with session.fit_scope(component="deterministic_aerial_online", data_view=views.train) as train_rows:
                model, device = _fit_logistic(train_rows, 120, None)
            with session.prediction_scope(component="deterministic_aerial_online", data_view=views.validation) as validation_rows:
                predictions = _predict_probabilities(model, device, validation_rows)
            score = session.evaluate_internal(views.validation, predictions, label_key="label")
            checkpoint = "aerial-full-runtime-gate.pt"
            torch.save(model.state_dict(), checkpoint)
            session.freeze_selection(checkpoint, based_on=views.validation, artifact_hash=checkpoint)
            with session.inference_scope(component="aerial-final-submission", data_view=views.inference) as inference_rows:
                _predict_probabilities(model, device, inference_rows)
            print("HOST_GATE_METRIC_JSON=" + json.dumps({
                "metric_name": "roc_auc",
                "metric_value": float(score),
                "device": str(device),
                "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
            }, sort_keys=True))

        if __name__ == "__main__":
            main()
        '''
    ).lstrip()


def candidate_source(task_id: str) -> str:
    if task_id == "denoising-dirty-documents":
        return _denoising_source()
    if task_id == "aerial-cactus-identification":
        return _aerial_source()
    raise ValueError(f"No deterministic full-runtime gate for {task_id}")


def _metric_payload(output: list[str]) -> dict[str, Any]:
    matches = [line.strip()[len(METRIC_PREFIX) :] for line in output if line.strip().startswith(METRIC_PREFIX)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {METRIC_PREFIX} marker; found {len(matches)}")
    payload = json.loads(matches[0])
    if not isinstance(payload.get("metric_value"), (int, float)):
        raise ValueError("Candidate metric marker lacks a numeric metric_value")
    return payload


def _authority_decisions(
    *, task_id: str, node: Any, contract: Any, registry_root: Path
) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    from authority.adapters.mlevolve.receipt_bridge import receipts_for_node
    from authority.authority_engine import AuthorityEngine
    from authority.collectors import TrustedCollectorHost
    from authority.evidence_graph import EvidenceGraph, EvidencePath
    from authority.models import (
        AuthorityRequest,
        Claim,
        ClaimType,
        DecisionStage,
        Operation,
        TaskContext,
    )
    from authority.protocol_registry import ProtocolRegistry

    receipts = receipts_for_node(
        node,
        contract.protocol_ref,
        f"online-full-runtime-gate:{task_id}",
        collector_host=TrustedCollectorHost(f"online-full-runtime-gate:{task_id}"),
        task_id=task_id,
    )
    receipt_types = {receipt.receipt_type.value for receipt in receipts}
    missing = sorted(REQUIRED_RECEIPTS - receipt_types)
    if missing:
        raise ValueError(f"Trusted receipt bridge is incomplete: {missing}")

    claim = Claim(
        claim_id=f"{node.id}:host-score",
        claim_type=ClaimType.SCORE,
        subject_artifact_id=str(node.id),
        task_scope={"task_id": task_id},
        method_fingerprint=str(node.method_fingerprint),
        protocol_ref=contract.protocol_ref,
        statement="Deterministic online Host full-runtime gate score",
        source_artifact_refs=[str(node.id)],
        evidence_refs=[receipt.receipt_id for receipt in receipts],
    )
    graph = EvidenceGraph()
    graph.add_claim(claim)
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(
        EvidencePath(
            path_id=f"{node.id}:host-path",
            claim_id=claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in receipts],
        )
    )
    engine = AuthorityEngine(ProtocolRegistry(registry_root), graph=graph)
    results: dict[str, dict[str, Any]] = {}
    for operation, stage in (
        (Operation.RANK, DecisionStage.BRANCH_SELECTION),
        (Operation.SELECT, DecisionStage.BRANCH_SELECTION),
        (Operation.PROMOTE_RESULT, DecisionStage.MEMORY_WRITEBACK),
    ):
        decision = engine.authorize(
            AuthorityRequest(
                artifact_id=str(node.id),
                claim_id=claim.claim_id,
                operation=operation,
                decision_stage=stage,
                active_protocol=contract.protocol_ref,
                task_context=TaskContext(task_id, contract.task_family),
                requesting_component="prevalence.full_runtime_online_gate",
            )
        )
        results[operation.value] = {
            "outcome": decision.outcome.value,
            "reason_codes": list(decision.reason_codes),
            "missing_obligations": list(decision.missing_obligations),
            "blocking_receipts": list(decision.blocking_receipts),
            "decision_id": decision.decision_id,
        }
        if not decision.allowed:
            raise ValueError(
                f"Authority {operation.value} did not allow: {results[operation.value]}"
            )
    return receipts, results


def run_task(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    from authority.protocol_execution_contract import read_contract_artifact
    from engine.executor import Interpreter

    expected_metric, expected_direction = EXPECTED_TASKS[task_id]
    bundle = args.bundle_root / task_id
    binding_path = bundle / "HOST_PROTOCOL_BINDING.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    contract_path = Path(binding["contract_path"])
    manifest_path = Path(binding["data_view_manifest_path"])
    contract = read_contract_artifact(contract_path)
    metric_spec = dict(contract.evaluator_spec.get("metric") or {})
    if metric_spec != {"name": expected_metric, "direction": expected_direction, "best_seed_selection": False}:
        raise ValueError(
            f"{task_id} metric/direction mismatch: expected {expected_metric}/{expected_direction}, got {metric_spec}"
        )
    if binding.get("contract_hash") != contract.contract_hash:
        raise ValueError(f"{task_id} binding Contract hash mismatch")

    task_root = (args.output_root / task_id).resolve()
    task_root.mkdir(parents=True, exist_ok=False)
    task_root.chmod(0o755)
    workspace = task_root / "workspace"
    workspace.mkdir(mode=0o755)
    reports = task_root / "reports"
    source = candidate_source(task_id)
    code_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    node_id = f"online-gate-{task_id}-{uuid.uuid4().hex[:12]}"

    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            search=SimpleNamespace(parallel_search_num=1, num_gpus=1),
            protocol_preflight=SimpleNamespace(
                enabled=True,
                report_root=str(reports),
                expected_contract_hash=contract.contract_hash,
                contract_path=str(contract_path),
                data_view_manifest_path=str(manifest_path),
                image_digest=str(binding["image_digest"]),
                sdk_hash=str(binding["sdk_hash"]),
                collector_private_key_path=str(args.collector_key),
                candidate_uid=args.candidate_uid,
                consume_collector_private_key=False,
            ),
        ),
        evaluation_authority=SimpleNamespace(
            mode="enforce",
            protocol_runtime_mode="host_sdk_enforce",
            runtime_protocol_observer_enabled=True,
            protocol_registry=str(args.source_root / "mlevolve" / "config" / "protocols"),
        ),
        start_cpu_id="0",
        cpu_number=str(args.cpu_number),
    )
    result = Interpreter(
        workspace,
        timeout=args.timeout,
        max_parallel_run=1,
        cfg=cfg,
    ).run(source, id=node_id)
    if result.exc_type is not None:
        raise RuntimeError(
            json.dumps(
                {
                    "exc_type": result.exc_type,
                    "exc_info": result.exc_info,
                    "term_out": result.term_out,
                    "protocol_observation": result.protocol_observation,
                },
                indent=2,
                default=str,
            )
        )
    metric = _metric_payload(result.term_out)
    if metric["metric_name"] != expected_metric:
        raise ValueError(f"{task_id} candidate printed the wrong metric")
    if metric.get("device") != "cuda":
        raise ValueError(f"{task_id} deterministic online gate did not train on CUDA")

    observation = dict(result.protocol_observation or {})
    full_runtime = dict(observation.get("host_full_runtime") or {})
    if full_runtime.get("status") != "pass" or full_runtime.get("missing_events") != []:
        raise ValueError(f"{task_id} Host full-runtime evidence is incomplete: {full_runtime}")
    preflight = dict(observation.get("protocol_preflight") or {})
    if preflight.get("status") != "pass":
        raise ValueError(f"{task_id} Protocol Preflight did not pass: {preflight}")

    node = SimpleNamespace(
        id=node_id,
        code=source,
        code_sha256_expected=code_sha256,
        method_fingerprint=code_sha256,
        metric=SimpleNamespace(
            value=float(metric["metric_value"]),
            maximize=expected_direction == "maximize",
        ),
        exec_time=float(result.exec_time),
        is_buggy=False,
        is_valid=True,
        protocol_observation=observation,
        leakage_audit={},
        protocol_repair={},
    )
    receipts, decisions = _authority_decisions(
        task_id=task_id,
        node=node,
        contract=contract,
        registry_root=args.source_root / "mlevolve" / "config" / "protocols",
    )
    return {
        "status": "pass",
        "task_id": task_id,
        "node_id": node_id,
        "code_sha256": code_sha256,
        "contract_hash": contract.contract_hash,
        "protocol_ref": contract.protocol_ref.__dict__,
        "metric": metric,
        "preflight": {
            "status": preflight.get("status"),
            "report_hash": preflight.get("report_hash"),
            "missing_receipts": preflight.get("missing_receipts"),
            "candidate_uid_isolated": preflight.get("candidate_uid_isolated"),
        },
        "host_full_runtime": full_runtime,
        "trusted_receipts": [
            {
                "receipt_id": receipt.receipt_id,
                "receipt_type": receipt.receipt_type.value,
                "trust_status": receipt.trust_status,
                "payload_hash": receipt.payload_hash,
            }
            for receipt in receipts
        ],
        "authority": decisions,
        "term_out": result.term_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--collector-key", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=sorted(EXPECTED_TASKS), default=list(EXPECTED_TASKS))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--cpu-number", type=int, default=8)
    parser.add_argument("--candidate-uid", type=int, default=65534)
    args = parser.parse_args()

    args.source_root = args.source_root.resolve(strict=True)
    args.bundle_root = args.bundle_root.resolve(strict=True)
    args.collector_key = args.collector_key.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        raise ValueError(f"Refusing to reuse online gate output root: {args.output_root}")
    args.output_root.mkdir(parents=True, mode=0o755)
    sys.path.insert(0, str(args.source_root / "mlevolve"))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The online full-runtime gate requires an allocated CUDA GPU")
    report: dict[str, Any] = {
        "schema": "mlevolve_prevalence_full_runtime_online_gate_v1",
        "source_root": str(args.source_root),
        "bundle_root": str(args.bundle_root),
        "gpu": {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
        },
        "tasks": {},
        "status": "pass",
    }
    for task_id in args.tasks:
        try:
            report["tasks"][task_id] = run_task(args, task_id)
        except Exception as error:
            report["status"] = "fail"
            report["tasks"][task_id] = {
                "status": "fail",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
    report_path = args.output_root / "ONLINE_FULL_RUNTIME_GATE.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
