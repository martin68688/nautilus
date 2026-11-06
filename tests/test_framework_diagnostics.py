"""Framework-level fixes: component health, method-family quota, Host selection."""

import json
import logging
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))
sys.path.append(str(REPO / "tests"))

from config import OmegaConf, _load_cfg, Config  # noqa: E402
from engine import component_health, method_family  # noqa: E402
from engine.executor import Interpreter  # noqa: E402
from fixed_holdout import portfolio  # noqa: E402
from fixed_holdout.numerai_source import MEMORY_NOTE, SPEED_RULE, build_source  # noqa: E402
from fixed_holdout.portfolio import choose_members  # noqa: E402
from fixed_holdout.prepare import prepare_task  # noqa: E402
from fixed_holdout.selection import (  # noqa: E402
    apply_to_node,
    noise_verdict,
    paired_noise_band,
    predictions_path,
    score_selection,
)
from utils.metric import MetricValue  # noqa: E402
from test_fixed_holdout_numerai import _write_raw_numerai  # noqa: E402

# Lines taken from the 2026-10-23 Numerai medium run (EraMLP was silently dropped).
NAN_LOG = """Training EraMLP ...
EraMLP epoch 1/8 loss=nan val_corr=-0.00693
EraMLP epoch 2/8 loss=nan val_corr=-0.00693
EraMLP val_corr=-0.00693
LGBM seed=42 best_iter=1085 val_corr=0.02643
ensemble w_lgbm=0.8 w_mlp=0.2 val_corr=0.02600
Final Validation Score: 0.0264
"""
CONSTANT_LOG = "\n".join(
    f"EraMLP epoch {i}/8 loss=0.0000 val_corr=-0.0038" for i in range(1, 6)
) + "\nXGB seed=42 best_iter=1766 val_corr=0.02815\nneutralize mode=none k=0 val_corr=-0.001\n"
FALLBACK_LOG = "EraMLP training failed (shape mismatch); falling back to LGBM only\nLGBM val_corr=0.026\n"
HEALTHY_LOG = "\n".join(
    f"EraMLP epoch {i}/4 loss={0.9 - 0.1 * i:.4f} val_corr={0.004 * i:.4f}" for i in range(1, 5)
) + "\nLGBM val_corr=0.026\n"


def _kinds(issues):
    return {(issue["component"], issue["kind"]) for issue in issues}


def test_detects_nan_loss_and_below_trivial_component_but_not_aggregates():
    issues = component_health.detect_component_issues(NAN_LOG, trivial_score=0.0)
    assert _kinds(issues) == {
        ("EraMLP", "non_finite_loss"),
        ("EraMLP", "below_trivial_score"),
    }


def test_detects_constant_loss_and_silent_fallback():
    assert ("EraMLP", "constant_loss") in _kinds(
        component_health.detect_component_issues(CONSTANT_LOG)
    )
    assert _kinds(component_health.detect_component_issues(FALLBACK_LOG)) == {
        ("EraMLP", "silent_fallback")
    }
    assert component_health.detect_component_issues(HEALTHY_LOG, trivial_score=0.0) == []


def _agent(**cfg):
    defaults = dict(enabled=True, trivial_score=0.0, max_repairs_per_run=6, max_repairs_per_branch=2)
    return SimpleNamespace(
        cfg=SimpleNamespace(component_health=SimpleNamespace(**{**defaults, **cfg})),
        journal=[],
    )


def _node(term_out, **kwargs):
    fields = dict(
        id="n1", is_buggy=False, component_issues=[], component_repair_target={},
        component_repair_status="", branch_id=1, term_out=term_out,
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def test_repair_is_scheduled_once_and_judged_by_component_health():
    agent = _agent()
    parent = _node(NAN_LOG)
    component_health.record(agent, parent)
    assert component_health.needs_repair(agent, parent)
    parent.component_repair_status = "dispatched"
    assert not component_health.needs_repair(agent, parent)

    target = {"issues": parent.component_issues}
    fixed = _node(HEALTHY_LOG, id="n2", component_repair_target=target)
    component_health.record(agent, fixed)
    assert fixed.component_repair_status == "resolved"

    still_broken = _node(CONSTANT_LOG + "EraMLP epoch 9/9 loss=nan\n", id="n3", component_repair_target=target)
    component_health.record(agent, still_broken)
    assert still_broken.component_repair_status == "unresolved"

    crashed = _node("", id="n4", is_buggy=True, component_repair_target=target)
    component_health.record(agent, crashed)
    assert crashed.component_repair_status == "failed"

    agent.journal = [SimpleNamespace(component_repair_target={"x": 1}, branch_id=1)] * 2
    fresh = _node(NAN_LOG, id="n5")
    component_health.record(agent, fresh)
    assert not component_health.needs_repair(agent, fresh)  # branch budget used up


def test_repair_brief_reports_observations_not_a_solution():
    lines = component_health.repair_instructions(
        component_health.detect_component_issues(NAN_LOG, trivial_score=0.0)
    )
    text = "\n".join(lines)
    assert "EraMLP" in text and "loss=nan" in text
    assert "instead of removing it" in text


def test_method_family_tags_follow_code_and_runtime_removal():
    code = "import lightgbm as lgb\nimport torch\nclass EraMLP(torch.nn.Module): pass\n"
    assert method_family.classify(code) == ["neural_net", "gbdt"]
    assert method_family.classify(code, "EraMLP removed (broken model)") == ["gbdt"]


def _family_node(node_id, metric, families, branch=1):
    return SimpleNamespace(
        id=node_id, is_buggy=False, metric=MetricValue(metric, maximize=True),
        method_families=families, is_terminal=False, branch_id=branch,
        reached_child_limit=lambda scfg, for_topk=False: False,
    )


def test_family_quota_routes_to_best_node_of_missing_family(monkeypatch):
    trees = [_family_node(f"t{i}", 0.03 + i * 0.001, ["gbdt"]) for i in range(3)]
    nets = [_family_node("nn1", 0.012, ["neural_net", "gbdt"], 2), _family_node("nn2", 0.015, ["neural_net"], 2)]
    agent = SimpleNamespace(
        cfg=SimpleNamespace(family_quota=SimpleNamespace(enabled=True, families=["neural_net", "gbdt"], probability=1.0)),
        metric_maximize=True,
        branch_all_nodes={1: trees, 2: nets},
        scfg=None,
    )
    assert method_family.pick_underrepresented(agent, trees).id == "nn2"
    assert method_family.pick_underrepresented(agent, trees + [nets[1]]) is None
    agent.cfg.family_quota.probability = 0.0
    assert method_family.pick_underrepresented(agent, trees) is None


def _selection_split(tmp_path):
    train_path, validation_path, features_path = _write_raw_numerai(tmp_path, n_eras=48)
    prepared = build_source(
        train_path, validation_path, features_path, tmp_path / "datasets",
        lockbox_eras=6, embargo_eras=2, selection_eras=8, train_era_stride=2,
    )
    split = prepare_task(tmp_path / "datasets", "numerai-v6", tmp_path / "fixed", copy_mode="copy")
    return prepared, split


def _selection_and_lockbox(split):
    test = pd.read_parquet(split / "train_view" / "input" / "test.parquet")
    selection_eras = set(pd.read_csv(split / "selection_view" / "labels.csv")["era"].astype(str))
    in_selection = test["era"].isin(selection_eras)
    return test, test[in_selection], test[~in_selection]


def test_selection_set_is_carved_out_with_embargo_and_hidden_labels(tmp_path):
    prepared, split = _selection_split(tmp_path)
    public = split / "train_view" / "input"
    train = pd.read_parquet(public / "train.parquet")
    test, selection, lockbox = _selection_and_lockbox(split)
    assert not (public / "selection.parquet").exists()
    assert "target" not in test.columns
    assert train["era"].max() < selection["era"].min() < selection["era"].max() < lockbox["era"].min()
    assert selection["era"].nunique() == 8 and lockbox["era"].nunique() == 6
    assert list(test.index) == list(selection.index) + list(lockbox.index)  # selection rows first
    assert not list((split / "train_view").rglob("labels.csv"))
    view = split / "selection_view"
    manifest = json.loads((view / "fixed_holdout_manifest.json").read_text())
    assert manifest["role"] == "selection_view" and manifest["row_count"] == len(selection)
    assert manifest["predictions_source"] == "submission"
    train_manifest = json.loads((split / "train_view" / "fixed_holdout_manifest.json").read_text())
    evaluator = json.loads((split / "evaluator_view" / "fixed_holdout_manifest.json").read_text())
    assert train_manifest["row_count"] == evaluator["submission_row_count"] == len(test)
    assert evaluator["row_count"] == len(lockbox)
    assert split.name.endswith(evaluator["holdout_id_sha256"][:12])  # lockbox IDs name the split
    description = (public / "description.md").read_text()
    assert "selection" not in description and "part of `test.parquet`" in description
    assert "only features of that era and of earlier" in description
    assert "only every 2th weekly era is kept" in description


def test_host_selection_score_replaces_self_reported_metric(tmp_path):
    _, split = _selection_split(tmp_path)
    test, _, _ = _selection_and_lockbox(split)
    manifest = split / "selection_view" / "fixed_holdout_manifest.json"
    preds = test[["id"]].assign(prediction=test["feature_a"] + 1e-3 * np.arange(len(test)))

    workspace = tmp_path / "ws"
    node = SimpleNamespace(id="abc", is_buggy=False, metric=MetricValue(0.9, maximize=True),
                           host_selection={}, _term_out=[], analysis="")
    agent = SimpleNamespace(cfg=SimpleNamespace(
        workspace_dir=workspace,
        host_selection=SimpleNamespace(enabled=True, manifest_path=str(manifest), windows=4,
                                       objective="mean", std_penalty=0.5),
    ))
    path = predictions_path(workspace, node.id, manifest)
    assert path.name == "submission_abc.csv"  # selection rows come from the node's submission
    path.parent.mkdir(parents=True)
    preds.to_csv(path, index=False)
    apply_to_node(agent, node, logging.getLogger("test"))
    assert node.is_buggy is False
    assert node.host_selection["agent_reported_metric"] == 0.9
    assert len(node.host_selection["window_means"]) == 4
    assert node.metric.value == pytest.approx(np.mean(node.host_selection["window_means"]))
    assert node.metric.value > 0.1

    strict = score_selection(manifest, path, windows=4, objective="mean_minus_std", std_penalty=1.0)
    assert strict["score"] <= node.metric.value

    path.unlink()
    missing = SimpleNamespace(id="abc", is_buggy=False, metric=MetricValue(0.9, maximize=True),
                              host_selection={}, _term_out=[], analysis="")
    apply_to_node(agent, missing, logging.getLogger("test"))
    assert missing.is_buggy is True and "HOST_SELECTION_ERROR" in missing.analysis


def test_selection_predictions_are_isolated_per_node():
    code = "pd.DataFrame(x).to_csv('./submission/selection_predictions.csv', index=False)"
    isolated = Interpreter.isolate_submission_path(None, code, "node42")
    assert "selection_predictions_node42.csv" in isolated


def test_framework_config_resolves_and_validates_against_schema():
    cfg = _load_cfg(REPO / "mlevolve" / "config" / "config_numerai_framework.yaml", use_cli_args=False)
    cfg.exp_name = "numerai-v6-test"  # normally filled in by prep_cfg
    merged = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    assert merged.host_selection.enabled is True
    assert merged.component_health.trivial_score == 0.0
    assert merged.family_quota.families == ["neural_net", "gbdt"]
    assert merged.fixed_holdout.enabled is True
    assert merged.agent.use_global_memory is True


def test_refit_view_adds_labelled_selection_and_embargo_eras(tmp_path):
    _, split = _selection_split(tmp_path)
    public = split / "train_view" / "input"
    refit_input = split / "refit_view" / "input"
    train = pd.read_parquet(public / "train.parquet")
    refit = pd.read_parquet(refit_input / "train.parquet")
    _, selection, lockbox = _selection_and_lockbox(split)
    assert refit["era"].max() >= selection["era"].max()
    assert refit["era"].max() < lockbox["era"].min()
    assert set(train["era"]) < set(refit["era"])
    assert refit["target"].notna().all()
    assert (refit_input / "test.parquet").read_bytes() == (public / "test.parquet").read_bytes()


def test_refit_runner_retrains_selected_program_on_refit_view(tmp_path):
    from fixed_holdout.refit import refit_node

    _, split = _selection_split(tmp_path)
    program = (
        "import pandas as pd\n"
        "train = pd.read_parquet('./input/train.parquet')\n"
        "test = pd.read_parquet('./input/test.parquet')\n"
        "print('latest training era', train['era'].max())\n"
        "pd.DataFrame({'id': test['id'], 'prediction': float(train['era'].nunique())})"
        ".to_csv('./submission/submission.csv', index=False)\n"
    )
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "journal.json").write_text(json.dumps({"nodes": [{"id": "abc123", "code": program}]}))
    (run / "logs" / "fixed_holdout_evaluation_request.json").write_text(json.dumps({"selected_node_id": "abc123"}))

    receipt = refit_node(split, run, run / "refit", python=sys.executable, timeout=120)
    out = pd.read_csv(run / "refit" / receipt["submission"])
    refit_eras = pd.read_parquet(split / "refit_view" / "input" / "train.parquet")["era"].nunique()
    assert receipt["node_id"] == "abc123"
    assert out["prediction"].iloc[0] == refit_eras


def test_bmc_matches_numerai_tools_correlation_contribution():
    scoring = pytest.importorskip("numerai_tools.scoring")
    from fixed_holdout.evaluate import _benchmark_contribution

    rng = np.random.default_rng(3)
    benchmark = rng.normal(size=400)
    predictions = 0.6 * benchmark + rng.normal(size=400)
    target = np.clip(np.round((0.3 * predictions + rng.normal(size=400)) * 2) / 4 + 0.5, 0, 1)
    index = pd.RangeIndex(400)
    expected = scoring.correlation_contribution(
        pd.DataFrame({"p": predictions}, index=index), pd.Series(benchmark, index=index),
        pd.Series(target, index=index),
    )["p"]
    assert _benchmark_contribution(predictions, benchmark, target) == pytest.approx(expected, abs=1e-12)


def _benchmark_split(tmp_path):
    train_path, validation_path, features_path = _write_raw_numerai(tmp_path, n_eras=48)
    validation = pd.read_parquet(validation_path)
    bench_path = tmp_path / "raw" / "validation_benchmark_models.parquet"
    rng = np.random.default_rng(5)
    validation[["numerai_ticker", "date"]].assign(
        lgbm_jupiter_60=rng.uniform(size=len(validation))
    ).to_parquet(bench_path, index=False)
    build_source(
        train_path, validation_path, features_path, tmp_path / "datasets",
        lockbox_eras=6, embargo_eras=2, selection_eras=8, train_era_stride=2,
        benchmarks_path=bench_path,
    )
    return prepare_task(tmp_path / "datasets", "numerai-v6", tmp_path / "fixed", copy_mode="copy")


def test_benchmark_stays_host_only_and_tickers_are_explained(tmp_path):
    split = _benchmark_split(tmp_path)
    public = split / "train_view" / "input"
    for name in ("train.parquet", "test.parquet"):
        columns = pd.read_parquet(public / name).columns
        assert "benchmark" not in columns and "numerai_ticker" in columns
    assert "benchmark" not in pd.read_parquet(split / "refit_view" / "input" / "train.parquet").columns
    assert "real stock identifier" in (public / "description.md").read_text()
    evaluator = json.loads((split / "evaluator_view" / "fixed_holdout_manifest.json").read_text())
    selection = json.loads((split / "selection_view" / "fixed_holdout_manifest.json").read_text())
    assert evaluator["benchmark_column"] == selection["benchmark_column"] == "benchmark"
    assert "benchmark_column" in json.loads((split / "train_view" / "fixed_holdout_manifest.json").read_text())


def test_lockbox_and_selection_report_bmc(tmp_path):
    from fixed_holdout.evaluate import evaluate_submission

    split = _benchmark_split(tmp_path)
    evaluator = split / "evaluator_view" / "fixed_holdout_manifest.json"
    labels = pd.read_csv(split / "evaluator_view" / "labels.csv")
    test = pd.read_parquet(split / "train_view" / "input" / "test.parquet")

    # One submission covers selection + lockbox rows; the evaluator scores the lockbox rows.
    copy_of_benchmark = tmp_path / "bench_copy.csv"
    bench = test[["id"]].merge(labels[["id", "benchmark"]], on="id", how="left")
    test[["id"]].assign(prediction=bench["benchmark"].fillna(0.5).to_numpy()).to_csv(copy_of_benchmark, index=False)
    details = evaluate_submission(evaluator, copy_of_benchmark)["metric_details"]
    assert abs(details["bmc_mean"]) < 1e-9  # nothing beyond the benchmark itself
    assert details["n_eras"] == labels["era"].nunique()

    lockbox_only = tmp_path / "lockbox_only.csv"
    labels[["id"]].assign(prediction=labels["benchmark"]).to_csv(lockbox_only, index=False)
    with pytest.raises(ValueError, match="rows"):
        evaluate_submission(evaluator, lockbox_only)

    informed = tmp_path / "informed.csv"
    test[["id"]].assign(prediction=test["feature_a"].to_numpy() + 1e-3 * np.arange(len(test))).to_csv(informed, index=False)
    details = evaluate_submission(evaluator, informed)["metric_details"]
    assert details["bmc_mean"] > 0.05 and "per_era_bmc" in details

    preds = informed
    manifest = split / "selection_view" / "fixed_holdout_manifest.json"
    by_corr = score_selection(manifest, preds, windows=4)
    by_bmc = score_selection(manifest, preds, windows=4, score_metric="bmc")
    assert by_corr["bmc_mean"] == pytest.approx(by_bmc["bmc_mean"])
    assert by_bmc["score"] == pytest.approx(np.mean(by_bmc["window_means"]))
    assert by_bmc["score"] != pytest.approx(by_corr["score"])


def test_sigkill_is_reported_as_memory_error(tmp_path):
    import signal
    import subprocess

    code = "import os, signal\nprint('training...')\nos.kill(os.getpid(), signal.SIGKILL)\n"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == -signal.SIGKILL
    source = (REPO / "mlevolve" / "engine" / "executor.py").read_text()
    assert "proc.returncode == -signal.SIGKILL" in source and 'exc_type = "MemoryError"' in source


def test_resources_line_appears_in_description(tmp_path):
    train_path, validation_path, features_path = _write_raw_numerai(tmp_path, n_eras=48)
    prepared = build_source(train_path, validation_path, features_path, tmp_path / "d",
                            lockbox_eras=6, embargo_eras=2, resources="24 CPU threads, 38 GB RAM")
    text = (prepared / "public" / "description.md").read_text()
    assert "24 CPU threads, 38 GB RAM" in text and "killed by the system" in text


def test_patcher_drops_stray_markdown_fences_only_when_they_break_the_program():
    from agents.coder.diff_coder.patcher import SearchReplacePatcher

    broken = "import os\nx = 1\n```\nprint(x)\n```python\n"
    fixed = SearchReplacePatcher._strip_stray_fences(broken)
    assert "```" not in fixed
    compile(fixed, "<t>", "exec")
    with_prose = "x = 1\n```\n***Next: add a helper.***\ny = x + 1\n"
    compile(SearchReplacePatcher._strip_stray_fences(with_prose), "<t>", "exec")
    valid = 'doc = """\n```\ncode\n```\n"""\n'
    assert SearchReplacePatcher._strip_stray_fences(valid) == valid


def test_syntax_error_message_shows_the_offending_line():
    source = (REPO / "mlevolve" / "engine" / "executor.py").read_text()
    assert "Show the offending line" in source and "{lines[i - 1]}" in source


def test_reference_files_all_targets_and_no_model_recipe(tmp_path):
    train_path, validation_path, features_path = _write_raw_numerai(tmp_path, n_eras=48)
    validation = pd.read_parquet(validation_path)
    neutralizer = tmp_path / "raw" / "validation_neutralizer.parquet"
    validation[["numerai_ticker", "date"]].assign(neutralizer_column_0=0.1).to_parquet(neutralizer, index=False)
    prepared = build_source(
        train_path, validation_path, features_path, tmp_path / "d",
        lockbox_eras=6, embargo_eras=2, selection_eras=8, extra_targets=("all",),
        reference_files={"neutralizer/part_b.parquet": neutralizer},
    )
    public = prepared / "public"
    assert (public / "neutralizer" / "part_b.parquet").read_bytes() == neutralizer.read_bytes()
    train = pd.read_parquet(public / "train.parquet")
    # "all" keeps every listed target present in the data (the fake data lacks jupiter_60).
    assert {"target", "target_jupiter_20"} <= set(train.columns) and "target_jupiter_60" not in train.columns
    text = (public / "description.md").read_text()
    assert "neutralizer/*.parquet" in text and "benchmark/*.parquet" not in text
    assert "float features may be NaN" in text
    assert "LightGBM" not in text and "n_estimators" not in text
    split = prepare_task(tmp_path / "d", "numerai-v6", tmp_path / "fixed", copy_mode="copy")
    assert (split / "train_view" / "input" / "neutralizer" / "part_b.parquet").exists()
    assert (split / "refit_view" / "input" / "neutralizer" / "part_b.parquet").exists()


def test_corr_bmc_selection_score_and_description(tmp_path):
    train_path, validation_path, features_path = _write_raw_numerai(tmp_path, n_eras=48)
    validation = pd.read_parquet(validation_path)
    bench_path = tmp_path / "raw" / "validation_benchmark_models.parquet"
    validation[["numerai_ticker", "date"]].assign(
        lgbm_jupiter_60=np.random.default_rng(5).uniform(size=len(validation))
    ).to_parquet(bench_path, index=False)
    prepared = build_source(
        train_path, validation_path, features_path, tmp_path / "d",
        lockbox_eras=6, embargo_eras=2, selection_eras=8, benchmarks_path=bench_path,
    )
    text = (prepared / "public" / "description.md").read_text()
    assert "BMC" not in text and "lgbm" not in text and "benchmark/" not in text
    assert "ranks\nsolutions during the search." in text and "beyond" not in text
    assert not (prepared / "public" / "benchmark").exists()
    split = prepare_task(tmp_path / "d", "numerai-v6", tmp_path / "fixed", copy_mode="copy")
    test = pd.read_parquet(split / "train_view" / "input" / "test.parquet")
    preds = tmp_path / "p.csv"
    test[["id"]].assign(prediction=test["feature_a"] + 1e-3 * np.arange(len(test))).to_csv(preds, index=False)
    manifest = split / "selection_view" / "fixed_holdout_manifest.json"
    corr = score_selection(manifest, preds, windows=1)
    bmc = score_selection(manifest, preds, windows=1, score_metric="bmc")
    mix = score_selection(manifest, preds, windows=1, score_metric="corr_bmc")
    assert mix["score"] == pytest.approx(0.5 * corr["score"] + 2 * bmc["score"])


def test_benchmark_files_are_refused_as_public_references(tmp_path):
    train_path, validation_path, features_path = _write_raw_numerai(tmp_path, n_eras=48)
    with pytest.raises(ValueError, match="Host-only"):
        build_source(
            train_path, validation_path, features_path, tmp_path / "d",
            lockbox_eras=6, embargo_eras=2, selection_eras=8,
            reference_files={"benchmark/part.parquet": features_path},
        )


def _split_with_benchmark(tmp_path):
    train_path, validation_path, features_path = _write_raw_numerai(tmp_path, n_eras=48)
    validation = pd.read_parquet(validation_path)
    bench_path = tmp_path / "raw" / "validation_benchmark_models.parquet"
    validation[["numerai_ticker", "date"]].assign(
        lgbm_jupiter_60=np.random.default_rng(5).uniform(size=len(validation))
    ).to_parquet(bench_path, index=False)
    build_source(
        train_path, validation_path, features_path, tmp_path / "d",
        lockbox_eras=6, embargo_eras=2, selection_eras=8, benchmarks_path=bench_path,
    )
    return prepare_task(tmp_path / "d", "numerai-v6", tmp_path / "fixed", copy_mode="copy")


def test_benchmark_copy_is_closed_and_earns_no_rank(tmp_path):
    split = _split_with_benchmark(tmp_path)
    labels = pd.read_csv(split / "selection_view" / "labels.csv")
    workspace = tmp_path / "workspace"
    (workspace / "submission").mkdir(parents=True)
    manifest = split / "selection_view" / "fixed_holdout_manifest.json"
    labels[["id"]].assign(prediction=labels["benchmark"]).to_csv(
        predictions_path(workspace, "n1", manifest), index=False
    )
    cfg = SimpleNamespace(
        enabled=True, manifest_path=str(manifest), windows=1, objective="mean", std_penalty=0.5,
        score_metric="corr_bmc", copy_threshold=0.999,
    )
    agent = SimpleNamespace(cfg=SimpleNamespace(host_selection=cfg, workspace_dir=workspace))
    node = SimpleNamespace(
        id="n1", is_buggy=False, metric=SimpleNamespace(value=0.1), _term_out=[], analysis="",
        is_terminal=False, continue_improve=True,
    )

    apply_to_node(agent, node, logging.getLogger("test"))

    assert node.is_buggy is True and node.is_terminal is True and node.continue_improve is False
    assert node.host_selection["benchmark_copy"] is True
    assert "0.999" not in node._term_out[-1] and "benchmark" not in node._term_out[-1]


def test_paired_noise_band_keeps_the_champion_inside_the_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(0.03, 0.04, 52)
    eras = [f"e{i:02d}" for i in range(52)]
    reference = dict(zip(eras, base))
    noisy = dict(zip(eras, base + rng.normal(0.001, 0.02, 52)))
    clear = dict(zip(eras, base + rng.normal(0.02, 0.005, 52)))

    assert paired_noise_band(reference, reference) == (0.0, 0.0)
    difference, band = paired_noise_band(noisy, reference)
    assert abs(difference) < band
    difference, band = paired_noise_band(clear, reference)
    assert difference > band
    assert paired_noise_band(noisy, reference) == paired_noise_band(noisy, reference)


def _band_agent():
    cfg = SimpleNamespace(enabled=True, noise_band_z=2.0, noise_band_block=13, noise_band_samples=500)
    return SimpleNamespace(cfg=SimpleNamespace(host_selection=cfg))


def test_noise_verdict_compares_host_series_and_defers_without_them():
    eras = [f"e{i:02d}" for i in range(52)]
    low = np.random.default_rng(2).normal(0.03, 0.04, 52)
    node = lambda values: SimpleNamespace(host_selection={"per_era_score": dict(zip(eras, values))})
    agent = _band_agent()

    assert noise_verdict(agent, node(low + 0.02), node(low)) == "better"
    assert noise_verdict(agent, node(low), node(low + 0.02)) == "worse"
    assert noise_verdict(agent, node(low), node(low)) == "tied"
    assert noise_verdict(agent, SimpleNamespace(host_selection={}), node(low)) is None
    agent.cfg.host_selection.noise_band_z = 0.0
    assert noise_verdict(agent, node(low + 0.02), node(low)) is None


def test_portfolio_prefers_dissimilar_members_and_folds_copies():
    similarity = np.array([
        [1.00, 0.95, 0.30, 1.00],
        [0.95, 1.00, 0.40, 0.95],
        [0.30, 0.40, 1.00, 0.30],
        [1.00, 0.95, 0.30, 1.00],
    ])
    assert choose_members([0.04, 0.03, 0.02, 0.05], similarity, 2, 0.999) == [2, 3]


def test_portfolio_selects_refits_and_combines(tmp_path, monkeypatch):
    split = _split_with_benchmark(tmp_path)
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "workspace" / "submission").mkdir(parents=True)
    labels = pd.read_csv(split / "selection_view" / "labels.csv")
    test = pd.read_parquet(split / "refit_view" / "input" / "test.parquet", columns=["id", "era"])
    rng = np.random.default_rng(3)
    nodes = []
    for index in range(3):
        node_id = f"node{index}"
        test[["id"]].assign(prediction=rng.uniform(size=len(test))).to_csv(
            run / "workspace" / "submission" / f"submission_{node_id}.csv", index=False
        )
        nodes.append({
            "id": node_id, "is_buggy": False, "code": "pass",
            "host_selection": {"score": 0.01 * index, "corr_mean": 1.0, "bmc_mean": 0.0},
        })
    nodes.append({"id": "broken", "is_buggy": True, "code": "", "host_selection": {}})
    (run / "logs" / "journal.json").write_text(json.dumps({"nodes": nodes}))

    def fake_refit(split_root, run_dir, output_dir, *, node_id, python, timeout):
        if node_id == "node1":
            raise RuntimeError("no memory")
        shutil.copy(run / "workspace" / "submission" / f"submission_{node_id}.csv",
                    output_dir / f"submission_{node_id}.csv")
        (output_dir / f"refit_receipt_{node_id}.json").write_text("{}")

    monkeypatch.setattr(portfolio, "refit_node", fake_refit)
    receipt = portfolio.deliver(split, run, run / "refit", python="python", timeout=1, size=5)

    assert sorted(m["node_id"] for m in receipt["members"]) == ["node0", "node1", "node2"]
    assert sorted(receipt["delivered_nodes"]) == ["node0", "node2"]
    assert "node1" in receipt["refit_failed"]
    output = pd.read_csv(run / "refit" / "submission_portfolio.csv")
    assert list(output["id"]) == list(test["id"])
    assert labels["id"].isin(output["id"]).all() and output["prediction"].notna().all()
    assert json.loads((run / "logs" / "portfolio.json").read_text())["fallback_single_node"] is False


def test_portfolio_falls_back_to_best_single_node_below_the_floor(tmp_path):
    split = _split_with_benchmark(tmp_path)
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    nodes = [
        {"id": "a", "is_buggy": False, "host_selection": {"score": 0.01, "corr_mean": -9.0}},
        {"id": "b", "is_buggy": False, "host_selection": {"score": 0.02, "corr_mean": -9.0}},
    ]
    (run / "logs" / "journal.json").write_text(json.dumps({"nodes": nodes}))

    result = portfolio.select_portfolio(split, run)

    assert result["fallback_single_node"] is True
    assert [m["node_id"] for m in result["members"]] == ["b"]


def _improve_agent(retry):
    scfg = SimpleNamespace(
        metric_improvement_threshold=0.0001, max_improve_failure=3, num_improves=3,
        retry_improve_from_local_best=retry, back_debug_depth=3, max_debug_depth=5,
    )
    return SimpleNamespace(
        acfg=SimpleNamespace(check_data_leakage=False), scfg=scfg, metric_maximize=True,
        search_start_time=None, current_node_list=[],
    )


def _improve_pair(child_metric):
    from engine.search_node import SearchNode

    best = SearchNode(code="best", plan="best", stage="improve", metric=MetricValue(0.04, maximize=True),
                      is_buggy=False, is_valid=True)
    best.local_best_node = best
    best.continue_improve = True
    child = SearchNode(code="child", plan="child", stage="improve", parent=best,
                       metric=MetricValue(child_metric, maximize=True), is_buggy=False, is_valid=True)
    best.expected_child_count = 1
    child.local_best_node = best
    return best, child


def test_failed_improvement_retries_from_branch_best_only_when_enabled():
    from engine.evaluation import check_improvement

    best, worse = _improve_pair(0.03)
    check_improvement(_improve_agent(retry=True), worse, best)
    assert worse.is_terminal and not worse.continue_improve
    assert best.continue_improve is False  # select() expands the branch best again

    better = _improve_pair(0.05)[1]
    better.parent = best
    best.children.add(better) if isinstance(best.children, set) else best.children.append(better)
    better.local_best_node = best
    check_improvement(_improve_agent(retry=True), better, best)
    assert better.local_best_node is better and best.continue_improve is True

    best, worse = _improve_pair(0.03)
    check_improvement(_improve_agent(retry=False), worse, best)
    assert worse.continue_improve and not worse.is_terminal  # original behaviour: extend the child


def test_repaired_worse_attempt_also_retries_from_branch_best():
    from engine.evaluation import check_improvement
    from engine.search_node import SearchNode

    best, _ = _improve_pair(0.03)
    failed = SearchNode(code="failed", plan="failed", stage="improve", parent=best, is_buggy=True)
    failed.local_best_node = best
    repaired = SearchNode(code="fixed", plan="fixed", stage="debug", parent=failed,
                          metric=MetricValue(0.005, maximize=True), is_buggy=False, is_valid=True)
    repaired.local_best_node = best

    check_improvement(_improve_agent(retry=True), repaired, failed)

    assert repaired.is_terminal and not repaired.continue_improve
    assert best.continue_improve is False  # select() expands the branch best again


OLD_DESCRIPTION = """## Files
- `neutralizer/*.parquet`: Numerai's per-stock risk exposures, keyed by date.
- `benchmark/*.parquet`: predictions of Numerai's official benchmark model
  `lgbm_jupiter_60`, keyed by `numerai_ticker` and `date`, also for the
  unlabelled eras (as in the live tournament).
  Neither folder contains labels; join them on `numerai_ticker` and `date == era`.

## Required output
finite numeric predictions.
The Host holds labels of part of `test.parquet` and scores your submission
on those rows with the metric above.
"""


def test_memory_note_states_what_pandas_does():
    rows = np.zeros(4, dtype=np.float32)
    frame = pd.DataFrame({"bin": np.zeros(4, np.int8), "rank": rows, "era": np.zeros(4, np.int16)})
    assert frame.values.dtype == np.float32
    frame["freq"] = np.zeros(4, np.int32)
    assert frame.values.dtype == np.float64
    assert frame.to_numpy(dtype=np.float32).dtype == np.float32


def test_strip_benchmark_updates_views_and_keeps_host_labels(tmp_path):
    from fixed_holdout.common import read_manifest, tree_sha256
    from fixed_holdout.strip_benchmark import RETIRED_OBJECTIVE, clean_description, strip_benchmark

    text = clean_description(OLD_DESCRIPTION)
    assert "benchmark" not in text.lower() and "lgbm" not in text
    assert "They contain no labels; join them" in text and "`neutralizer/*.parquet`" in text
    assert "on those rows.\n" in text and "metric above" not in text
    ranked = "that Host score is what ranks\nsolutions during the search.\n"
    stated = clean_description(OLD_DESCRIPTION + ranked + RETIRED_OBJECTIVE)
    assert RETIRED_OBJECTIVE not in stated and clean_description(stated) == stated
    noted = clean_description(OLD_DESCRIPTION + "and keep integer features as int8.\n")
    assert noted.count(MEMORY_NOTE) == 1 and noted.count(SPEED_RULE) == 1
    assert clean_description(noted) == noted
    budget ="- Compute available to your program: 32 CPU cores, no GPU, about 120 GB RAM. A program\n"
    assert "24 threads. A program" in clean_description(OLD_DESCRIPTION + budget, "24 threads")
    with pytest.raises(ValueError, match="no compute budget"):
        clean_description(OLD_DESCRIPTION, "24 threads")

    split = _split_with_benchmark(tmp_path)
    for view in ("train_view", "refit_view"):
        (split / view / "input" / "benchmark").mkdir()
        (split / view / "input" / "benchmark" / "part.parquet").write_bytes(b"x")
    labels = (split / "selection_view" / "labels.csv").read_bytes()

    digest = strip_benchmark(split)

    assert not (split / "train_view" / "input" / "benchmark").exists()
    assert not (split / "refit_view" / "input" / "benchmark").exists()
    assert tree_sha256(split / "train_view" / "input") == digest
    train = read_manifest(split / "train_view" / "fixed_holdout_manifest.json", "train_view")
    assert train["public_tree_sha256"] == digest
    refit = read_manifest(split / "refit_view" / "fixed_holdout_manifest.json", "refit_view")
    assert refit["refit_tree_sha256"] == tree_sha256(split / "refit_view" / "input")
    evaluator = read_manifest(split / "evaluator_view" / "fixed_holdout_manifest.json", "evaluator_view")
    assert evaluator["public_tree_sha256"] == digest
    assert (split / "selection_view" / "labels.csv").read_bytes() == labels


def _banded_pair(child_shift):
    best, child = _improve_pair(0.04 + child_shift)
    eras = [f"e{i:02d}" for i in range(52)]
    base = np.random.default_rng(4).normal(0.04, 0.04, 52)
    wiggle = np.random.default_rng(5).normal(0, 0.02, 52)
    wiggle -= wiggle.mean()
    best.host_selection = {"per_era_score": dict(zip(eras, base))}
    child.host_selection = {"per_era_score": dict(zip(eras, base + child_shift + wiggle))}
    agent = _improve_agent(retry=True)
    agent.cfg = _band_agent().cfg
    return agent, best, child


def test_gain_inside_the_noise_band_keeps_the_branch_best_and_the_child():
    from engine.evaluation import check_improvement

    agent, best, tied = _banded_pair(0.0001)
    check_improvement(agent, tied, best)
    assert tied.local_best_node is best  # a nominal gain does not replace the branch best
    assert tied.continue_improve and not tied.is_terminal  # no retry for a tie

    agent, best, better = _banded_pair(0.02)
    check_improvement(agent, better, best)
    assert better.local_best_node is better

    agent, best, worse = _banded_pair(-0.02)
    check_improvement(agent, worse, best)
    assert worse.is_terminal and best.continue_improve is False


def test_benchmark_copy_closes_its_lineage():
    from engine.evaluation import check_improvement

    best, copy = _improve_pair(0.04)
    copy.is_buggy = True
    copy.is_terminal = True
    copy.continue_improve = False
    copy.host_selection = {"benchmark_copy": True}

    assert check_improvement(_improve_agent(retry=True), copy, best) is True
    assert copy.is_terminal and not copy.continue_improve


def test_champion_is_not_replaced_inside_the_noise_band(monkeypatch):
    from engine import solution_manager

    agent = _band_agent()
    eras = [f"e{i:02d}" for i in range(52)]
    base = np.random.default_rng(6).normal(0.04, 0.04, 52)
    wiggle = np.random.default_rng(7).normal(0.0005, 0.02, 52)
    champion = SimpleNamespace(
        metric=MetricValue(0.04, maximize=True),
        host_selection={"per_era_score": dict(zip(eras, base))},
    )
    agent.best_node = champion
    challenger = SimpleNamespace(
        metric=MetricValue(0.0405, maximize=True),
        host_selection={"per_era_score": dict(zip(eras, base + wiggle))},
    )
    assert solution_manager._beats_best(agent, challenger) is False
    clear = SimpleNamespace(
        metric=MetricValue(0.06, maximize=True),
        host_selection={"per_era_score": dict(zip(eras, base + 0.02))},
    )
    assert solution_manager._beats_best(agent, clear) is True


def test_resource_trace_summary_reports_peak_and_cores():
    from engine.executor import _ResourceTrace

    trace = _ResourceTrace(pid=-1)  # no /proc entry: sampler thread never starts
    trace.samples = [(0, 6e9, 0.0), (600, 60e9, 600.0), (1200, 150e9, 1200.0), (1800, 197e9, 1800.0)]
    text = trace.summary()
    assert "peak memory 197 GB at min 30" in text and "1.0 cores" in text


def test_debug_chain_stops_after_consecutive_memory_errors(monkeypatch):
    from engine import evaluation
    from engine.search_node import SearchNode

    first = SearchNode(code="a", plan="a", stage="draft", is_buggy=True, exc_type="MemoryError")
    second = SearchNode(code="b", plan="b", stage="debug", parent=first, is_buggy=True, exc_type="MemoryError")
    other = SearchNode(code="c", plan="c", stage="debug", parent=first, is_buggy=True, exc_type="ValueError")
    assert evaluation._consecutive_memory_errors(second) == 2
    assert evaluation._consecutive_memory_errors(other) == 0

    monkeypatch.setattr(evaluation, "backpropagate", lambda node, value, add_to_tree=True: None)
    monkeypatch.setattr(evaluation, "get_node_reward", lambda agent, node: -1)
    agent = _improve_agent(retry=True)
    second.local_best_node = second
    agent.scfg.max_consecutive_memory_errors = 0
    evaluation.check_improvement(agent, second, first)
    assert second.is_terminal is False  # off by default
    agent.scfg.max_consecutive_memory_errors = 2
    evaluation.check_improvement(agent, second, first)
    assert second.is_terminal is True and first.is_terminal is True


def test_ledger_shows_the_parts_of_the_host_score():
    from engine.research_loop import ResearchLedger

    host = {"corr_mean": 0.027, "bmc_mean": 0.0083, "benchmark_similarity": 0.451}
    start = SimpleNamespace(
        id="s", parent=None, is_buggy=False, stage="draft", plan="",
        metric=SimpleNamespace(value=0.03), host_selection=host,
    )
    child = SimpleNamespace(
        id="c", parent=start, is_buggy=False, stage="improve",
        plan="DIRECTION: targets\nHYPOTHESIS: another label helps",
        metric=SimpleNamespace(value=0.04), host_selection={**host, "bmc_mean": 0.012},
    )
    ledger = ResearchLedger({"targets": "labels"})
    ledger.register(child, "targets")
    ledger.conclude(child, 0.0)

    text = ledger.prompt_text("targets", start)
    assert "Current solution on the Host rows: Host score 0.0300, correlation 0.0270" in text
    assert "added beyond the benchmark +0.0083" in text and "similarity to the benchmark 0.45" in text
    assert "added beyond the benchmark +0.0120" in text
    assert "Current solution" not in ledger.prompt_text("targets")


def test_duplicate_code_is_closed_before_it_runs():
    import threading

    from fixed_holdout.selection import close_duplicate_code

    earlier = SimpleNamespace(id="old", code="x = 1  # first\nprint(x)\n")
    agent = SimpleNamespace(
        cfg=SimpleNamespace(host_selection=SimpleNamespace(enabled=True, block_duplicate_code=True)),
        journal=SimpleNamespace(nodes=[earlier]),
        journal_lock=threading.Lock(),
    )
    same = SimpleNamespace(id="new", code="# reformatted\nx = 1\n\nprint(x)", metric=None, _term_out=None)
    other = SimpleNamespace(id="other", code="x = 2\nprint(x)\n", metric=None, _term_out=None)
    empty = SimpleNamespace(id="empty", code="", metric=None, _term_out=None)

    assert close_duplicate_code(agent, other, logging.getLogger("test")) is False
    assert close_duplicate_code(agent, empty, logging.getLogger("test")) is False
    assert close_duplicate_code(agent, same, logging.getLogger("test")) is True
    assert same.is_buggy is True and same.is_terminal is True and same.continue_improve is False
    assert same.host_selection == {"duplicate_of": "old"}
    assert same.metric.value is None and "old" in same._term_out[0]

    agent.cfg.host_selection.block_duplicate_code = False
    twin = SimpleNamespace(id="twin", code=earlier.code)
    assert close_duplicate_code(agent, twin, logging.getLogger("test")) is False


def _run_candidate(tmp_path, code, stop_after=3, timeout=20):
    cfg = SimpleNamespace(
        start_cpu_id=0,
        cpu_number=1,
        agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=1, num_gpus=0)),
        evaluation_authority=SimpleNamespace(
            mode="off", runtime_protocol_observer_enabled=False
        ),
    )
    interpreter = Interpreter(
        tmp_path, timeout=timeout, cfg=cfg, nan_metric_stop_after=stop_after
    )
    return interpreter.run(code, "node-nan")


def test_run_reporting_nan_metrics_is_stopped_early(tmp_path):
    code = (
        "import time\n"
        "for epoch in range(40):\n"
        "    print(f'Epoch {epoch} - Loss: nan - Val Score: nan', flush=True)\n"
        "    time.sleep(0.1)\n"
        "time.sleep(60)\n"
    )
    result = _run_candidate(tmp_path, code)
    assert result.exc_type == "NonFiniteMetricError"
    assert result.exec_time < 15
    assert "Val Score: nan" in "".join(result.term_out)
    assert "NonFiniteMetricError: the Host stopped" in "".join(result.term_out)


def test_finite_metric_resets_the_nan_streak(tmp_path):
    code = (
        "for epoch in range(12):\n"
        "    print(f'Epoch {epoch} - Loss: nan - Val Score: 0.03', flush=True)\n"
        "    print('train_loss=nan')\n"
        "    print('train_loss=nan')\n"
        "print('nan_count: 0, filled NaN values')\n"
    )
    result = _run_candidate(tmp_path, code)
    assert result.exc_type is None


def test_timeout_still_reported_when_watching_metrics(tmp_path):
    code = "import time\nprint('working', flush=True)\ntime.sleep(60)\n"
    result = _run_candidate(tmp_path, code, timeout=2)
    assert result.exc_type == "TimeoutError"
    assert "working" in "".join(result.term_out)


def test_numerai_config_turns_the_nan_stop_on():
    from config import _load_cfg

    cfg = _load_cfg(
        REPO / "mlevolve" / "config" / "config_numerai_framework.yaml",
        use_cli_args=False,
    )
    assert cfg.exec.nan_metric_stop_after == 3


def _neutralizer_split(root):
    columns = {f"neutralizer_column_{i}": [0.1 * i + r / 10 for r in range(4)] for i in range(3)}
    exposures = pd.DataFrame(
        {
            "numerai_ticker": ["A", "B", "A", "C"],
            "date": ["e1", "e1", "e2", "e2"],
            "data_type": "train",
            **columns,
        }
    )
    train = pd.DataFrame({"numerai_ticker": ["B", "A"], "era": ["e1", "e2"]})
    test = pd.DataFrame({"numerai_ticker": ["C", "D"], "era": ["e2", "e2"]})
    text = (
        "Files\n- `neutralizer/*.parquet`: exposures. They contain no labels; "
        "join them on `numerai_ticker` and `date == era`.\n\nThe Host ranks\n"
        "solutions during the search.\n"
    )
    for view in ("train_view", "refit_view"):
        input_dir = root / view / "input"
        (input_dir / "neutralizer").mkdir(parents=True)
        exposures.iloc[:2].to_parquet(input_dir / "neutralizer" / "part_a.parquet")
        exposures.iloc[2:].to_parquet(input_dir / "neutralizer" / "part_b.parquet")
        train.to_parquet(input_dir / "train.parquet")
        test.to_parquet(input_dir / "test.parquet")
        (input_dir / "description.md").write_text(text)
    return exposures


def test_neutralizer_arrays_follow_the_row_order_of_each_file(tmp_path):
    from fixed_holdout.align_neutralizer import align_split

    exposures = _neutralizer_split(tmp_path)
    digest, coverage = align_split(tmp_path)
    assert coverage["train_view"] == {"train": 1.0, "test": 0.5}

    input_dir = tmp_path / "train_view" / "input"
    names = json.loads((input_dir / "neutralizer_columns.json").read_text())
    train = np.load(input_dir / "neutralizer_train.npy", mmap_mode="r")
    test = np.load(input_dir / "neutralizer_test.npy", mmap_mode="r")
    assert train.dtype == np.float16 and train.shape == (2, 3)
    expected = exposures[names].to_numpy(dtype=np.float16)
    assert np.array_equal(train[0], expected[1])
    assert np.array_equal(train[1], expected[2])
    assert np.array_equal(test[0], expected[3])
    assert np.isnan(test[1]).all()

    assert not (input_dir / "neutralizer").exists()
    assert (tmp_path / "neutralizer_source" / "train_view" / "part_a.parquet").exists()
    text = (input_dir / "description.md").read_text()
    assert "neutralizer_train.npy" in text and "neutralizer/*.parquet" not in text
    assert align_split(tmp_path)[0] == digest



# ---- verdict-only Host feedback ------------------------------------------------

@pytest.fixture
def verdict_mode():
    from fixed_holdout import selection

    selection._VERDICT["on"] = True
    selection._HIDDEN.clear()
    yield selection
    selection._VERDICT["on"] = False
    selection._HIDDEN.clear()


def test_verdict_mode_follows_the_config():
    from fixed_holdout.selection import configure_feedback

    on = SimpleNamespace(host_selection=SimpleNamespace(enabled=True, feedback="verdict"))
    assert configure_feedback(on) is True
    assert configure_feedback(SimpleNamespace(host_selection=SimpleNamespace(enabled=True))) is False
    assert configure_feedback(SimpleNamespace(host_selection=SimpleNamespace(enabled=False, feedback="verdict"))) is False


def test_host_numbers_are_masked_in_prompts_but_other_numbers_stay(verdict_mode):
    verdict_mode.hide_host_numbers({"score": 0.046704, "corr_mean": 0.04665, "bmc_mean": 0.011691,
                                    "benchmark_similarity": 0.7613, "window_means": [0.0576, 0.044]})
    prompt = {
        "Memory": "Validation Metric: 0.0467 then 0.046704 and 0.04665; best 0.0576",
        "Parts": ["bmc 0.011691", "similarity 0.7613"],
        "Own": "Final Validation Score: 0.03267 ; lr 0.0005 ; epochs 25 ; seed 0.5",
    }
    masked = verdict_mode.mask_prompt(prompt)
    assert "0.0467" not in masked["Memory"] and "0.04665" not in masked["Memory"]
    assert "0.0576" not in masked["Memory"]
    assert masked["Parts"] == ["bmc [hidden]", "similarity [hidden]"]
    assert masked["Own"] == prompt["Own"]  # the agent's own validation score is not a Host number


def test_masking_is_off_without_verdict_mode():
    from fixed_holdout import selection

    selection._VERDICT["on"] = False
    selection._HIDDEN.add(0.046704)
    try:
        assert selection.mask_prompt("score 0.0467") == "score 0.0467"
    finally:
        selection._HIDDEN.clear()


def test_node_display_reports_only_the_verdict(verdict_mode):
    from engine.search_node import SearchNode

    parent = SearchNode(code="a", plan="a", stage="draft", metric=MetricValue(0.04, maximize=True),
                        is_buggy=False, is_valid=True)
    parent.host_selection = {"score": 0.04, "per_era_score": {"e1": 0.04}}
    child = SearchNode(code="b", plan="b", stage="improve", parent=parent,
                       metric=MetricValue(0.05, maximize=True), is_buggy=False, is_valid=True)
    child.host_selection = {"score": 0.05, "verdict_vs_parent": "tied"}
    text = parent._format_metric_change(child)
    assert text == verdict_mode.VERDICT_TEXT["tied"]
    assert "0.05" not in text and "0.04" not in text
    assert "0.04" not in child.fetch_parent_memory()


def test_apply_to_node_records_the_verdict_against_the_scored_parent(tmp_path, verdict_mode):
    _, split = _selection_split(tmp_path)
    test, _, _ = _selection_and_lockbox(split)
    manifest = split / "selection_view" / "fixed_holdout_manifest.json"
    workspace = tmp_path / "ws"
    agent = SimpleNamespace(cfg=SimpleNamespace(
        workspace_dir=workspace,
        host_selection=SimpleNamespace(enabled=True, manifest_path=str(manifest), windows=4, objective="mean",
                                       std_penalty=0.5, feedback="verdict", noise_band_z=2.0,
                                       noise_band_block=2, noise_band_samples=200),
    ))

    def scored(node_id, values, parent=None):
        node = SimpleNamespace(id=node_id, is_buggy=False, metric=MetricValue(0.9, maximize=True),
                               host_selection={}, _term_out=[], analysis="", parent=parent)
        path = predictions_path(workspace, node_id, manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        test[["id"]].assign(prediction=values).to_csv(path, index=False)
        apply_to_node(agent, node, logging.getLogger("test"))
        return node

    good = test["feature_a"].to_numpy() + 1e-3 * np.arange(len(test))
    parent = scored("p", good)
    assert parent.host_selection["verdict_vs_parent"] is None
    same = scored("c", good, parent=SimpleNamespace(is_buggy=True, host_selection={}, parent=parent))
    assert same.host_selection["verdict_vs_parent"] == "tied"  # the buggy step in between is skipped
    masked = verdict_mode.mask_prompt(f"Best Score: {same.metric.value:.4f} ({same.metric.value})")
    assert masked == "Best Score: [hidden] ([hidden])"


def test_portfolio_top_worst_window_takes_the_most_robust_nodes():
    similarity = np.eye(4)
    similarity[0, 3] = similarity[3, 0] = 1.0  # node 3 copies node 0
    nodes = [{"host_selection": {"score": s, "window_means": w}} for s, w in [
        (0.050, [0.09, 0.01]), (0.040, [0.04, 0.04]), (0.030, [0.03, 0.03]), (0.050, [0.09, 0.01])]]
    scores = [portfolio.worst_window(n) for n in nodes]
    assert scores == [0.01, 0.04, 0.03, 0.01]
    assert portfolio.top_members(scores, similarity, 2, 0.999) == [1, 2]
    assert portfolio.top_members(scores, similarity, 5, 0.999) == [1, 2, 3]


def test_portfolio_choice_is_recorded_and_validated(tmp_path):
    split = _split_with_benchmark(tmp_path)
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    nodes = [
        {"id": "a", "is_buggy": False, "host_selection": {"score": 0.03, "corr_mean": -9.0, "window_means": [0.06, 0.0]}},
        {"id": "b", "is_buggy": False, "host_selection": {"score": 0.02, "corr_mean": -9.0, "window_means": [0.02, 0.02]}},
    ]
    (run / "logs" / "journal.json").write_text(json.dumps({"nodes": nodes}))

    result = portfolio.select_portfolio(split, run, choose="top_worst_window")
    assert result["choose"] == "top_worst_window"
    assert [m["node_id"] for m in result["members"]] == ["b"]  # fallback keeps the best worst window
    with pytest.raises(ValueError):
        portfolio.select_portfolio(split, run, choose="nope")
