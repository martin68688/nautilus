"""Framework-level fixes: component health, method-family quota, Host selection."""

import json
import logging
from pathlib import Path
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
from fixed_holdout.numerai_source import build_source  # noqa: E402
from fixed_holdout.prepare import prepare_task  # noqa: E402
from fixed_holdout.selection import apply_to_node, predictions_path, score_selection  # noqa: E402
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
        selection_metric="corr_bmc",
    )
    text = (prepared / "public" / "description.md").read_text()
    assert "0.5 * CORR + 2 * BMC" in text and "def bmc(" in text
    split = prepare_task(tmp_path / "d", "numerai-v6", tmp_path / "fixed", copy_mode="copy")
    test = pd.read_parquet(split / "train_view" / "input" / "test.parquet")
    preds = tmp_path / "p.csv"
    test[["id"]].assign(prediction=test["feature_a"] + 1e-3 * np.arange(len(test))).to_csv(preds, index=False)
    manifest = split / "selection_view" / "fixed_holdout_manifest.json"
    corr = score_selection(manifest, preds, windows=1)
    bmc = score_selection(manifest, preds, windows=1, score_metric="bmc")
    mix = score_selection(manifest, preds, windows=1, score_metric="corr_bmc")
    assert mix["score"] == pytest.approx(0.5 * corr["score"] + 2 * bmc["score"])


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
