import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from config import _populate_run_identity, save_run_identity


REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_memory_run_identity_binds_exact_clean_snapshot(tmp_path, monkeypatch):
    graph = tmp_path / "run_forest_graph.json"
    graph.write_text(
        json.dumps(
            {
                "meta": {
                    "source_membership_verified": True,
                    "leak_verified": True,
                    "source_runs": ["run-a", "run-b"],
                }
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "run_forest_index.npz"
    index.write_bytes(b"exact-index-snapshot")
    cfg = OmegaConf.create(
        {
            "external_skill_memory": {
                "enable": True,
                "graph_path": str(graph),
                "index_path": str(index),
            },
            "run_identity": {
                "memory_enabled": True,
                "memory_system": "run_forest_stage_hybrid",
                "memory_version": "stage_hybrid_v2",
                "memory_snapshot_sha256": "",
                "memory_index_sha256": "",
                "memory_source_count": 0,
                "memory_source_runs": [],
                "code_revision": "",
                "code_worktree_sha256": "",
            },
        }
    )
    monkeypatch.setenv("MLEVOLVE_CODE_REVISION", "0cb6cdd2")
    monkeypatch.setenv("MLEVOLVE_CODE_WORKTREE_SHA256", "worktree-digest")

    _populate_run_identity(cfg)

    assert cfg.run_identity.memory_snapshot_sha256 == _sha256(graph)
    assert cfg.run_identity.memory_index_sha256 == _sha256(index)
    assert cfg.run_identity.memory_source_runs == ["run-a", "run-b"]
    assert cfg.run_identity.memory_source_count == 2
    assert cfg.run_identity.code_revision == "0cb6cdd2"
    assert cfg.run_identity.code_worktree_sha256 == "worktree-digest"


def test_run_identity_is_persisted_without_waiting_for_a_journal(tmp_path):
    cfg = OmegaConf.create(
        {
            "log_dir": str(tmp_path / "early-failure" / "logs"),
            "run_identity": {
                "schema": "mlevolve_run_identity_v1",
                "experiment_group": "stage_hybrid_v2_all_clean_history",
                "baseline_reference_group": "baseline_no_external_memory",
                "memory_enabled": True,
                "memory_system": "run_forest_stage_hybrid",
                "memory_version": "stage_hybrid_v2",
                "memory_snapshot_sha256": "graph-hash",
                "memory_index_sha256": "index-hash",
                "memory_source_count": 29,
                "memory_source_runs": ["run-a"],
                "code_revision": "3ac19fd3",
                "code_worktree_sha256": "source-hash",
                "identity_source": "declared_at_runtime",
            },
        }
    )

    identity_path = save_run_identity(cfg)

    assert identity_path == tmp_path / "early-failure" / "logs" / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assert identity["memory_enabled"] is True
    assert identity["memory_version"] == "stage_hybrid_v2"
    assert identity["experiment_group"] == "stage_hybrid_v2_all_clean_history"


def test_run_persists_identity_before_loading_task_or_generating_drafts():
    source = (REPO / "mlevolve" / "run.py").read_text(encoding="utf-8")
    identity_write = source.index("identity_path = save_run_identity(cfg)")
    task_load = source.index("task_desc = load_task_desc(cfg)")
    journal_create = source.index("journal = Journal()")
    assert identity_write < task_load < journal_create


def test_memory_run_identity_fails_closed_on_unclean_graph(tmp_path):
    graph = tmp_path / "unclean.json"
    graph.write_text(
        json.dumps(
            {
                "meta": {
                    "source_membership_verified": True,
                    "leak_verified": False,
                    "source_runs": ["run-a"],
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "external_skill_memory": {
                "enable": True,
                "graph_path": str(graph),
                "index_path": "",
            },
            "run_identity": {
                "memory_enabled": True,
                "memory_system": "run_forest_stage_hybrid",
                "memory_version": "stage_hybrid_v2",
                "memory_snapshot_sha256": "",
                "memory_index_sha256": "",
                "memory_source_count": 0,
                "memory_source_runs": [],
                "code_revision": "",
                "code_worktree_sha256": "",
            },
        }
    )

    try:
        _populate_run_identity(cfg)
    except ValueError as exc:
        assert "source-verified and leak-verified" in str(exc)
    else:
        raise AssertionError("unclean memory graph must fail closed")


def test_baseline_identity_cannot_carry_memory_snapshot_fields():
    cfg = OmegaConf.create(
        {
            "external_skill_memory": {"enable": False, "graph_path": "", "index_path": ""},
            "run_identity": {
                "memory_enabled": True,
                "memory_system": "legacy",
                "memory_version": "legacy",
                "memory_snapshot_sha256": "stale",
                "memory_index_sha256": "stale",
                "memory_source_count": 9,
                "memory_source_runs": ["stale"],
                "code_revision": "",
                "code_worktree_sha256": "",
            },
        }
    )

    _populate_run_identity(cfg)

    assert cfg.run_identity.memory_enabled is False
    assert cfg.run_identity.memory_system == "none"
    assert cfg.run_identity.memory_version == "none"
    assert cfg.run_identity.memory_snapshot_sha256 == ""
    assert cfg.run_identity.memory_index_sha256 == ""
    assert cfg.run_identity.memory_source_count == 0
    assert cfg.run_identity.memory_source_runs == []


def test_historical_source_runs_are_explicitly_labeled_as_no_memory_baselines():
    allowlist = json.loads(
        (REPO / "paper-skills/eval_skill_memory/clean_run_allowlist.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (REPO / "paper-skills/eval_skill_memory/run_identity_registry_v1.json").read_text(encoding="utf-8")
    )
    allowed = {row["run_id"] for row in allowlist["entries"] if row.get("allowed")}
    baseline = registry["groups"]["baseline_no_external_memory"]
    assert baseline["memory_enabled"] is False
    assert baseline["memory_system"] == "none"
    assert baseline["memory_version"] == "none"
    assert set(baseline["run_ids"]) == allowed
    memory_group = registry["groups"]["stage_hybrid_v2_all_clean_history"]
    assert memory_group["memory_enabled"] is True
    assert memory_group["memory_version"] == "stage_hybrid_v2"
