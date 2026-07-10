import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from agents.leakage_audit import audit_code
from agents import result_parse_agent
from engine.search_node import SearchNode


def test_static_audit_blocks_transform_fit_on_holdout_but_allows_train_only_fit():
    leaked = """
X_train, X_val = train_test_split(texts, test_size=0.2)
all_texts = np.concatenate([X_train, X_val, test_texts])
sequences = [extract_punctuation(x) for x in all_texts]
vectorizer = CountVectorizer(analyzer="char")
features = vectorizer.fit_transform(sequences)
"""
    audit = audit_code(leaked)
    assert audit["status"] == "blocked"
    assert audit["hard_block"] is True
    assert {item["issue_code"] for item in audit["issues"]} == {"TRANSFORM_FIT_ON_HOLDOUT"}

    clean = """
X_train, X_val = train_test_split(texts, test_size=0.2)
vectorizer = CountVectorizer(analyzer="char")
train_features = vectorizer.fit_transform(X_train)
val_features = vectorizer.transform(X_val)
test_features = vectorizer.transform(test_texts)
"""
    clean_audit = audit_code(clean)
    assert clean_audit["status"] == "clean"
    assert clean_audit["paper_grade_eligible"] is True


def test_static_audit_keeps_selection_bias_separate_from_hard_leakage():
    code = """
val_probas = {"a": a_val_probs, "b": b_val_probs}
best_weights = None
best_ll = 99
for w1 in np.arange(0.1, 0.9, 0.1):
    candidate = w1 * val_probas["a"] + (1 - w1) * val_probas["b"]
    ll = log_loss(y_val, candidate)
    if ll < best_ll:
        best_ll = ll
        best_weights = (w1, 1 - w1)
print("optimized ensemble weights", best_weights)
print("validation log loss", best_ll)
"""
    audit = audit_code(code)
    assert audit["status"] == "protocol_biased"
    assert audit["hard_block"] is False
    assert audit["metric_disposition"] == "protocol_biased"
    assert audit["memory_disposition"] == "negative_only"


def test_preflight_block_is_structured_persisted_and_saved_as_negative_memory(tmp_path):
    class FakeGlobalMemory:
        def __init__(self):
            self.saved = []

        def save_leakage_audit(self, node):
            self.saved.append(node.leakage_audit)
            return True

    code = """
X_train, X_val = train_test_split(texts)
combined = np.concatenate([X_train, X_val, test_texts])
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(combined)
"""
    node = SearchNode(code=code, plan="leaky", stage="draft")
    memory = FakeGlobalMemory()
    agent = SimpleNamespace(
        acfg=SimpleNamespace(check_data_leakage=True),
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        global_memory=memory,
    )

    assert result_parse_agent.run_pre_execution_leakage_audit(agent, node) is True
    assert node.is_buggy is True
    assert node.is_valid is False
    assert node.leakage_audit["memory_disposition"] == "quarantine"
    assert memory.saved
    registry = tmp_path / "global_memory" / "leakage_audits" / f"{node.leakage_audit['code_sha256']}.json"
    assert registry.exists()
    saved = json.loads(registry.read_text(encoding="utf-8"))
    assert saved["audit"]["issues"][0]["issue_code"] == "TRANSFORM_FIT_ON_HOLDOUT"


def test_runforest_builds_and_retrieves_failure_pattern(tmp_path):
    sys.path.insert(0, str(REPO / "paper-skills" / "hyper_memory"))
    from build_run_forest_memory import build_artifact
    from agents.memory.external_skill_memory import RunForestMemoryLayer

    run_id = "20260101_000000_spooky-author-identification"
    run_dir = tmp_path / "runs" / run_id / "logs"
    run_dir.mkdir(parents=True)
    leaked_code = """
X_train, X_val = train_test_split(texts)
all_texts = np.concatenate([X_train, X_val, test_texts])
v = CountVectorizer()
X = v.fit_transform(all_texts)
"""
    journal = {
        "nodes": [
            {"id": "root", "code": "", "plan": "root", "stage": "root", "step": 0, "metric": {"value": None, "maximize": False}},
            {
                "id": "child",
                "code": leaked_code,
                "plan": "three model text pipeline",
                "analysis": "historical result",
                "stage": "draft",
                "step": 1,
                "branch_id": 1,
                "metric": {"value": 0.2, "maximize": False},
                "is_buggy": False,
                "is_valid": True,
            },
        ],
        "node2parent": {"child": "root"},
        "node2best_local_node": {"child": "child"},
    }
    (run_dir / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(json.dumps({
        "schema": "clean-run-allowlist-v1",
        "entries": [{"run_id": "20260101_000000", "allowed": True}],
        "blocked_runs": [],
    }), encoding="utf-8")
    sop_graph = tmp_path / "sops.json"
    sop_graph.write_text(json.dumps({"nodes": [{
        "id": "sg_test",
        "type": "SOP",
        "title": "Train-only preprocessing",
        "action": "Fit vectorizers on train only",
        "source_branches": [["20260101_000000", "1"]],
    }]}), encoding="utf-8")

    graph, index, report = build_artifact(
        tmp_path / "runs",
        sop_graph,
        allowlist_path=allowlist,
        require_clean_provenance=True,
    )
    assert report["failure_pattern_count"] == 1
    patterns = [node for node in graph["nodes"] if node.get("type") == "FailurePattern"]
    assert patterns[0]["issue_code"] == "TRANSFORM_FIT_ON_HOLDOUT"
    graph_path = tmp_path / "run_forest_graph.json"
    index_path = tmp_path / "run_forest_index.npz"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    np.savez_compressed(index_path, **index)

    layer = RunForestMemoryLayer(str(graph_path), index_path=str(index_path), top_k=3)
    text, refs = layer.retrieve_for_node(
        stage="debug",
        task_id="spooky-author-identification",
        task_desc="text classification",
        query_parts=["CountVectorizer fit on train validation and test"],
    )
    assert "TRANSFORM_FIT_ON_HOLDOUT" in text
    assert "Fit the transformer on the training partition only" in text
    assert any(ref.startswith("failure::leakage::") for ref in refs)
