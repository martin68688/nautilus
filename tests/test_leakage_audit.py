import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from agents.leakage_audit import (
    audit_code,
    load_registry_audit,
    persist_audit,
    rank_eligible,
    structural_sha256,
)
from agents import result_parse_agent
from agents.memory.global_memory import GlobalMemoryLayer
from engine import solution_manager
from engine.execution import validate_executed_node
from engine.evaluation import check_improvement, get_node_reward
from engine.search_node import Journal, SearchNode
from utils.metric import MetricValue


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


def test_train_side_split_with_valid_in_name_is_not_tainted():
    code = """
df_train_valid, df_holdout = train_test_split(df, test_size=0.2)
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(df_train_valid)
holdout_features = vectorizer.transform(df_holdout)
"""
    audit = audit_code(code)
    assert audit["status"] == "clean"


def test_dev_fit_and_pipeline_fit_are_hard_blocked():
    code = """
X_train, X_dev = train_test_split(X)
pipe = Pipeline([])
pipe.fit(X_dev)
"""
    audit = audit_code(code)
    assert audit["hard_block"] is True
    assert "TRANSFORM_FIT_ON_HOLDOUT" in {item["issue_code"] for item in audit["issues"]}


def test_model_eval_set_is_monitoring_not_holdout_fit():
    code = """
X_train, X_val, y_train, y_val = train_test_split(X, y)
model = XGBClassifier()
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="mlogloss")
"""
    audit = audit_code(code)
    assert audit["status"] == "clean"


def test_train_only_fit_before_cv_is_warning_not_hard_block():
    code = """
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(train_texts)
folds = StratifiedKFold(n_splits=5)
"""
    audit = audit_code(code)
    assert audit["status"] == "warning"
    assert audit["hard_block"] is False


def test_structural_hash_ignores_local_variable_renames():
    left = "x = TfidfVectorizer()\ny = x.fit_transform(train_texts)"
    right = "encoder = TfidfVectorizer()\nfeatures = encoder.fit_transform(train_texts)"
    assert structural_sha256(left) == structural_sha256(right)


def test_rank_policy_fails_closed_for_non_clean_or_missing_audit():
    agent = SimpleNamespace(acfg=SimpleNamespace(check_data_leakage=True))
    clean = SearchNode(code="print('ok')", plan="clean", stage="draft", is_buggy=False, is_valid=True)
    clean.leakage_audit = audit_code(clean.code)
    assert rank_eligible(agent, clean) is True

    biased = SearchNode(code="print('biased')", plan="biased", stage="draft", is_buggy=False, is_valid=True)
    biased.leakage_audit = audit_code("""
dev_probas = {'a': a_dev_pred, 'b': b_dev_pred}
best_weights = None
best_score = 99
result = minimize(score, [0.5], args=(y_dev, dev_probas))
print('optimized ensemble weights', result.x)
""")
    assert biased.leakage_audit["status"] == "protocol_biased"
    assert rank_eligible(agent, biased) is False

    missing = SearchNode(code="print('old')", plan="old", stage="draft", is_buggy=False, is_valid=True)
    assert rank_eligible(agent, missing) is False


def test_audit_enforced_journal_rejects_missing_audits():
    missing = SearchNode(
        code="print('legacy')",
        plan="legacy",
        stage="draft",
        metric=MetricValue(0.01, maximize=False),
        is_buggy=False,
        is_valid=True,
    )
    journal = Journal(nodes=[missing], audit_enforced=True)
    assert journal.get_best_node(only_good=True) is None


def test_registry_concurrent_occurrences_are_not_lost(tmp_path):
    code = "X_train, X_val = train_test_split(X)\nTfidfVectorizer().fit_transform(X_val)"
    audit = audit_code(code)
    agent = SimpleNamespace(cfg=SimpleNamespace(workspace_dir=tmp_path))

    def write(index):
        node = SearchNode(code=code, plan="same", stage="draft")
        node.id = f"node-{index}"
        node.leakage_audit = audit
        persist_audit(agent, node)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(20)))

    path = tmp_path / "global_memory" / "leakage_audits" / f"{audit['code_sha256']}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "mlevolve_leakage_registry_record_v2"
    assert len(payload["occurrences"]) == 20


def test_non_clean_metric_cannot_update_best_solution(tmp_path):
    agent = SimpleNamespace(
        acfg=SimpleNamespace(check_data_leakage=True),
        best_node=None,
        top_candidates=[],
        top_k=3,
        metric_maximize=False,
        cfg=SimpleNamespace(workspace_dir=tmp_path),
    )
    node = SearchNode(
        code="print('provisional')",
        plan="provisional",
        stage="draft",
        metric=MetricValue(0.01, maximize=False),
        is_buggy=False,
        is_valid=True,
    )
    node.leakage_audit = audit_code("""
dev_probas = {'a': a_dev_pred, 'b': b_dev_pred}
best_weights = None
best_score = 99
result = minimize(score, [0.5], args=(y_dev, dev_probas))
print('optimized ensemble weights', result.x)
""")
    solution_manager.update_best_solution(agent, node)
    assert agent.best_node is None
    assert agent.top_candidates == []
    assert not (tmp_path / "best_solution").exists()


def test_global_memory_concurrent_negative_occurrences_are_not_lost(tmp_path):
    class FakeRetriever:
        vector_index = None

        def build_index(self, records, texts):
            self.vector_index = object()

        def add_to_index(self, records, texts):
            return None

    layer = GlobalMemoryLayer.__new__(GlobalMemoryLayer)
    layer.memory_dir = tmp_path
    layer._lock = __import__("threading").RLock()
    layer._load_error = None
    layer.records = []
    layer.node_metadata_map = {}
    layer.retriever = FakeRetriever()

    code = "X_train, X_val = train_test_split(X)\nTfidfVectorizer().fit_transform(X_val)"
    audit = audit_code(code)

    def write(index):
        node = SearchNode(code=code, plan="same", stage="draft", id=f"node-{index}")
        node.leakage_audit = audit
        layer.save_leakage_audit(node)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(20)))

    record_id = f"leakage_{audit['code_sha256']}"
    assert len(layer.node_metadata_map[record_id]["source_node_ids"]) == 20
    payload = json.loads((tmp_path / "records.json").read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert len(payload[0]["source_node_ids"]) == 20


def test_structural_history_cannot_block_fresh_clean_code(tmp_path):
    class FakeLayer:
        def structural_failure_patterns(self, code):
            return [{
                "issue_code": "TRANSFORM_FIT_ON_HOLDOUT",
                "category": "transductive_contamination",
                "severity": "high",
                "execution_disposition": "block",
                "evidence": "Historical scaler.fit(X_val)",
                "remediation": "Fit train only",
            }]

    node = SearchNode(
        code="scaler = StandardScaler()\nfeatures = scaler.fit_transform(X_train)",
        plan="safe",
        stage="draft",
    )
    agent = SimpleNamespace(
        acfg=SimpleNamespace(check_data_leakage=True),
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        global_memory=None,
        external_skill_memory=FakeLayer(),
    )
    assert result_parse_agent.run_pre_execution_leakage_audit(agent, node) is False
    assert node.leakage_audit["status"] == "clean"


def test_non_clean_metric_is_withheld_from_reward_and_child_memory():
    parent = SearchNode(code="root", plan="parent", stage="draft")
    child = SearchNode(
        code="print('biased')",
        plan="biased child",
        stage="improve",
        parent=parent,
        metric=MetricValue(0.01, maximize=False),
        is_buggy=False,
        is_valid=True,
        analysis="amazing metric 0.01",
    )
    child.leakage_audit = audit_code("""
dev_probas = {'a': a_dev_pred, 'b': b_dev_pred}
best_weights = None
best_score = 99
result = minimize(score, [0.5], args=(y_dev, dev_probas))
print('optimized ensemble weights', result.x)
""")
    agent = SimpleNamespace(
        acfg=SimpleNamespace(check_data_leakage=True),
        best_metric=MetricValue(0.2, maximize=False),
        best_node=parent,
    )
    assert get_node_reward(agent, child) == 0
    memory = parent.fetch_child_memory()
    assert "metric withheld" in memory
    assert "Validation Metric" not in memory
    assert "amazing metric" not in memory

    child.local_best_node = parent
    parent.metric = MetricValue(0.2, maximize=False)
    agent.search_start_time = None
    assert check_improvement(agent, child, parent) is False
    assert child.local_best_node is parent
    assert child.continue_improve is True


def test_non_clean_node_is_not_registered_as_branch_success(tmp_path):
    node = SearchNode(
        code="print('biased')",
        plan="biased",
        stage="draft",
        branch_id=1,
        metric=MetricValue(0.1, maximize=False),
        is_buggy=False,
        is_valid=True,
    )
    node.leakage_audit = audit_code("""
dev_probas = {'a': a_dev_pred, 'b': b_dev_pred}
best_weights = None
best_score = 99
result = minimize(score, [0.5], args=(y_dev, dev_probas))
print('optimized ensemble weights', result.x)
""")
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / f"submission_{node.id}.csv").write_text("id,label\n1,0\n", encoding="utf-8")
    agent = SimpleNamespace(
        acfg=SimpleNamespace(check_data_leakage=True),
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        branch_successful_nodes={1: []},
    )
    validate_executed_node(agent, node)
    assert agent.branch_successful_nodes[1] == []


def test_corrupt_global_memory_is_not_overwritten(tmp_path):
    records_path = tmp_path / "records.json"
    records_path.write_text("{broken", encoding="utf-8")
    layer = GlobalMemoryLayer.__new__(GlobalMemoryLayer)
    layer.memory_dir = tmp_path
    layer._lock = __import__("threading").RLock()
    layer._load_error = None
    layer.records = []
    layer.node_metadata_map = {}
    layer.retriever = SimpleNamespace(build_index=lambda *args: None)
    layer._load_memory()
    assert layer._load_error

    node = SearchNode(code="print('x')", plan="x", stage="draft")
    node.leakage_audit = audit_code(node.code)
    assert layer.save_node(node) is False
    assert records_path.read_text(encoding="utf-8") == "{broken"


def test_corrupt_registry_is_explicitly_unavailable(tmp_path):
    agent = SimpleNamespace(cfg=SimpleNamespace(workspace_dir=tmp_path))
    digest = "a" * 64
    registry = tmp_path / "global_memory" / "leakage_audits"
    registry.mkdir(parents=True)
    (registry / f"{digest}.json").write_text("{broken", encoding="utf-8")
    audit = load_registry_audit(agent, digest)
    assert audit["status"] == "audit_unavailable"
    assert audit["detector_status"] == "registry_corrupt"
    assert audit["rank_eligible"] is False


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
