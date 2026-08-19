from __future__ import annotations

from types import SimpleNamespace

from engine.executor import Interpreter, _insert_host_preamble_after_future_imports


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        start_cpu_id=0,
        cpu_number=1,
        agent=SimpleNamespace(
            search=SimpleNamespace(parallel_search_num=1, num_gpus=0)
        ),
        evaluation_authority=SimpleNamespace(
            mode="off",
            runtime_protocol_observer_enabled=False,
        ),
    )


def test_executor_preamble_preserves_module_docstring_and_future_import(tmp_path):
    source = '''"""Candidate module."""\nfrom __future__ import annotations\n\nvalue: list[str] = []\nprint("future-ok")\n'''

    result = Interpreter(tmp_path, timeout=10, cfg=_cfg()).run(
        source,
        "future-import-candidate",
    )

    assert result.exc_type is None
    assert "future-ok" in "".join(result.term_out)


def test_host_preamble_is_inserted_after_all_future_imports():
    source = (
        '"""Candidate module."""\n'
        "from __future__ import annotations\n"
        "from __future__ import generator_stop\n"
        "VALUE = 1\n"
    )

    composed = _insert_host_preamble_after_future_imports(
        source,
        "import os\nos.environ['HOST_PREAMBLE'] = '1'\n",
    )

    assert composed.index("from __future__ import generator_stop") < composed.index(
        "import os"
    )
    assert composed.index("import os") < composed.index("VALUE = 1")
    compile(composed, "<composed>", "exec")


def test_invalid_candidate_syntax_is_classified_before_subprocess_start(tmp_path):
    result = Interpreter(tmp_path, timeout=10, cfg=_cfg()).run(
        "def broken(:\n    pass\n",
        "invalid-candidate",
    )

    assert result.exc_type == "CandidateSourceSyntaxError"
    assert result.exc_info["candidate_subprocess_started"] is False
    assert result.exc_info["host_instrumentation_failure"] is False
    assert "Candidate source failed syntax validation" in "".join(result.term_out)
