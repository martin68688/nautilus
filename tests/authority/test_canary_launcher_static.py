from __future__ import annotations

import ast
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "deploy" / "run_decision_admissibility_wp7_canary_devpod.sh"


def _embedded_python_blocks(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    blocks: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        if "<<'PY'" not in lines[index]:
            index += 1
            continue
        start = index + 1
        end = start
        while end < len(lines) and lines[end] != "PY":
            end += 1
        if end == len(lines):
            raise AssertionError(f"Unterminated Python heredoc beginning at line {start}")
        blocks.append((start + 1, "\n".join(lines[start:end]) + "\n"))
        index = end + 1
    return blocks


def test_canary_launcher_bash_and_embedded_python_are_syntactically_valid() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    blocks = _embedded_python_blocks(LAUNCHER.read_text())
    assert len(blocks) >= 4
    for line_number, source in blocks:
        ast.parse(source, filename=f"{LAUNCHER}:embedded-python-line-{line_number}")


def test_canary_launcher_pins_r25_protocol_v2_and_failure_finalizer() -> None:
    text = LAUNCHER.read_text()
    assert "decision-admissibility-wp7-certified-canary-r14-source" in text
    assert "wp7-canary-certified-image-task-heldout-aerial-r25" in text
    assert "evaluation_authority.active_protocol_version=2" in text
    assert "EXPECTED_PROTOCOL_REF" in text
    assert "trap finalize_launcher EXIT" in text
    assert "trap handle_launcher_term TERM" in text
    assert "failed_signal_term" in text
    assert "LAUNCHER_SIGNAL" in text
    assert "LAUNCHER_EXIT_CODE" in text


def test_canary_launcher_signal_handlers_fail_closed(tmp_path: Path) -> None:
    text = LAUNCHER.read_text()
    start = text.index("finalize_launcher() {")
    end = text.index(
        'verify_source_snapshot "$RUN_ROOT/SOURCE_PREFLIGHT_VERIFICATION.json"'
    )
    signal_runtime = text[start:end]
    run_root = tmp_path / "term"
    run_root.mkdir()
    script = f"""
set -uo pipefail
RUN_ROOT={shlex.quote(str(run_root))}
verify_source_snapshot() {{ :; }}
{signal_runtime}
kill -s TERM "$$"
exit 99
"""
    completed = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 143, completed.stderr
    assert (run_root / "LAUNCHER_SIGNAL").read_text().strip() == "TERM"
    assert (run_root / "STATE").read_text().strip() == "failed_signal_term"
    assert (run_root / "LAUNCHER_EXIT_CODE").read_text().strip() == "143"
