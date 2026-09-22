#!/usr/bin/env python3
"""Plant-only continuation for the 2026-09-21 original-task rerun.

The existing two-task continuation is intentionally not reused: its Plant
worker is waiting on an ignored run, while its iMet worker must remain intact.
This process owns only the new Plant R1 -> R2 -> R3 chain.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import orchestrate_rounds as base

HERE = Path(__file__).resolve().parent
TASK = dict(base.TASKS["plant"])
TASK.update(
    run1="plant-r1-20260920-original-c1",
    r1_unit="autoresearch-plant-r1-20260920-original-c1.service",
)
LOG = HERE / "plant_rerun_orchestrator.log"
SUFFIX = "original-c1"


def log(message: str) -> None:
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + message
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def wait_terminal(run_name: str) -> dict:
    remote_path = f'{TASK["remote_dir"]}/runs/{run_name}'
    log(f"wait {run_name}")
    while True:
        probe = base.remote(TASK, f"test -f {remote_path}/RUN_OUTCOME.json", check=False)
        if probe.returncode == 0:
            outcome = json.loads(base.remote(TASK, f"cat {remote_path}/RUN_OUTCOME.json").stdout)
            log(
                f"terminal {run_name} best={outcome.get('best_search')} "
                f"elapsed={outcome.get('elapsed_seconds')}"
            )
            return outcome
        time.sleep(60)


def launch(round_id: int) -> str:
    run_name = f"plant-r{round_id}-20260920-{SUFFIX}"
    unit = f"autoresearch-plant-r{round_id}-20260920-{SUFFIX}"
    container = f"{unit}-supervisor"
    task_root = TASK["remote_dir"]
    command = (
        f"sudo systemd-run --unit={unit} --service-type=exec "
        f"--property=RuntimeMaxSec=2h1min --property=KillMode=control-group "
        f"--working-directory={task_root} -- "
        f"/usr/bin/docker run --rm --gpus all --network host --name {container} "
        f"-v /var/run/docker.sock:/var/run/docker.sock "
        f"-v /usr/bin/docker:/usr/bin/docker:ro "
        f"-v /etc/autoresearch/apizh-terra-low.env:/etc/autoresearch/apizh-terra-low.env:ro "
        f"-v {task_root}:{task_root}:rw {base.IMAGE} env "
        f"AUTORESEARCH_ROUND={round_id} AUTORESEARCH_MEMORY=true "
        f"AUTORESEARCH_RUN_NAME={run_name} python {task_root}/run_worker.py"
    )
    result = base.remote(TASK, command, check=False)
    if result.returncode:
        raise RuntimeError(f"launch failed R{round_id}: {result.stdout}\n{result.stderr}")
    log(f"launched R{round_id} {run_name}")
    return run_name


def main() -> None:
    log("Plant-only rerun continuation started; iMet is out of scope")
    wait_terminal(TASK["run1"])
    r1_pack = base.distill(TASK, [TASK["run1"]], "r1-original-c1")
    base.install_pack(TASK, r1_pack)
    base.probe(TASK, 2)
    r2 = launch(2)
    wait_terminal(r2)
    r3_pack = base.distill(TASK, [TASK["run1"], r2], "r3-cumulative-original-c1")
    base.install_pack(TASK, r3_pack)
    base.probe(TASK, 3)
    r3 = launch(3)
    wait_terminal(r3)
    log(f"COMPLETE R3 {r3}")


if __name__ == "__main__":
    main()
