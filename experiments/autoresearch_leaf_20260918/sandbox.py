"""AWS Docker candidate isolation. Invoked only by the trusted host harness."""
from pathlib import Path
import re
import subprocess
import time
from core import run_process


def command(image, name, public, attempt, helpers, entry='train.py', run_label=None):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,62}', name):
        raise ValueError('unsafe container name')
    if '@sha256:' not in image:
        raise ValueError('image must be digest pinned')
    public, attempt, helpers = [Path(p).resolve(strict=True) for p in (public, attempt, helpers)]
    if entry not in ('train.py', 'replay.py'):
        raise ValueError('entry not allowed')
    cmd = ['docker', 'run', '--name', name, '--rm', '--init', '--network=none',
           '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
           '--user=65534:65534', '--pids-limit=256', '--gpus=device=0',
           '--cpus=4', '--memory=14g', '--memory-swap=14g', '--shm-size=2g',
           '--tmpfs=/tmp:rw,nosuid,nodev,size=1073741824',
           '--mount', f'type=bind,src={public},dst=/data,readonly',
           '--mount', f'type=bind,src={attempt / "train.py"},dst=/candidate/train.py,readonly',
           '--mount', f'type=bind,src={attempt / "spec.json"},dst=/spec.json,readonly',
           '--mount', f'type=bind,src={attempt / "artifacts"},dst=/artifacts',
           '--mount', f'type=bind,src={helpers / "budget_checkpoint.py"},dst=/helpers/budget_checkpoint.py,readonly',
           '--mount', f'type=bind,src={helpers / "replay.py"},dst=/helpers/replay.py,readonly',
           '--workdir=/artifacts', '--env=HOME=/tmp', '--env=PYTHONPATH=/helpers',
           '--env=PYTHONDONTWRITEBYTECODE=1', '--env=OMP_NUM_THREADS=4',
           '--env=OPENBLAS_NUM_THREADS=4', '--env=MKL_NUM_THREADS=4',
           '--env=MLEVOLVE_PUBLIC_DATA=/data', '--env=MLEVOLVE_BUDGET_SPEC=/spec.json',
           image, 'python', '-u', '/candidate/train.py' if entry == 'train.py' else '/helpers/replay.py']
    if run_label:
        cmd[2:2] = ['--label', 'autoresearch.leaf.run='+run_label]
    return cmd


def execute(image, name, public, attempt, helpers, deadline, replay=False, run_label=None):
    cmd = command(image, name, public, attempt, helpers, 'replay.py' if replay else 'train.py', run_label)
    try:
        return run_process(cmd, attempt, None, Path(attempt)/('replay.log' if replay else 'training.log'), deadline)
    finally:
        # Killing docker CLI alone does not kill its container. Target exact ID/name.
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=15, check=False)
