"""Operational invariants for the single-agent propose/execute/keep loop.

Adapted responsibilities from ruz048/autoresearch main.py at 1e9849798006.
No MLEvolve search engine, no candidate-log metric extraction, no git reset.
"""
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from contextlib import contextmanager


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def exclusive(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'CONTROLLER.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_runtime(root, manifest):
    for name, digest in manifest['files'].items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('unsafe runtime member')
        if sha(Path(root)/p) != digest:
            raise ValueError('runtime hash mismatch: '+name)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as out:
        json.dump(obj, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run_process(command, cwd, env, log, deadline):
    """One process group; deadline includes all subprocess work. No shell."""
    if time.time() >= deadline:
        return {'returncode': None, 'timeout': True, 'elapsed': 0.0}
    start = time.monotonic()
    with Path(log).open('xb') as out:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
        timeout = False
        try:
            proc.wait(timeout=max(.001, deadline - time.time()))
        except subprocess.TimeoutExpired:
            timeout = True
        finally:
            # Also reap descendants after a parent exits on its own.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    return {'returncode': proc.returncode, 'timeout': timeout,
            'elapsed': time.monotonic() - start}


def accepted_score(process, receipt):
    if process['returncode'] != 0 or process['timeout']:
        return None
    if receipt.get('schema') != 'leaf_independent_host_v1' or not receipt.get('verified'):
        return None
    value = receipt.get('search_logloss')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


class RunState:
    """State recoverable exclusively from immutable attempt results.

    Interrupted attempts consume elapsed wall time and are never promoted.
    A persisted absolute deadline is never reset by resume.
    """
    def __init__(self, root, contract, seconds=7200, clock=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        launch = self.root / 'LAUNCH.json'
        if not launch.exists():
            now = clock['started'] if clock else time.time()
            atomic_json(launch, {'contract': contract, 'started': now, 'deadline': now + seconds})
        self.launch = json.loads(launch.read_text())
        if self.launch['contract'] != contract:
            amendment_path=self.root/'AMENDMENT.json'
            if not amendment_path.exists():raise ValueError('resume contract mismatch')
            amendment=json.loads(amendment_path.read_text())
            if (amendment.get('source_launch_sha256')!=sha(launch) or
                amendment.get('from_contract')!=self.launch['contract'] or
                amendment.get('to_contract')!=contract or
                amendment.get('original_started')!=self.launch['started'] or
                amendment.get('original_deadline')!=self.launch['deadline'] or
                (clock and (clock['started']!=self.launch['started'] or clock['deadline']!=self.launch['deadline']))):
                raise ValueError('invalid protocol amendment or reset deadline')
        self.recover()

    def recover(self):
        self.history = []
        self.incumbent = None
        for folder in sorted(self.root.glob('attempt-*')):
            f = folder / 'RESULT.json'
            if not f.exists():
                self.history.append({'attempt': folder.name, 'score': None, 'status': 'interrupted'})
                continue
            row = json.loads(f.read_text())
            if row['code_sha256'] != sha(folder / 'train.py'):
                raise ValueError('candidate evidence changed')
            receipt_file = folder / 'HOST.json'
            receipt = json.loads(receipt_file.read_text()) if receipt_file.exists() else {}
            if row.get('host_sha256') != (sha(receipt_file) if receipt_file.exists() else None):
                raise ValueError('Host evidence changed')
            value = accepted_score(row['process'], receipt)
            if value != row['score']:
                raise ValueError('saved score disagrees with Host receipt')
            if value is not None and (self.incumbent is None or value < self.incumbent['score']):
                self.incumbent = {'attempt': folder.name, 'score': value, 'code_sha256': row['code_sha256']}
            self.history.append(row)
        atomic_json(self.root / 'STATE.json', {'incumbent': self.incumbent, 'history': self.history,
                                             'deadline': self.launch['deadline']})

    def new_attempt(self):
        folders = list(self.root.glob('attempt-*'))
        index = max([int(p.name.split('-')[-1]) for p in folders] + [0]) + 1
        folder = self.root / f'attempt-{index:05d}'
        folder.mkdir()
        return folder

    def finish(self, folder, process, receipt=None):
        if (folder / 'RESULT.json').exists():
            raise FileExistsError('immutable attempt already finalized')
        if receipt is not None:
            atomic_json(folder / 'HOST.json', receipt)
        value = accepted_score(process, receipt or {})
        keep = value is not None and (self.incumbent is None or value < self.incumbent['score'])
        proposal=json.loads((folder/'PROPOSAL.json').read_text()) if (folder/'PROPOSAL.json').exists() else {}
        error=json.loads((folder/'ERROR.json').read_text()) if (folder/'ERROR.json').exists() else None
        log=folder/'training.log'
        tail=log.read_text(errors='replace')[-8000:] if log.exists() and value is None else None
        atomic_json(folder / 'RESULT.json', {
            'attempt': folder.name, 'process': process, 'score': value,
            'status': 'keep' if keep else ('discard' if value is not None else 'failed'),
            'code_sha256': sha(folder / 'train.py'),
            'host_sha256': sha(folder / 'HOST.json') if (folder / 'HOST.json').exists() else None,
            'change':proposal.get('change'),'error':error,'untrusted_training_log_tail':tail,
            'finished': time.time()})
        self.recover()
