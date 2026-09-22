"""Whole-arm wall-clock backstop, including generation, Host and cleanup.

Launch THIS under a coordinator-owned systemd unit. Exact run-labelled candidate
containers are reaped after controller failure; no other workload is targeted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from core import atomic_json, exclusive, run_process


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--output',type=Path,required=True)
    args,_=p.parse_known_args()
    root=args.output.resolve()
    with exclusive(root/'supervisor'):
        clock_file=root/'CLOCK.json'
        if not clock_file.exists():
            now=time.time()
            atomic_json(clock_file,{'started':now,'deadline':now+7200,
                        'run_label':hashlib.sha256(str(root).encode()).hexdigest()[:24]})
        clock=json.loads(clock_file.read_text())
        label=clock['run_label']
        def cleanup():
            ids=subprocess.check_output(['docker','ps','-aq','--filter',
                    'label=autoresearch.leaf.run='+label],text=True,timeout=10).split()
            for identity in ids:
                subprocess.run(['docker','rm','-f',identity],check=True,capture_output=True,timeout=15)
            return ids
        cleanup()  # Reap only this arm's orphaned containers before resume.
        result=None
        try:
            result=run_process([sys.executable,str(Path(__file__).with_name('main.py')),*sys.argv[1:]],
                Path(__file__).parent,None,root/f'controller-{time.time_ns()}.log',clock['deadline'])
        finally:
            removed=cleanup()
            atomic_json(root/'SUPERVISOR.json',{'process':result,'removed_exact_run_containers':removed,
                         'deadline':clock['deadline'],'finished':time.time()})
        return 0 if result and result['returncode']==0 else 1


if __name__=='__main__': raise SystemExit(main())
