#!/usr/bin/env python3
"""Create evidence-grounded analyst synthesis for an adapted run ledger."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); src=Path(a.source); j=json.loads((src/'runs').glob if False else (next((src/'runs').glob('*/logs/journal.json'))).read_text())
    nodes={}
    for n in j['nodes']:
        nid=n['id']; scored=n.get('metric',{}).get('value') is not None
        change=n.get('analysis','')
        if scored:
            arch=(f"This node produced a Host-recomputed score after executing the declared candidate. "
                  f"The observed change was: {change[:500]}. The score is conditional on this split, budget, "
                  "container image and fixed metric; it is evidence for a candidate, not a universal recipe.")
            tune=(f"The candidate was accepted only after fresh-process reload and Host scoring. "
                  f"The measured path should be compared with later trials using the same data manifest and metric; "
                  f"its reported internal tuning values remain conditional. Observed description: {change[:420]}")
            debug=("The scored node passed the execution, reload and Host recomputation gates. Remaining risk is "
                   "that the Agent's unmeasured choices and internal validation behavior were not isolated causally; "
                   "a later run should preserve the contract while testing one change at a time.")
        else:
            arch=(f"This node did not receive a Host score because execution failed before a valid prediction result. "
                  f"The recorded failure or proposal was: {change[:600]}. No architectural benefit or ban is inferred.")
            tune=("There is no measured tuning result for this node because the candidate did not pass the Host "
                  "prediction contract. Treat its parameters as untested and keep the failure separate from scored evidence.")
            debug=(f"The executable failure is preserved as a conditional debug observation: {change[:650]}. "
                   "The next candidate must repair the observed contract or runtime issue and still pass fresh-process "
                   "reload plus Host recomputation before any score is compared.")
        nodes[nid]={
            'architecture_lesson':arch,
            'tuning_lesson':tune,
            'debug_lesson':debug,
            'alternative':("Retain the task contract and test a single measurable modification in the next candidate; "
                           "use the same data and Host scorer so the conditional comparison remains interpretable."),
            'match_terms':['autoresearch','host recomputation','fresh process','same data manifest']
        }
    out={'method':'analyst_conditional_trajectory_synthesis','nodes':nodes}
    Path(a.out).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'nodes':len(nodes)}))
if __name__=='__main__': main()
