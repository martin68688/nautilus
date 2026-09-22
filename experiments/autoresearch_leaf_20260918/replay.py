"""Fresh-process reload; runs with exactly the same isolated candidate mounts."""
import json
import os
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
from budget_checkpoint import probabilities, feature_frame, write_json


def main():
    namespace = runpy.run_path(os.environ.get('REPLAY_CANDIDATE','/candidate/train.py'), run_name='checkpoint_replay')
    predict = namespace['predict_checkpoint']
    spec=json.loads(Path(os.environ['MLEVOLVE_BUDGET_SPEC']).read_text())
    root = Path(spec['artifact_root'])
    name = json.loads((root/'CURRENT.json').read_text())['bundle']
    if Path(name).name != name:
        raise ValueError('unsafe bundle')
    bundle = root/name
    saved = np.load(bundle/'predictions.npz', allow_pickle=False)
    maximum_error = {}
    for split in ('tune', 'search', 'test'):
        frame = pd.read_csv(Path(spec['public_data'])/(split+'.csv'))
        fresh = probabilities(predict(bundle/'model', feature_frame(frame)), len(frame))
        if not np.allclose(fresh, saved[split+'_probs'], rtol=1e-4, atol=1e-6):
            raise ValueError('fresh-process reload differs: '+split)
        maximum_error[split] = float(np.max(np.abs(fresh-saved[split+'_probs'])))
    write_json(root/'REPLAY.json', {'passed': True, 'max_absolute_error': maximum_error,
                                  'fresh_process': True, 'bundle': name})


if __name__ == '__main__': main()
