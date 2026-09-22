"""Independent Host verifier, preserving original Leaf log-loss exactly.

Candidate Python/checkpoints are NEVER imported by this privileged process.
Adapted from old free_budget_executor.verify and budget_checkpoint.loss.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from core import sha


def regular(root, name):
    root = Path(root).resolve()
    p = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('unsafe relative file')
    for parent in (p, *p.parents):
        if parent == root: break
        if parent.is_symlink(): raise ValueError('symlink in artifact path')
    if not p.is_file() or not p.resolve().is_relative_to(root):
        raise ValueError('artifact escapes root or is not a file')
    return p


def score(labels, probs, classes):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (len(labels), len(classes)) or not np.isfinite(probs).all():
        raise ValueError('probability shape or finite check')
    if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(1), 1, atol=1e-5):
        raise ValueError('probability simplex')
    indices = pd.Series(labels).map({c:i for i,c in enumerate(classes)})
    if indices.isna().any(): raise ValueError('unknown labels')
    probs = np.clip(probs, 1e-15, 1-1e-15)
    probs /= probs.sum(1, keepdims=True)
    return float(-np.log(probs[np.arange(len(indices)), indices.to_numpy(dtype=int)]).mean())


def verify(artifacts, public, hidden, spec, expected_host_sha):
    artifacts, public, hidden = map(Path, (artifacts, public, hidden))
    if sha(hidden/'search.csv') != expected_host_sha:
        raise ValueError('Host label hash mismatch')
    current = json.loads(regular(artifacts, 'CURRENT.json').read_text())['bundle']
    if Path(current).name != current: raise ValueError('unsafe bundle name')
    receipt = json.loads(regular(artifacts, current+'/receipt.json').read_text())
    bundle = artifacts/current
    for k in ('node_id', 'code_sha256', 'data_manifest_sha256', 'contract_sha256'):
        if receipt[k] != spec[k]: raise ValueError('identity mismatch: '+k)
    if not any(k.startswith('model/') for k in receipt['files']):
        raise ValueError('missing checkpoint')
    if not {'predictions.npz', 'submission.csv'} <= receipt['files'].keys():
        raise ValueError('missing predictions')
    for name, digest in receipt['files'].items():
        if sha(regular(bundle, name)) != digest: raise ValueError('artifact hash mismatch')
    replay = json.loads(regular(artifacts, 'REPLAY.json').read_text())
    if replay.get('passed') is not True or replay.get('bundle') != current:
        raise ValueError('fresh replay missing')
    classes = json.loads((public/'schema.json').read_text())['classes']
    frames = {'tune': pd.read_csv(public/'tune.csv'), 'search': pd.read_csv(hidden/'search.csv'),
              'test': pd.read_csv(public/'test.csv')}
    values = np.load(regular(bundle, 'predictions.npz'), allow_pickle=False)
    for split, frame in frames.items():
        if not np.array_equal(values[split+'_ids'], frame.id.astype(str)):
            raise ValueError('row alignment: '+split)
    tune_loss = score(frames['tune'].species, values['tune_probs'], classes)
    if abs(tune_loss-receipt['tune_logloss']) > 1e-8: raise ValueError('tune score mismatch')
    loss = score(frames['search'].species, values['search_probs'], classes)
    # Validate test simplex without introducing a test target score.
    score([classes[0]]*len(frames['test']), values['test_probs'], classes)
    submission = pd.read_csv(regular(bundle, 'submission.csv'))
    if list(submission.columns) != ['id']+classes: raise ValueError('submission classes')
    if not np.array_equal(submission.id.astype(str), frames['test'].id.astype(str)):
        raise ValueError('submission IDs')
    if not np.allclose(submission[classes], values['test_probs'], rtol=1e-6, atol=1e-8):
        raise ValueError('submission probabilities')
    return {'schema':'leaf_independent_host_v1', 'verified':True, 'search_logloss':loss,
            'tune_logloss':tune_loss, 'bundle':current, 'replay':replay,
            'predictions_sha256':sha(bundle/'predictions.npz'),
            'data_manifest_sha256':spec['data_manifest_sha256'],
            'official_score':False}
