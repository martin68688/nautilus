import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from core import RunState, accepted_score, run_process, exclusive, verify_runtime


class Tests(unittest.TestCase):
    def test_exclusive_and_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            with exclusive(d):
                with self.assertRaises(BlockingIOError):
                    with exclusive(d): pass
            with exclusive(d): pass
            with self.assertRaises(ValueError):
                verify_runtime(d,{'files':{'../outside':'abc'}})
    def test_scores(self):
        process = {'returncode': 0, 'timeout': False}
        for bad in [None, float('nan'), float('inf'), -1, True, '0.1']:
            self.assertIsNone(accepted_score(process, {'schema': 'leaf_independent_host_v1', 'verified': True, 'search_logloss': bad}))
        receipt = {'schema': 'leaf_independent_host_v1', 'verified': True, 'search_logloss': 1.2e-3}
        self.assertEqual(accepted_score(process, receipt), .0012)
        self.assertIsNone(accepted_score({'returncode': 17, 'timeout': False}, receipt))
        self.assertIsNone(accepted_score({'returncode': 0, 'timeout': True}, receipt))

    def test_resume_and_immutability(self):
        with tempfile.TemporaryDirectory() as d:
            s = RunState(d, {'seed': 0})
            deadline = s.launch['deadline']
            for score in [.5, .7, .3]:
                f = s.new_attempt(); (f / 'train.py').write_text('pass\n')
                s.finish(f, {'returncode': 0, 'timeout': False},
                         {'schema': 'leaf_independent_host_v1', 'verified': True, 'search_logloss': score})
            s.new_attempt()
            resumed = RunState(d, {'seed': 0})
            self.assertEqual(resumed.incumbent['score'], .3)
            self.assertEqual(resumed.launch['deadline'], deadline)
            self.assertEqual(resumed.history[-1]['status'], 'interrupted')
            self.assertEqual(resumed.history[1]['status'], 'discard')
            with self.assertRaises(ValueError): RunState(d, {'seed': 1})
            (Path(d) / 'attempt-00001' / 'train.py').write_text('altered')
            with self.assertRaises(ValueError): RunState(d, {'seed': 0})

    def test_exit_and_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            p = run_process([sys.executable, '-c', 'print("val_bpb: 0.1");exit(17)'], d, {}, Path(d)/'a.log', time.time()+3)
            self.assertEqual(p['returncode'], 17)
            p = run_process([sys.executable, '-c', 'import time;time.sleep(10)'], d, {}, Path(d)/'b.log', time.time()+.1)
            self.assertTrue(p['timeout'])
            self.assertLess(p['elapsed'], 2)


if __name__ == '__main__': unittest.main()
