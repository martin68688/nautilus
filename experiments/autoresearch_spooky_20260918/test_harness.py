import unittest,tempfile,json
from pathlib import Path
import numpy as np
import harness as h
class HarnessTests(unittest.TestCase):
 def sample(self):
  out={}
  import csv
  for s in ['tune','search','test']:
   with (h.DATA/(s+'.csv')).open() as f:ids=np.array([r['id'] for r in csv.DictReader(f)])
   out[s]=np.full((len(ids),3),1/3);out[s+'_ids']=ids
  return out
 def test_host_score(self):
  a=self.sample();self.assertAlmostEqual(h.validate_predictions(a,a)['search_loss'],float(np.log(3)))
 def test_nan(self):
  a=self.sample();a['search'][0,0]=float('nan')
  with self.assertRaises(ValueError):h.validate_predictions(a,a)
 def test_wrong_rows(self):
  a=self.sample();a['search_ids'][0]='bad'
  with self.assertRaises(ValueError):h.validate_predictions(a,a)
 def test_reload_mismatch(self):
  a=self.sample();b=self.sample();b['search'][0]=[.2,.3,.5]
  with self.assertRaises(ValueError):h.validate_predictions(a,b)
 def test_memory_complete(self):
  b=h.bridge();self.assertEqual(len(b.cards),225);key=next(iter(b.cards));self.assertEqual(b.read([key])['cards'][0]['body'],b.cards[key])
 def test_bad_memory_reference(self):
  b=h.bridge();key=next(iter(b.cards))
  with self.assertRaises(ValueError):b.receipt([key],[{'key':'fake','action':'adopt','reason':'x'}])
 def test_restore_and_tamper(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);a=p/'attempt-000';a.mkdir();s='abc';(p/'incumbent.py').write_text(s);arr=self.sample()
   for name in ['predict1','predict2']:np.savez(a/(name+'.npz'),**arr)
   h.save(a/'HOST_RESULT.json',{'predictions_sha256':h.sha((a/'predict1.npz').read_bytes()),'source_sha256':h.sha(s.encode())})
   h.save(p/'STATE.json',{'best':float(np.log(3)),'best_attempt':'X-000','incumbent_sha256':h.sha(s.encode()),'history':[{'attempt':'X-000','decision':'keep'}]})
   self.assertEqual(h.restore_incumbent(p)[0],s)
   (p/'incumbent.py').write_text('changed')
   with self.assertRaises(AssertionError):h.restore_incumbent(p)
if __name__=='__main__':unittest.main()
