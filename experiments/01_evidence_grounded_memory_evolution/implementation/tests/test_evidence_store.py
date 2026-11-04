import tempfile,unittest
from pathlib import Path
from evidence_memory import store
class EvidenceStoreTests(unittest.TestCase):
 def setUp(self):
  self.assertTrue(hasattr(store,'EvidenceStore'),'EvidenceStore missing')
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.s=store.EvidenceStore(Path(self.tmp.name)/'memory.json')
  self.source={'dataset_id':'ds','conversation_id':'c1','session_id':'s1','turn_id':'t1','speaker':'user','text':'I live in Beijing.','known_at':'2024-01-01T12:00:00Z'}
 def test_same_source_idempotent_but_rewrite_rejected(self):
  sid=self.s.put_source(self.source);self.assertEqual(sid,self.s.put_source(self.source))
  wrong=dict(self.source,text='different')
  with self.assertRaises(ValueError):self.s.put_source(wrong)
 def test_distinct_sources_preserve_independent_occurrences(self):
  a=self.s.put_source(self.source);b=self.s.put_source(dict(self.source,session_id='s2'))
  self.assertNotEqual(a,b);self.assertEqual(len(self.s.sources()),2)
 def test_fact_requires_exact_quote_and_has_separate_id_per_source(self):
  a=self.s.put_source(self.source);b=self.s.put_source(dict(self.source,session_id='s2'))
  f={'subject':'user','predicate':'lives','value':'Beijing','quote':'I live in Beijing.','evidence_id':a}
  id1=self.s.put_candidate(f);id2=self.s.put_candidate(dict(f,evidence_id=b));self.assertNotEqual(id1,id2)
  with self.assertRaises(ValueError):self.s.put_candidate(dict(f,quote='I live in Shanghai.'))
 def test_roundtrip_and_candidates_are_not_trusted_facts(self):
  a=self.s.put_source(self.source);fid=self.s.put_candidate({'subject':'user','predicate':'lives','value':'Beijing','quote':'I live in Beijing.','evidence_id':a})
  self.assertEqual(store.EvidenceStore(self.s.path).get_candidate(fid)['status'],'unverified')
 def test_missing_identity_and_time_rejected(self):
  for key in ['dataset_id','conversation_id','session_id','turn_id','speaker','known_at']:
   x=dict(self.source);del x[key]
   with self.assertRaises(ValueError):self.s.put_source(x)
if __name__=='__main__':unittest.main()
