import tempfile,unittest
from pathlib import Path
from evidence_memory import store
class StateCommitTests(unittest.TestCase):
 def setUp(self):
  self.assertTrue(hasattr(store,'StateStore'),'StateStore missing')
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.s=store.StateStore(Path(self.tmp.name)/'m.json')
 def candidate(self,turn,text,value,at):
  eid=self.s.put_source({'dataset_id':'ds','conversation_id':'c','session_id':turn,'turn_id':'0','speaker':'user','text':text,'known_at':at})
  return self.s.put_candidate({'subject':'user','predicate':'residence','value':value,'quote':text,'evidence_id':eid,'scope':'home','modality':'asserted','cardinality':'single','valid_from':at,'extractor_id':'extractor'})
 def approve(self,fid,relation,old_state_id=None):
  return self.s.record_assessment(fid,{'old_state_id':old_state_id,'assessor_id':'independent-judge','source_supported':True,'identity_supported':True,'predicate_supported':True,'value_supported':True,'scope_supported':True,'modality_supported':True,'time_supported':True,'relation':relation,'relation_supported':True,'reason':'supported by cited user statement'})
 def test_unassessed_candidate_cannot_commit(self):
  f=self.candidate('1','I live in Beijing.','Beijing','2024-01-01T00:00:00Z')
  with self.assertRaises(ValueError):self.s.commit({'decision_id':'d1','action':'ADD','fact_id':f,'assessment_id':'missing','old_state_id':None,'known_at':'2024-01-01T00:00:00Z'},expected_revision=0,idempotency_key='k1')
 def test_add_replace_replay_and_conflicting_revision(self):
  t1='2024-01-01T00:00:00Z';t2='2024-06-01T00:00:00Z';f1=self.candidate('1','I live in Beijing.','Beijing',t1);a1=self.approve(f1,'add')
  d1={'decision_id':'d1','action':'ADD','fact_id':f1,'assessment_id':a1,'old_state_id':None,'known_at':t1};r1=self.s.commit(d1,expected_revision=0,idempotency_key='k1')
  self.assertEqual(self.s.commit(d1,expected_revision=0,idempotency_key='k1'),r1)
  f2=self.candidate('2','I moved from Beijing to Shanghai.','Shanghai',t2);a2=self.approve(f2,'change',r1['state_id']);d2={'decision_id':'d2','action':'REPLACE','fact_id':f2,'assessment_id':a2,'old_state_id':r1['state_id'],'known_at':t2}
  with self.assertRaises(ValueError):self.s.commit(d2,expected_revision=0,idempotency_key='k2')
  r2=self.s.commit(d2,expected_revision=1,idempotency_key='k2');self.assertNotEqual(r1['state_id'],r2['state_id'])
  states=self.s.list_states();self.assertEqual([x['status'] for x in states],['historical','current'])
 def test_future_evidence_and_plan_cannot_replace(self):
  t1='2024-01-01T00:00:00Z';f=self.candidate('1','I live in Beijing.','Beijing',t1);a=self.approve(f,'add')
  d={'decision_id':'d','action':'ADD','fact_id':f,'assessment_id':a,'old_state_id':None,'known_at':'2023-12-01T00:00:00Z'}
  with self.assertRaises(ValueError):self.s.commit(d,expected_revision=0,idempotency_key='k')
 def test_no_self_assessment_or_unsupported_approval(self):
  f=self.candidate('1','I live in Beijing.','Beijing','2024-01-01T00:00:00Z')
  with self.assertRaises(ValueError):self.s.record_assessment(f,{'assessor_id':'extractor','relation':'add','reason':'yes'})
  with self.assertRaises(ValueError):self.s.record_assessment(f,{'assessor_id':'judge','relation':'add','reason':'yes','source_supported':True})
 def test_change_review_must_be_bound_to_specific_old_state(self):
  t1='2024-01-01T00:00:00Z';t2='2024-06-01T00:00:00Z'
  f1=self.candidate('1','I live in Beijing.','Beijing',t1);a1=self.approve(f1,'add');r=self.s.commit({'decision_id':'d1','action':'ADD','fact_id':f1,'assessment_id':a1,'old_state_id':None,'known_at':t1},expected_revision=0,idempotency_key='k1')
  f2=self.candidate('2','I moved to Shanghai.','Shanghai',t2)
  with self.assertRaises(ValueError):self.approve(f2,'change')
  a2=self.approve(f2,'change',r['state_id'])
  with self.assertRaises(ValueError):self.s.commit({'decision_id':'d2','action':'REPLACE','fact_id':f2,'assessment_id':a2,'old_state_id':'different','known_at':t2},expected_revision=1,idempotency_key='k2')
 def test_approximate_cannot_replace_precise_state(self):
  t1='2024-01-01T00:00:00Z';t2='2024-06-01T00:00:00Z'
  f1=self.candidate('1','I live in Beijing.','Beijing',t1);a1=self.approve(f1,'add');r=self.s.commit({'decision_id':'d1','action':'ADD','fact_id':f1,'assessment_id':a1,'old_state_id':None,'known_at':t1},expected_revision=0,idempotency_key='k1')
  eid=self.s.put_source({'dataset_id':'ds','conversation_id':'c','session_id':'2','turn_id':'0','speaker':'user','text':'Maybe I moved to Shanghai.','known_at':t2})
  f=self.s.put_candidate({'subject':'user','predicate':'residence','value':'Shanghai','quote':'Maybe I moved to Shanghai.','evidence_id':eid,'scope':'home','modality':'approximate','cardinality':'single','valid_from':t2,'extractor_id':'extractor'})
  a=self.approve(f,'change',r['state_id'])
  with self.assertRaises(ValueError):self.s.commit({'decision_id':'d2','action':'REPLACE','fact_id':f,'assessment_id':a,'old_state_id':r['state_id'],'known_at':t2},expected_revision=1,idempotency_key='k2')
if __name__=='__main__':unittest.main()

class KnownTimeViewTests(StateCommitTests):
 def test_future_supersession_is_not_visible_in_past(self):
  t1='2024-01-01T00:00:00Z'; t2='2024-06-01T00:00:00Z'
  f1=self.candidate('1','I live in Beijing.','Beijing',t1); a1=self.approve(f1,'add')
  r=self.s.commit({'decision_id':'d1','action':'ADD','fact_id':f1,'assessment_id':a1,'old_state_id':None,'known_at':t1},expected_revision=0,idempotency_key='k1')
  f2=self.candidate('2','I moved from Beijing to Shanghai.','Shanghai',t2); a2=self.approve(f2,'change',r['state_id'])
  self.s.commit({'decision_id':'d2','action':'REPLACE','fact_id':f2,'assessment_id':a2,'old_state_id':r['state_id'],'known_at':t2},expected_revision=1,idempotency_key='k2')
  before=self.s.states_known_at('2024-03-01T00:00:00Z')
  self.assertEqual(len(before),1)
  self.assertEqual(before[0]['value'],'Beijing')
  self.assertEqual(before[0]['status'],'current')
  self.assertIsNone(before[0]['valid_to'])
  self.assertNotIn('superseded_by',before[0])
  self.assertEqual(len(self.s.states_known_at(t2)),2)
