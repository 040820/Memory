import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from relation_judge import make_payload,parse_relation

class RelationJudgeTests(unittest.TestCase):
 def setUp(self):
  self.old={'fact_id':'f1','value':'Beijing','quote':'I lived in Beijing.'};self.new={'fact_id':'f2','value':'Shanghai','quote':'I moved from Beijing to Shanghai.'}
  self.sources={'ev1':{'evidence_id':'ev1','speaker':'user','text':'I lived in Beijing.','session_time':'2023/01/01'},'ev2':{'evidence_id':'ev2','speaker':'user','text':'I moved from Beijing to Shanghai.','session_time':'2023/02/01'}}
  self.old['evidence_id']='ev1';self.new['evidence_id']='ev2'
  self.output={'relation':'CHANGE','old_quote':'I lived in Beijing.','new_quote':'I moved from Beijing to Shanghai.','reason':'Explicit move.'}
 def test_no_link_is_candidate_only(self):
  self.output['relation']='UNRELATED';r=parse_relation(json.dumps(self.output),self.old,self.new,self.sources)
  self.assertEqual(r['relation'],'UNRELATED');self.assertFalse(r['applied']);self.assertEqual(r['action'],'DEFER')
 def test_bad_quote_never_used_as_support(self):
  self.output['new_quote']='I moved to Paris.';r=parse_relation(json.dumps(self.output),self.old,self.new,self.sources)
  self.assertEqual(r['status'],'invalid_evidence');self.assertFalse(r['applied'])
 def test_unrelated_not_forced_into_complement(self):
  payload=make_payload(self.old,self.new,self.sources,model='x',max_tokens=4096)
  self.assertIn('UNRELATED',payload['messages'][0]['content']);self.assertFalse(payload['enable_thinking'])
 def test_prompt_excludes_gold(self):
  self.old['answer']='SECRET_GOLD';self.sources['ev1']['answer_session_ids']='SECRET_GOLD'
  self.assertNotIn('SECRET_GOLD',str(make_payload(self.old,self.new,self.sources,model='x',max_tokens=4096)))
 def test_invalid_relation_is_error(self):
  self.output['relation']='I think so';self.assertEqual(parse_relation(json.dumps(self.output),self.old,self.new,self.sources)['status'],'invalid_schema')

class TemporalLabelTests(RelationJudgeTests):
 def test_unknown_order_is_visible_to_model(self):
  p=make_payload(self.old,self.new,self.sources,model='x',max_tokens=4096,time_order='ORDER_UNKNOWN')
  self.assertIn('ORDER_UNKNOWN',p['messages'][1]['content'])

class UncertaintyFlagsTests(RelationJudgeTests):
 def test_possible_new_value_is_flagged_and_never_authorizes_update(self):
  self.new['modality']='POSSIBLE'
  r=parse_relation(json.dumps(self.output),self.old,self.new,self.sources)
  self.assertIn('NEW_STATEMENT_UNCERTAIN',r['caveats']);self.assertFalse(r['eligible_for_state_update']);self.assertFalse(r['applied'])
 def test_model_relation_remains_unverified_even_for_explicit_change(self):
  r=parse_relation(json.dumps(self.output),self.old,self.new,self.sources)
  self.assertIn('MODEL_RELATION_UNVERIFIED',r['caveats']);self.assertFalse(r['eligible_for_state_update'])
