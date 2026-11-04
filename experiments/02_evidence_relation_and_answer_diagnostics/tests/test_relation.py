import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from relation import make_payload, parse_proposal

class RelationTests(unittest.TestCase):
 def setUp(self):
  self.item={'id':'case1','old':{'evidence_id':'old1','speaker':'user','text':'I live in Beijing.'},'new':{'evidence_id':'new1','speaker':'user','text':'I moved from Beijing to Shanghai.'}}
  self.output={'relation':'CHANGE','old_quote':'I live in Beijing.','new_quote':'I moved from Beijing to Shanghai.','same_subject':True,'compatible_scope':True,'old_supported':True,'new_supported':True,'explicit_change':True,'explicit_correction':False,'new_value_asserted':True,'reason':'The user explicitly describes moving.'}
 def parse(self):return parse_proposal(json.dumps(self.output),self.item)
 def test_change_only_suggests_replace_not_apply(self):
  result=self.parse();self.assertEqual(result['action'],'REPLACE_PROPOSED');self.assertFalse(result['applied'])
 def test_new_quote_must_be_exact_source(self):
  self.output['new_quote']='I relocated to Paris.';self.assertEqual(self.parse()['action'],'DEFER')
 def test_recent_alone_cannot_replace(self):
  self.output['explicit_change']=False;self.assertEqual(self.parse()['action'],'DEFER')
 def test_unknown_scope_or_subject_defers_change(self):
  self.output['compatible_scope']=False;self.assertEqual(self.parse()['action'],'DEFER')
 def test_duplicate_and_coexist_do_not_replace(self):
  self.output['relation']='DUPLICATE';self.assertEqual(self.parse()['action'],'MERGE_PROPOSED')
  self.output['relation']='COEXIST';self.assertEqual(self.parse()['action'],'RETAIN_BOTH_PROPOSED')
 def test_invalid_schema_distinct_from_uncertainty(self):
  self.assertEqual(parse_proposal('bad',self.item)['status'],'invalid_json')
  self.output['same_subject']='yes';self.assertEqual(self.parse()['status'],'invalid_schema')
  self.output['same_subject']=True;self.output['relation']='UNCERTAIN';self.assertEqual(self.parse()['action'],'DEFER')
 def test_gold_is_not_in_prompt(self):
  self.item['answer']='SECRET_GOLD';self.item['old']['label']='SECRET_GOLD'
  p=make_payload(self.item,model='qwen3.8-flash',max_tokens=4096)
  self.assertNotIn('SECRET_GOLD',str(p));self.assertFalse(p['enable_thinking'])
 def test_distinct_source_ids_required(self):
  self.item['new']['evidence_id']='old1';self.assertEqual(self.parse()['action'],'DEFER')

class ChangeWithoutNewValueTests(RelationTests):
 def test_old_state_ceased_does_not_invent_new_value(self):
  self.output['new_value_asserted']=False
  result=self.parse()
  self.assertEqual(result['action'],'RETIRE_OLD_PROPOSED')
  self.assertFalse(result['applied'])
