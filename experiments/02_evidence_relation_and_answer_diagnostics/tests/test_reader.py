import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reader import make_reader_payload,validate_answer

class ReaderTests(unittest.TestCase):
 def setUp(self):self.item={'question':'Where?','answer':'SECRET_GOLD','old':{'evidence_id':'o','speaker':'user','text':'Old.'},'new':{'evidence_id':'n','speaker':'user','text':'New.'}}
 def test_baseline_has_no_gold_or_relation(self):
  p=make_reader_payload(self.item,None,model='x',max_tokens=4096)
  self.assertNotIn('SECRET_GOLD',str(p));self.assertNotIn('relation_proposal',p['messages'][1]['content'])
 def test_relation_only_receives_proposal_not_gold(self):
  p=make_reader_payload(self.item,{'relation':'CHANGE','action':'REPLACE_PROPOSED','reason':'explicit','secret_gold':'SECRET_GOLD'},model='x',max_tokens=4096)
  self.assertIn('CHANGE',p['messages'][1]['content']);self.assertNotIn('SECRET_GOLD',str(p))
 def test_citations_checked_not_semantic_verified(self):
  x=validate_answer(json.dumps({'answer':'New','citations':['n']}),self.item)
  self.assertEqual(x['status'],'parsed');self.assertTrue(x['citation_ids_exist']);self.assertFalse(x['semantic_support_verified'])
  x=validate_answer(json.dumps({'answer':'New','citations':['bad']}),self.item);self.assertFalse(x['citation_ids_exist'])
 def test_invalid_json_is_failure(self):
  self.assertEqual(validate_answer('oops',self.item)['status'],'invalid_json')
