import sys,unittest,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fact_extraction import create_payload, parse_facts

class ExtractionTests(unittest.TestCase):
 def setUp(self):
  self.sources=[{'evidence_id':'e1','speaker':'user','text':'I have 1250 followers on Instagram.','session_id':'s1','session_time':'2023/05/25 (Thu) 05:26','turn_index':1}]
  self.fact={'evidence_id':'e1','subject_key':'user','attribute_key':'follower_count','object_key':'instagram','scope_key':'UNKNOWN','value':'1250','quote':'1250 followers on Instagram','polarity':'POSITIVE','modality':'ASSERTED'}
 def parse(self):return parse_facts(json.dumps({'facts':[self.fact]}),self.sources)
 def test_fact_is_traceable_unverified(self):
  x=self.parse();self.assertEqual(x['status'],'parsed');self.assertEqual(x['facts'][0]['attribute_key'],'follower_count');self.assertEqual(x['facts'][0]['normalization_status'],'model_proposed');self.assertEqual(x['facts'][0]['status'],'unverified')
 def test_quote_mismatch_goes_to_skip(self):
  self.fact['quote']='1400 followers';self.assertEqual(self.parse()['skipped'][0]['reason'],'quote_not_in_source')
 def test_unknown_id_and_other_subject_rejected(self):
  self.fact['evidence_id']='missing';self.assertEqual(self.parse()['skipped'][0]['reason'],'unknown_evidence')
  self.fact['evidence_id']='e1';self.fact['subject_key']='sister';self.assertEqual(self.parse()['skipped'][0]['reason'],'non_user_subject')
 def test_payload_only_contains_source_ids_and_text(self):
  self.sources[0]['answer']='SECRET';self.sources[0]['session_id']='answer_secret'
  p=create_payload(self.sources,model='x',max_tokens=4096)
  self.assertNotIn('SECRET',str(p));self.assertNotIn('answer_secret',str(p));self.assertFalse(p['enable_thinking'])
 def test_bad_schema_not_empty_success(self):
  self.assertEqual(parse_facts('{"facts":{}}',self.sources)['status'],'invalid_schema')

class ListFormatTests(ExtractionTests):
 def test_top_level_fact_list_is_parsed(self):
  self.assertEqual(parse_facts(json.dumps([self.fact,self.fact]),self.sources)['status'],'parsed')
  self.assertEqual(len(parse_facts(json.dumps([self.fact,self.fact]),self.sources)['facts']),1)
