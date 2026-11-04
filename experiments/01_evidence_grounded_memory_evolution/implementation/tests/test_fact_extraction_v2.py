import json,unittest
from evidence_memory.fact_extraction_v2 import parse_candidates,create_payload

class FactExtractionV2Tests(unittest.TestCase):
 def setUp(self):
  self.source={'speaker':'user','evidence_id':'ev1','text':'I plan to move to Shanghai next year.'}
  self.fact={'subject':'user','predicate':'moving plan','value':'move to Shanghai next year','quote':'I plan to move to Shanghai next year.','time_reference':'FUTURE','modality':'PLAN','polarity':'POSITIVE','precision':'NOT_APPLICABLE','scope':'moving plan'}
 def check(self):return parse_candidates(json.dumps({'facts':[self.fact]}),self.source)
 def test_plan_retained_as_unverified_with_source(self):
  x=self.check()['candidates'][0];self.assertEqual(x['modality'],'PLAN');self.assertEqual(x['status'],'unverified');self.assertEqual(x['evidence_id'],'ev1');self.assertEqual(x['quote_spans'],[[0,37]])
 def test_invented_quote_skipped(self):
  self.fact['quote']='I live in Shanghai';self.assertEqual(self.check()['skipped'][0]['reason'],'quote_not_in_source')
 def test_missing_qualifier_is_preserved_unknown_not_fabricated(self):
  del self.fact['time_reference'];x=self.check()['candidates'][0];self.assertEqual(x['time_reference'],'UNSPECIFIED');self.assertEqual(x['status'],'unverified')
 def test_invalid_enum_is_skipped(self):
  self.fact['modality']='guess';self.assertEqual(self.check()['skipped'][0]['reason'],'invalid_qualifier')
 def test_invalid_schema_not_empty_success(self):
  self.assertEqual(parse_candidates('{"facts":{}}',self.source)['status'],'invalid_schema')
  self.assertEqual(parse_candidates('[]',self.source)['status'],'parsed')
 def test_other_subject_skipped(self):
  self.fact['subject']='sister';self.assertEqual(self.check()['skipped'][0]['reason'],'non_user_subject')
 def test_payload_does_not_send_gold_and_thinking_disabled(self):
  self.assertNotIn('SECRET_GOLD',str(create_payload(dict(self.source,answer='SECRET_GOLD'),model='x',max_tokens=4096)))
  self.assertFalse(create_payload(self.source,model='x',max_tokens=4096)['chat_template_kwargs']['enable_thinking'])
 def test_invalid_source_rejected(self):
  with self.assertRaises(ValueError):create_payload(dict(self.source,speaker='assistant'),model='x',max_tokens=4096)

class FactExtractionV2SourceTests(FactExtractionV2Tests):
 def test_parse_without_bound_user_source_rejected(self):
  self.source['speaker']='assistant'
  self.assertEqual(self.check()['status'],'invalid_source')
 def test_source_without_evidence_id_rejected(self):
  del self.source['evidence_id']
  self.assertEqual(self.check()['status'],'invalid_source')
