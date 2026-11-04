import json, unittest
from evidence_memory.semantic_review_v22 import review_payload,parse_and_route

class ComponentEvidenceTests(unittest.TestCase):
 def setUp(self):
  self.source={'evidence_id':'ev1','speaker':'user','text':'I live in Beijing.'}
  self.candidate={'subject':'user','predicate':'residence','value':'Beijing','quote':'I live in Beijing.','time_reference':'PRESENT','modality':'ASSERTED','polarity':'POSITIVE','precision':'NOT_APPLICABLE','scope':'residence'}
  self.review={'source_subject':'USER','subject_reference':'I','candidate_subject_match':'MATCH','time_reference':'PRESENT','modality':'ASSERTED','polarity':'POSITIVE','precision':'NOT_APPLICABLE','scope_match':'MATCH','semantic_support':'SUPPORTED','evidence_quotes':[{'evidence_id':'ev1','quote':'I live in Beijing.'}],'reason_codes':[],'reason':'User says this.','source_restatement':'The user says they live in Beijing.','source_qualifiers':[],'comparisons':dict.fromkeys(['subject','predicate','value','polarity','time','modality','scope'],'MATCH'),
  'component_evidence':{'subject':{'anchor':'I','quote':'I live in Beijing.','evidence_id':'ev1'},'relation':{'anchor':'live in','quote':'I live in Beijing.','evidence_id':'ev1'},'value':{'anchor':'Beijing','quote':'I live in Beijing.','evidence_id':'ev1'}},'reference_resolution':[]}
 def run_review(self):return parse_and_route(json.dumps(self.review),self.source,self.candidate)
 def test_covered_candidate_only_enters_further_review(self):
  r=self.run_review();self.assertEqual(r['route'],'REVIEW_NEXT');self.assertFalse(r['applied']);self.assertEqual(r['component_evidence_check'],'PASS')
 def test_value_anchor_outside_quoted_evidence_defers(self):
  self.review['component_evidence']['value']['quote']='I live in';r=self.run_review();self.assertEqual(r['route'],'DEFER');self.assertIn('COMPONENT_ANCHOR_NOT_IN_QUOTE:value',r['reasons'])
 def test_absent_component_defers(self):
  del self.review['component_evidence']['value'];r=self.run_review();self.assertEqual(r['route'],'DEFER');self.assertIn('COMPONENT_EVIDENCE_MISSING:value',r['reasons'])
 def test_component_id_mismatch_defers(self):
  self.review['component_evidence']['subject']['evidence_id']='wrong';self.assertEqual(self.run_review()['route'],'DEFER')
 def test_fabricated_reference_resolution_defers(self):
  self.review['reference_resolution']=[{'expression':'there','antecedent':'Beijing','evidence_id':'ev1','quote':'I live in Beijing.'}]
  self.assertIn('REFERENCE_EXPRESSION_NOT_IN_QUOTE',self.run_review()['reasons'])
 def test_payload_contains_no_gold(self):
  p=review_payload(dict(self.source,answer='SECRET_GOLD'),self.candidate,model='x',max_tokens=4096)
  self.assertNotIn('SECRET_GOLD',str(p))

class AmbiguousAnchorTests(ComponentEvidenceTests):
 def test_bare_anaphor_cannot_be_value_anchor(self):
  self.source['text']='I lived there.';self.candidate['quote']='I lived there.'
  for c in self.review['component_evidence'].values():c['quote']='I lived there.'
  self.review['evidence_quotes'][0]['quote']='I lived there.'
  self.review['component_evidence']['relation']['anchor']='lived'
  self.review['component_evidence']['value']['anchor']='there'
  result=self.run_review();self.assertEqual(result['route'],'DEFER');self.assertIn('BARE_ANAPHOR_VALUE_ANCHOR',result['reasons'])
