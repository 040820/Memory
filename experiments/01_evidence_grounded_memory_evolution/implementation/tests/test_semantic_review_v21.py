import json,unittest
from evidence_memory.semantic_review_v21 import review_payload,parse_and_route

class SemanticReviewV21Tests(unittest.TestCase):
 def setUp(self):
  self.source={'evidence_id':'ev1','speaker':'user','text':'I live in Beijing.'}
  self.candidate={'subject':'user','predicate':'residence','value':'Beijing','quote':'I live in Beijing.','time_reference':'PRESENT','modality':'ASSERTED','polarity':'POSITIVE','precision':'NOT_APPLICABLE','scope':'residence'}
  self.review={'source_subject':'USER','subject_reference':'I','candidate_subject_match':'MATCH','time_reference':'PRESENT','modality':'ASSERTED','polarity':'POSITIVE','precision':'NOT_APPLICABLE','scope_match':'MATCH','semantic_support':'SUPPORTED','evidence_quotes':[{'evidence_id':'ev1','quote':'I live in Beijing.'}],'reason_codes':[],'reason':'Directly stated.','source_restatement':'The user says they live in Beijing.','source_qualifiers':[], 'comparisons':dict.fromkeys(['subject','predicate','value','polarity','time','modality','scope'],'MATCH')}
 def run_review(self):return parse_and_route(json.dumps(self.review),self.source,self.candidate)
 def test_complete_supported_keeps_advisory_route(self):
  r=self.run_review();self.assertEqual(r['route'],'REVIEW_NEXT');self.assertFalse(r['applied']);self.assertEqual(r['model_review']['source_restatement'],self.review['source_restatement'])
 def test_missing_required_comparison_is_technical_error(self):
  del self.review['comparisons']['scope'];self.assertEqual(self.run_review()['route'],'TECHNICAL_ERROR')
 def test_unsupported_comparison_defer_even_if_model_claims_support(self):
  self.review['comparisons']['value']='UNKNOWN';r=self.run_review();self.assertEqual(r['route'],'DEFER');self.assertIn('SUPPORTED_WITH_UNMATCHED_COMPONENT',r['reasons'])
 def test_bad_qualifier_quote_defer(self):
  self.review['source_qualifiers']=['not present'];r=self.run_review();self.assertEqual(r['route'],'DEFER');self.assertIn('QUALIFIER_NOT_IN_SOURCE',r['reasons'])
 def test_qualifiers_are_retained_as_quotes_not_proof(self):
  self.review['source_qualifiers']=['Beijing'];r=self.run_review();self.assertEqual(r['model_review']['source_qualifiers'],['Beijing']);self.assertFalse(r['applied'])
 def test_payload_no_gold_and_model_thinking_disabled(self):
  p=review_payload(dict(self.source,answer='SECRET_GOLD'),self.candidate,model='model',max_tokens=4096);self.assertNotIn('SECRET_GOLD',str(p));self.assertFalse(p['chat_template_kwargs']['enable_thinking'])

class V21RequiredComponentsTests(SemanticReviewV21Tests):
 def test_subject_cannot_be_waived_as_not_applicable(self):
  self.review['comparisons']['subject']='NOT_APPLICABLE'
  self.assertEqual(self.run_review()['route'],'DEFER')

class V21NestedFormatTests(SemanticReviewV21Tests):
 def test_nested_judgment_is_accepted_with_exact_validation(self):
  keys=('source_restatement','source_qualifiers','comparisons')
  nested={k:self.review[k] for k in keys};nested['judgment']={k:v for k,v in self.review.items() if k not in keys}
  result=parse_and_route(json.dumps(nested),self.source,self.candidate)
  self.assertEqual(result['route'],'REVIEW_NEXT')
 def test_disagreeing_nested_and_top_level_fields_are_rejected(self):
  nested=dict(self.review);nested['judgment']=dict(self.review,semantic_support='CONTRADICTED')
  self.assertEqual(parse_and_route(json.dumps(nested),self.source,self.candidate)['route'],'TECHNICAL_ERROR')
 def test_contradicted_comparison_normalized_as_mismatch(self):
  self.review['semantic_support']='CONTRADICTED';self.review['comparisons']['value']='CONTRADICTED'
  result=self.run_review();self.assertEqual(result['route'],'DEFER');self.assertEqual(result['model_review']['comparisons']['value'],'MISMATCH')

class V21FieldsFormatTests(SemanticReviewV21Tests):
 def test_nested_fields_are_accepted_with_exact_validation(self):
  keys=('source_restatement','source_qualifiers','comparisons')
  nested={k:self.review[k] for k in keys};nested['fields']={k:v for k,v in self.review.items() if k not in keys}
  self.assertEqual(parse_and_route(json.dumps(nested),self.source,self.candidate)['route'],'REVIEW_NEXT')
 def test_conflicting_wrappers_are_rejected(self):
  keys=('source_restatement','source_qualifiers','comparisons');nested={k:self.review[k] for k in keys}
  nested['fields']={k:v for k,v in self.review.items() if k not in keys};nested['judgment']=dict(nested['fields'],semantic_support='INSUFFICIENT')
  self.assertEqual(parse_and_route(json.dumps(nested),self.source,self.candidate)['route'],'TECHNICAL_ERROR')
