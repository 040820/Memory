import json
import unittest
from evidence_memory.semantic_review_v2 import review_payload, parse_and_route


class SemanticReviewV2Tests(unittest.TestCase):
 def setUp(self):
  self.source={'evidence_id':'ev1','speaker':'user','text':'I live in Beijing.'}
  self.candidate={'subject':'user','predicate':'residence','value':'Beijing','quote':'I live in Beijing.', 'time_reference':'PRESENT','modality':'ASSERTED','polarity':'POSITIVE','precision':'NOT_APPLICABLE','scope':'residence'}
  self.review={'source_subject':'USER','subject_reference':'I','candidate_subject_match':'MATCH','time_reference':'PRESENT','modality':'ASSERTED','polarity':'POSITIVE','precision':'NOT_APPLICABLE','scope_match':'MATCH','semantic_support':'SUPPORTED','evidence_quotes':[{'evidence_id':'ev1','quote':'I live in Beijing.'}],'reason_codes':[],'reason':'The user says this directly.'}
 def run_review(self):return parse_and_route(json.dumps(self.review),self.source,self.candidate)
 def test_supported_can_only_enter_further_review(self):
  r=self.run_review();self.assertEqual(r['route'],'REVIEW_NEXT');self.assertFalse(r['applied']);self.assertEqual(r['evidence_spans'][0]['start'],0);self.assertEqual(r['evidence_spans'][0]['end'],18)
 def test_quote_offsets_repeated(self):
  self.source['text']='I live here. I live here.';self.candidate['quote']='I live here.';self.review['evidence_quotes'][0]['quote']='I live here.'
  r=self.run_review();self.assertEqual([(x['start'],x['end']) for x in r['evidence_spans']],[(0,12),(13,25)])
 def test_bad_quote_and_wrong_id_defer(self):
  self.review['evidence_quotes'][0]['quote']='invented';self.assertEqual(self.run_review()['route'],'DEFER')
  self.review['evidence_quotes'][0]={'evidence_id':'wrong','quote':'I live in Beijing.'};self.assertEqual(self.run_review()['route'],'DEFER')
 def test_missing_and_invalid_fields_technical_error(self):
  del self.review['polarity'];self.assertEqual(self.run_review()['route'],'TECHNICAL_ERROR')
  self.assertEqual(parse_and_route('not json',self.source,self.candidate)['route'],'TECHNICAL_ERROR')
 def test_inconsistent_subject_cannot_route(self):
  self.review['source_subject']='OTHER';self.assertEqual(self.run_review()['route'],'DEFER')
 def test_unstated_candidate_qualifiers_defer(self):
  del self.candidate['modality'];self.assertEqual(self.run_review()['route'],'DEFER')
 def test_plan_as_current_defer_even_if_model_says_supported(self):
  self.review['modality']='PLAN';self.review['time_reference']='FUTURE';self.assertEqual(self.run_review()['route'],'DEFER')
 def test_supported_plan_with_plan_candidate_can_proceed(self):
  self.review['modality']='PLAN';self.review['time_reference']='FUTURE';self.candidate.update(modality='PLAN',time_reference='FUTURE')
  self.assertEqual(self.run_review()['route'],'REVIEW_NEXT')
 def test_negation_mismatch_defer(self):
  self.review['polarity']='NEGATIVE';self.assertEqual(self.run_review()['route'],'DEFER')
 def test_approximation_cannot_become_exact(self):
  self.review['precision']='APPROXIMATE';self.candidate['precision']='EXACT';self.assertEqual(self.run_review()['route'],'DEFER')
 def test_insufficient_never_approves(self):
  self.review['semantic_support']='INSUFFICIENT';self.assertEqual(self.run_review()['route'],'DEFER')
 def test_payload_disables_thinking_and_excludes_gold(self):
  p=review_payload(dict(self.source,answer='SECRET_GOLD'),self.candidate,model='test',max_tokens=4096)
  self.assertFalse(p['chat_template_kwargs']['enable_thinking']);self.assertNotIn('SECRET_GOLD',str(p))
 def test_assistant_source_rejected(self):
  self.source['speaker']='assistant'
  with self.assertRaises(ValueError):review_payload(self.source,self.candidate,model='x',max_tokens=100)

class MalformedModelOutputTests(SemanticReviewV2Tests):
 def test_non_string_enum_is_technical_error(self):
  self.review['semantic_support']=['SUPPORTED']
  self.assertEqual(self.run_review()['route'],'TECHNICAL_ERROR')
 def test_missing_subject_reference_defers(self):
  self.review['subject_reference']=''
  self.assertEqual(self.run_review()['route'],'DEFER')
 def test_missing_source_text_defers(self):
  del self.source['text']
  self.assertEqual(self.run_review()['route'],'DEFER')

class SourceCheckReportingTests(SemanticReviewV2Tests):
 def test_source_status_is_explicit(self):
  self.assertEqual(self.run_review()['source_check'],'PASS')
  self.review['evidence_quotes'][0]['quote']='invented'
  self.assertEqual(self.run_review()['source_check'],'FAIL')

class OptionalDiagnosticCodesTests(SemanticReviewV2Tests):
 def test_missing_model_reason_codes_defaults_to_empty(self):
  del self.review['reason_codes']
  r=self.run_review()
  self.assertEqual(r['route'],'REVIEW_NEXT')
  self.assertEqual(r['model_review']['reason_codes'],[])

class UnknownScopeTests(SemanticReviewV2Tests):
 def test_unknown_scope_must_defer_even_if_model_marks_match(self):
  self.candidate['scope']='UNKNOWN'
  result=self.run_review()
  self.assertEqual(result['route'],'DEFER')
  self.assertIn('SCOPE_UNCLEAR',result['reasons'])
