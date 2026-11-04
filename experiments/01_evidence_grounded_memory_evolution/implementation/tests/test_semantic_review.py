import json
import unittest
from evidence_memory.semantic_review import parse_review, review_payload


class SemanticReviewTests(unittest.TestCase):
 def setUp(self):
  self.source={'speaker':'user','text':'I am planning to move to Beijing.'}
  self.candidate={'subject':'user','predicate':'residence','value':'Beijing','quote':'planning to move to Beijing'}
  self.good={'subject_support':'supported','temporal_modality':'plan','semantic_support':'contradicted','reason':'A plan is not a current residence.','quote':'planning to move to Beijing'}
 def test_plan_does_not_approve_current_fact(self):
  result=parse_review(json.dumps(self.good),self.source)
  self.assertEqual(result['status'],'reviewed')
  self.assertEqual(result['recommendation'],'DEFER')
  self.assertFalse(result['applied'])
 def test_supported_review_remains_advisory(self):
  self.good.update(temporal_modality='current',semantic_support='supported')
  result=parse_review(json.dumps(self.good),self.source)
  self.assertEqual(result['recommendation'],'eligible_for_further_review')
  self.assertFalse(result['applied'])
 def test_fabricated_quote_is_not_accepted(self):
  self.good['quote']='I live in Beijing'
  self.assertEqual(parse_review(json.dumps(self.good),self.source)['status'],'invalid_evidence')
 def test_invalid_schema_and_unparseable_do_not_approve(self):
  self.assertEqual(parse_review('bad',self.source)['status'],'invalid_json')
  self.good['semantic_support']='yes'
  self.assertEqual(parse_review(json.dumps(self.good),self.source)['status'],'invalid_schema')
 def test_unsupported_subject_and_unknown_defer(self):
  self.good.update(subject_support='other_person',temporal_modality='past',semantic_support='supported')
  self.assertEqual(parse_review(json.dumps(self.good),self.source)['recommendation'],'DEFER')
 def test_qwen_array_wrapper_and_no_gold(self):
  self.assertEqual(parse_review(json.dumps([self.good]),self.source)['status'],'reviewed')
  payload=review_payload(dict(self.source, answer='GOLD_SECRET'),self.candidate,model='qwen3-8b',max_tokens=4096)
  self.assertNotIn('GOLD_SECRET',str(payload))
  self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])

class ReviewInputValidationTests(unittest.TestCase):
 def test_rejects_candidate_quote_not_in_source(self):
  with self.assertRaises(ValueError):
   review_payload({'speaker':'user','text':'I live in Beijing.'}, {'subject':'user','predicate':'home','value':'Paris','quote':'I live in Paris.'}, model='qwen3-8b', max_tokens=4096)
 def test_rejects_assistant_turn_as_user_statement(self):
  with self.assertRaises(ValueError):
   review_payload({'speaker':'assistant','text':'I live in Beijing.'}, {'subject':'user','predicate':'home','value':'Beijing','quote':'I live in Beijing.'}, model='qwen3-8b', max_tokens=4096)
