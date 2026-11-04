import unittest
from evidence_memory.fact_extraction import parse_candidates, make_messages


class FactExtractionTests(unittest.TestCase):
    def setUp(self):
        self.source = {'speaker': 'user', 'text': 'I live in Beijing.'}

    def test_extract_valid_candidate_as_unverified(self):
        raw = '{"facts":[{"subject":"user","predicate":"residence","value":"Beijing","quote":"I live in Beijing."}]}'
        result = parse_candidates(raw, self.source)
        self.assertEqual(result['status'], 'parsed')
        self.assertEqual(result['candidates'][0]['status'], 'unverified')
        self.assertEqual(result['candidates'][0]['quote'], 'I live in Beijing.')

    def test_invented_quote_is_skipped_not_a_fact(self):
        raw = '{"facts":[{"subject":"user","predicate":"residence","value":"Paris","quote":"I live in Paris."}]}'
        result = parse_candidates(raw, self.source)
        self.assertEqual(result['candidates'], [])
        self.assertEqual(result['skipped'][0]['reason'], 'quote_not_in_source')

    def test_invalid_json_and_wrong_shape_are_distinct_from_no_facts(self):
        self.assertEqual(parse_candidates('bad', self.source)['status'], 'invalid_json')
        self.assertEqual(parse_candidates('{"facts":{}}', self.source)['status'], 'invalid_schema')
        self.assertEqual(parse_candidates('{"facts":[]}', self.source)['status'], 'parsed')

    def test_unsupported_or_empty_fields_are_skipped(self):
        raw = '{"facts":[{"subject":"user","predicate":"","value":"x","quote":"I live"}]}'
        self.assertEqual(parse_candidates(raw, self.source)['skipped'][0]['reason'], 'invalid_fields')

    def test_prompt_contains_source_and_no_gold(self):
        messages = make_messages(self.source)
        self.assertIn(self.source['text'], messages[1]['content'])
        self.assertNotIn('answer_session_ids', str(messages))

if __name__ == '__main__': unittest.main()

class RequestConfigurationTests(unittest.TestCase):
    def test_payload_disables_thinking_and_does_not_contain_gold(self):
        from evidence_memory.fact_extraction import create_payload
        source = {'speaker': 'user', 'text': 'I live in Beijing.', 'answer': 'SECRET_GOLD'}
        payload = create_payload(source, model='qwen3-8b', max_tokens=1024)
        self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(payload['model'], 'qwen3-8b')
        self.assertNotIn('SECRET_GOLD', str(payload))

class ObservedModelFormatTests(unittest.TestCase):
    def test_model_empty_list_is_valid_empty_extraction(self):
        result = parse_candidates('[]', {'text': 'Hello', 'speaker':'user'})
        self.assertEqual(result['status'], 'parsed')
        self.assertEqual(result['candidates'], [])

    def test_model_singleton_wrapper_is_parsed_with_quote_validation(self):
        raw='[{"facts":[{"subject":"user","predicate":"home","value":"Beijing","quote":"I live in Beijing."}]}]'
        result=parse_candidates(raw, {'text':'I live in Beijing.', 'speaker':'user'})
        self.assertEqual(len(result['candidates']),1)
        self.assertEqual(result['candidates'][0]['status'],'unverified')
