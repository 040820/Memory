import copy
import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).with_name('validate_cases.py')
if PATH.exists():
    spec = importlib.util.spec_from_file_location('validate_cases', PATH)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
else:
    validator = None


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(validator, 'validator is missing')
        self.inputs = [{'case_id': 'x', 'messages': [{'id': 'm1', 'text': 'I live in A.'}],
                        'queries': [{'id': 'q1', 'question': 'Where?'}]}]
        self.gold = [{'case_id': 'x', 'relation': 'add', 'evidence': [{'message_id': 'm1', 'quote': 'live in A'}],
                      'answers': [{'query_id': 'q1', 'expected': 'A'}]}]

    def test_valid_pair_passes(self):
        self.assertEqual(validator.validate(self.inputs, self.gold), [])

    def test_duplicate_and_leaked_gold_fail(self):
        records = copy.deepcopy(self.inputs)
        records[0]['answer'] = 'A'
        errors = validator.validate(records + copy.deepcopy(records), self.gold)
        self.assertTrue(any('duplicate' in e for e in errors))
        self.assertTrue(any('forbidden' in e for e in errors))

    def test_nonexistent_or_inexact_evidence_fails(self):
        gold = copy.deepcopy(self.gold)
        gold[0]['evidence'][0]['quote'] = 'I live in B'
        self.assertTrue(any('quote' in e for e in validator.validate(self.inputs, gold)))

    def test_unmatched_gold_and_query_fail(self):
        gold = copy.deepcopy(self.gold)
        gold[0]['answers'][0]['query_id'] = 'missing'
        self.assertTrue(validator.validate(self.inputs, gold))
        self.assertTrue(validator.validate(self.inputs, []))


if __name__ == '__main__':
    unittest.main()
