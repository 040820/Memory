import unittest
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('fact_records', Path(__file__).resolve().parents[1] / 'coregraph' / 'fact_records.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FactRecords = module.FactRecords

class RecordTests(unittest.TestCase):
    def test_parallel_relations_keep_occurrences_and_sources(self):
        s = FactRecords()
        s.add_chunk('a/1', 'I work in Beijing')
        s.add_chunk('b/1', 'I visited Beijing')
        x = s.add_fact('user', 'works in', 'Beijing', 'a/1')
        y = s.add_fact('user', 'visited', 'Beijing', 'b/1')
        self.assertEqual(2, len(s.find_facts('Beijing')))
        self.assertNotEqual(x, y)
        self.assertEqual('I work in Beijing', s.chunks['a/1'])

    def test_decisions_are_not_automatically_committed(self):
        s = FactRecords(); s.add_chunk('1', 'a'); s.add_chunk('2', 'b')
        a=s.add_fact('user','lives','Shanghai','1')
        b=s.add_fact('user','lives','Beijing','2')
        s.add_change(a,b,'change','2','user says moved')
        t=FactRecords.from_dict(s.to_dict())
        self.assertEqual(t.changes_for(a), t.changes_for(b))
        self.assertEqual('recorded_unverified',t.changes_for(a)[0]['decision_status'])
