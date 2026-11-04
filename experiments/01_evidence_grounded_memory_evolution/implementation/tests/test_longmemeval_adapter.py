import tempfile
import unittest
from pathlib import Path
from evidence_memory.store import EvidenceStore
from evidence_memory.longmemeval import adapt_sessions, time_metadata


class LongMemEvalAdapterTests(unittest.TestCase):
    def test_minute_timestamp_keeps_precision_and_no_timezone(self):
        time = time_metadata('2023/05/20 (Sat) 02:21')
        self.assertEqual(time['known_at'], '2023-05-20T02:21')
        self.assertEqual(time['time_precision'], 'minute')
        self.assertEqual(time['timezone'], 'unknown')
        self.assertEqual(time['raw_session_time'], '2023/05/20 (Sat) 02:21')
        self.assertEqual(time['known_at_basis'], 'session_time')

    def test_missing_or_invalid_time_is_unknown(self):
        for value in ('', None, 'not-a-date'):
            info = time_metadata(value)
            self.assertIsNone(info['known_at'])
            self.assertEqual(info['time_precision'], 'unknown')

    def test_source_fidelity_stable_ids_and_distinct_turns(self):
        sessions = [[{'role':'user','content':'  I live in Beijing.\n'},
                     {'role':'assistant','content':'Understood.'}],
                    [{'role':'user','content':'I live in Shanghai.'}]]
        rows = adapt_sessions('case-1', ['s1', 's2'],
                              ['2023/05/20 (Sat) 02:21', ''], sessions)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]['text'], '  I live in Beijing.\n')
        self.assertEqual(rows[0]['turn_id'], '0')
        self.assertEqual(rows[1]['speaker'], 'assistant')
        self.assertIsNone(rows[2]['known_at'])
        with tempfile.TemporaryDirectory() as td:
            store = EvidenceStore(Path(td)/'sources.json')
            first = [store.put_source(row) for row in rows]
            second = [store.put_source(row) for row in rows]
            self.assertEqual(first, second)
            self.assertEqual(len(set(first)), 3)
            self.assertEqual(store.sources()[first[0]]['text'], rows[0]['text'])
            self.assertEqual(store.sources()[first[2]]['time_precision'], 'unknown')

    def test_misalignment_duplicate_ids_and_invalid_turns_fail(self):
        with self.assertRaises(ValueError): adapt_sessions('q', ['s'], [], [[]])
        with self.assertRaises(ValueError): adapt_sessions('q', ['s','s'], ['', ''], [[],[]])
        with self.assertRaises(ValueError): adapt_sessions('q', ['s'], [''], [[{'role':'user'}]])

if __name__ == '__main__': unittest.main()

class EvidenceLocatorTests(unittest.TestCase):
    def test_quote_locator_returns_all_exact_character_spans(self):
        with tempfile.TemporaryDirectory() as td:
            store = EvidenceStore(Path(td) / 'evidence.json')
            row = adapt_sessions('q', ['s'], [''], [[{'role':'user', 'content':'北京，北京。'}]])[0]
            eid = store.put_source(row)
            self.assertEqual(store.locate_quote(eid, '北京'), [(0, 2), (3, 5)])
            self.assertEqual(store.locate_quote(eid, '上海'), [])
