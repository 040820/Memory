import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name('make_pilot.py')


class PilotTest(unittest.TestCase):
    def test_blind_export_is_reproducible_and_preserves_dialogue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mem = root / 'mem.jsonl'
            mem.write_text(json.dumps({'ID': 'p1', 'Fixed_Profile': {'secret': 'PROFILE_SECRET'}, 'Full_Session_Chain': [
                {'Session_ID': 1, 'Date': '2022-01-01', 'Session_Dialogue': {'dialogue_turn_1': [{'role': 'user', 'content': 'I live in Boston.'}]}},
                {'Session_ID': 2, 'Date': '2022-02-01', 'Updated_Attributes': [{'Attribute': 'Residence', 'Before': 'Boston', 'After': 'Paris'}], 'Session_Dialogue': {'dialogue_turn_1': [{'role': 'user', 'content': 'I moved to Paris.'}]}, 'Session_Questions': [{'question_id': 'Q_001', 'answer': 'GOLD_SECRET'}]}
            ]}) + '\n', encoding='utf8')
            long = root / 'long.json'
            long.write_text(json.dumps([{'question_id': 'l1', 'question_type': 'knowledge-update', 'question': 'SECRET_QUESTION', 'answer': 'SECRET_ANSWER', 'answer_session_ids': ['s2'], 'question_date': '2023/01/01', 'haystack_session_ids': ['s1', 's2'], 'haystack_dates': ['2022/01/01', '2022/01/02'], 'haystack_sessions': [[{'role': 'user', 'content': 'old'}], [{'role': 'user', 'content': 'new', 'has_answer': True}]]}]), encoding='utf8')
            loco = root / 'loco.json'
            loco.write_text(json.dumps([{'sample_id': 'c1', 'conversation': {'session_1_date_time': '2022-01-01', 'session_1': [{'speaker': 'Amy', 'dia_id': 'D1:1', 'text': 'I moved home.'}]}, 'qa': [{'answer': 'HIDDEN'}]}]), encoding='utf8')
            def run(name):
                out = root / name
                result = subprocess.run([sys.executable, str(SCRIPT), '--memconflict', str(mem), '--longmemeval', str(long), '--locomo', str(loco), '--output', str(out), '--mem-per-stratum', '1', '--long-count', '1', '--locomo-count', '1'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                return out
            a, b = run('a'), run('b')
            blind = (a / 'annotation.jsonl').read_text(encoding='utf8')
            self.assertEqual(blind, (b / 'annotation.jsonl').read_text(encoding='utf8'))
            cases = [json.loads(x) for x in blind.splitlines()]
            self.assertEqual(len(cases), 3)
            self.assertNotIn('PROFILE_SECRET', blind)
            self.assertNotIn('GOLD_SECRET', blind)
            self.assertNotIn('SECRET_ANSWER', blind)
            self.assertNotIn('SECRET_QUESTION', blind)
            self.assertNotIn('has_answer', blind)
            self.assertIn('I live in Boston.', blind)
            self.assertIn('I moved to Paris.', blind)
            self.assertIn('D1:1', blind)
            self.assertIn('GOLD_SECRET', (a / 'references.jsonl').read_text(encoding='utf8'))
            self.assertTrue(all(c['annotation']['relation'] is None for c in cases))
            again = subprocess.run([sys.executable, str(SCRIPT), '--output', str(a)], capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)


if __name__ == '__main__':
    unittest.main()
