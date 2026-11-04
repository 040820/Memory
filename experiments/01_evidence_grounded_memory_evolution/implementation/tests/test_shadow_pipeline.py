import tempfile
import unittest
from pathlib import Path

from evidence_memory.pipeline import ShadowPipeline
from init.config import Config


class ShadowPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pipeline = ShadowPipeline(Path(self.tmp.name) / 'evidence.json')
        self.source = dict(dataset_id='d', conversation_id='c', session_id='s',
                           turn_id='1', speaker='user', text='I live in Beijing.',
                           known_at='2024-01-01T00:00:00')

    def tearDown(self):
        self.tmp.cleanup()

    def test_config_is_off_by_default_and_parses_shadow(self):
        self.assertEqual(Config().evolution.mode, 'off')
        path = Path(self.tmp.name) / 'config.yaml'
        path.write_text('evolution:\n  mode: shadow\n', encoding='utf8')
        self.assertEqual(Config.parse(path).evolution.mode, 'shadow')

    def test_candidate_is_unverified_and_cannot_apply(self):
        candidate = dict(subject='user', predicate='city', value='Beijing',
                         quote='I live in Beijing.')
        trace = self.pipeline.process_source(self.source, [candidate])
        self.assertFalse(trace['applied'])
        self.assertEqual(trace['candidates'][0]['status'], 'unverified')
        self.assertEqual(trace['candidates'][0]['action'], 'DEFER')
        self.assertEqual(self.pipeline.store.list_states(), [])

    def test_bad_quote_is_skipped_and_replay_is_idempotent(self):
        bad = dict(subject='user', predicate='city', value='Paris', quote='Paris')
        one = self.pipeline.process_source(self.source, [bad])
        two = self.pipeline.process_source(self.source, [bad])
        self.assertEqual(one, two)
        self.assertEqual(one['candidates'][0]['status'], 'skipped')
        self.assertEqual(len(self.pipeline.store.sources()), 1)

    def test_source_without_candidates_is_not_a_decision(self):
        trace = self.pipeline.process_source(self.source, [])
        self.assertEqual(trace['candidates'], [])
        self.assertFalse(trace['applied'])

if __name__ == '__main__':
    unittest.main()

class IntegrationGuardTests(unittest.TestCase):
    def test_unconnected_modes_fail_loudly(self):
        from evidence_memory.pipeline import ensure_graph_mode
        ensure_graph_mode('off')
        for mode in ('shadow', 'active', 'typo'):
            with self.assertRaises((NotImplementedError, ValueError)):
                ensure_graph_mode(mode)
