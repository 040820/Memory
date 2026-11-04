import asyncio
import json
import unittest
from types import SimpleNamespace

from coregraph.evidence_bridge import EvidenceBridge, server_evidence_id
from coregraph.evidence_memory import EvidenceMemory
from coregraph.evidence_review import longmemeval
from query.query_processor import QueryProcessor

# extraction canned by user-turn text (quote must be an exact excerpt)
EXTRACTION = {
    'I live in Shanghai': {
        'subject': 'user', 'predicate': 'live in', 'value': 'Shanghai',
        'quote': 'I live in Shanghai', 'time_reference': 'PRESENT',
        'modality': 'ASSERTED', 'polarity': 'POSITIVE',
        'precision': 'NOT_APPLICABLE', 'scope': 'current'},
    'I live in Beijing now': {
        'subject': 'user', 'predicate': 'live in', 'value': 'Beijing',
        'quote': 'I live in Beijing now', 'time_reference': 'PRESENT',
        'modality': 'ASSERTED', 'polarity': 'POSITIVE',
        'precision': 'NOT_APPLICABLE', 'scope': 'current'},
}


class FakeCall:
    """Offline LLM transport: synthesizes valid responses from the payload."""

    def __init__(self, review_route='REVIEW_NEXT', relation='CHANGE'):
        self.review_route = review_route
        self.relation = relation
        self.payloads = []

    async def __call__(self, payload):
        self.payloads.append(payload)
        system = payload['messages'][0]['content']
        user = json.loads(payload['messages'][1]['content'])
        if 'Extract personal life statements' in system:
            entry = EXTRACTION.get(user['source_text'])
            if entry is None:
                return json.dumps({'facts': []})
            if isinstance(entry, dict) and 'facts' in entry:
                return json.dumps(entry)        # full canned response
            return json.dumps({'facts': [entry]})
        if 'Review the full candidate claim' in system:
            return self._review(user)
        if 'Assess the relationship' in system:
            return self._relation(user)
        raise AssertionError('unexpected payload routed to fake LLM')

    def _review(self, user):
        if self.review_route == 'TECHNICAL_ERROR':
            return 'not-json-at-all'
        source, cand = user['source'], user['candidate']
        eid = source['evidence_id']
        support = ('INSUFFICIENT' if self.review_route == 'DEFER'
                   else 'SUPPORTED')        # INSUFFICIENT -> DEFER via v2 rules
        return json.dumps({
            'source_subject': 'USER', 'subject_reference': cand['quote'],
            'candidate_subject_match': 'MATCH',
            'time_reference': cand.get('time_reference', 'UNSPECIFIED'),
            'modality': cand.get('modality', 'UNKNOWN'),
            'polarity': cand.get('polarity', 'UNKNOWN'),
            'precision': cand.get('precision', 'UNKNOWN'),
            'scope_match': 'MATCH', 'semantic_support': support,
            'evidence_quotes': [{'evidence_id': eid, 'quote': cand['quote']}],
            'reason': 'supported by the source',
            'source_restatement': 'The user states: ' + cand['quote'],
            'source_qualifiers': [cand['quote']],
            'comparisons': {k: 'MATCH' for k in
                            ('subject', 'predicate', 'value', 'polarity',
                             'time', 'modality', 'scope')},
            'component_evidence': {
                'subject': {'evidence_id': eid, 'anchor': cand['quote'],
                            'quote': cand['quote']},
                'relation': {'evidence_id': eid, 'anchor': cand['quote'],
                             'quote': cand['quote']},
                'value': {'evidence_id': eid, 'anchor': cand['quote'],
                          'quote': cand['quote']}},
            'reference_resolution': []})

    def _relation(self, user):
        return json.dumps({
            'relation': self.relation,
            'old_quote': user['old']['fact']['quote'],
            'new_quote': user['new']['fact']['quote'],
            'reason': 'sources explicitly describe a move'})


def make_sources():
    return longmemeval.adapt_sessions(
        'conv1', ['s1', 's2'],
        ['2023/01/01 (Sun) 10:00', '2023/01/02 (Mon) 10:00'],
        [[{'role': 'user', 'content': 'I live in Shanghai'},
          {'role': 'assistant', 'content': 'Noted.'}],
         [{'role': 'user', 'content': 'I live in Beijing now'},
          {'role': 'assistant', 'content': 'Okay.'}]])


def ingest(store, call):
    bridge = EvidenceBridge(store, call)
    outcome = asyncio.run(bridge.ingest_sources(
        make_sources(), model='fake-model'))
    return bridge, outcome


class BridgeIngestTests(unittest.TestCase):
    def test_ingest_creates_namespaced_chunks_and_user_facts(self):
        store, call = EvidenceMemory(), FakeCall()
        bridge, outcome = ingest(store, call)
        self.assertEqual(4, len(store.chunks))            # 2 sessions x 2 turns
        self.assertEqual(2, len(outcome['fact_ids']))
        self.assertEqual(2, len(call.payloads))           # extraction only for user turns
        for fid in outcome['fact_ids']:
            fact = store.facts[fid]
            self.assertEqual('unverified', fact['support_status'])
            self.assertIn('time_reference', fact['raw'])  # qualifiers preserved
            chunk = store.chunks[fact['evidence_id']]
            self.assertEqual('user', chunk['speaker'])
            self.assertTrue(chunk['server_evidence_id'].startswith('ev_'))
            self.assertEqual(fact['raw']['quote'], chunk['text'])
        # replay: same ids, no collision, no duplicates
        again = asyncio.run(bridge.ingest_sources(
            make_sources(), model='fake-model'))
        self.assertEqual(sorted(outcome['fact_ids']), sorted(again['fact_ids']))
        self.assertEqual(4, len(store.chunks))

    def test_server_evidence_id_is_stable_and_unique(self):
        # adapt_sessions does not preassign evidence ids; the bridge derives them
        sources = make_sources()
        self.assertNotIn('evidence_id', sources[0])
        first = server_evidence_id(sources[0])
        self.assertEqual(first, server_evidence_id(dict(sources[0])))
        ids = [server_evidence_id(s) for s in sources]
        self.assertEqual(len(ids), len(set(ids)))
        with self.assertRaises(ValueError):
            server_evidence_id({'session_id': 's'})   # missing identity fields

    def test_extraction_skips_non_user_and_unmatched_quotes(self):
        EXTRACTION['Bad turn'] = {'facts': [
            {'subject': 'assistant', 'predicate': 'p', 'value': 'v',
             'quote': 'Bad turn', 'scope': 'x'},
            {'subject': 'user', 'predicate': 'p', 'value': 'v',
             'quote': 'words not present', 'scope': 'x'}]}
        try:
            store = EvidenceMemory()
            call = FakeCall()
            sources = [{'dataset_id': 'd', 'conversation_id': 'c',
                        'session_id': 's', 'turn_id': '0', 'speaker': 'user',
                        'text': 'Bad turn'}]
            outcome = asyncio.run(EvidenceBridge(store, call).ingest_sources(
                sources, model='fake-model'))
            self.assertEqual([], outcome['fact_ids'])
            self.assertEqual(2, len(outcome['skipped']))
            self.assertEqual({}, store.facts)
        finally:
            EXTRACTION.pop('Bad turn', None)


class BridgeReviewTests(unittest.TestCase):
    def test_review_next_recorded_and_roundtrips(self):
        store = EvidenceMemory()
        bridge, outcome = ingest(store, FakeCall())
        fid = outcome['fact_ids'][0]
        result = asyncio.run(bridge.review_fact(
            fid, model='fake-model', review_id='rev-1'))
        self.assertEqual('REVIEW_NEXT', result['route'])
        self.assertFalse(result['applied'])
        self.assertEqual('review_next', store.facts[fid]['support_status'])
        self.assertEqual(1, len(store.facts[fid]['reviews']))
        # retry with same review_id: idempotent
        again = asyncio.run(bridge.review_fact(
            fid, model='fake-model', review_id='rev-1'))
        self.assertEqual('rev-1', again['review_record']['review_id'])
        self.assertEqual(1, len(store.facts[fid]['reviews']))
        # contradictory retry rejected
        with self.assertRaises(ValueError):
            store.record_review(fid, 'DEFER', ['x'], review_id='rev-1')
        # serialization preserves review history and support status
        restored = EvidenceMemory.from_dict(store.to_dict())
        self.assertEqual('review_next', restored.facts[fid]['support_status'])
        self.assertEqual(store.facts[fid]['reviews'],
                         restored.facts[fid]['reviews'])
        self.assertEqual(4, restored.to_dict()['version'])

    def test_defer_and_technical_error_routes(self):
        store = EvidenceMemory()
        bridge, outcome = ingest(store, FakeCall(review_route='DEFER'))
        fid = outcome['fact_ids'][0]
        result = asyncio.run(bridge.review_fact(fid, model='fake-model'))
        self.assertEqual('DEFER', result['route'])
        self.assertEqual('review_defer', store.facts[fid]['support_status'])

        store2 = EvidenceMemory()
        bridge2, outcome2 = ingest(store2, FakeCall(review_route='TECHNICAL_ERROR'))
        fid2 = outcome2['fact_ids'][0]
        result2 = asyncio.run(bridge2.review_fact(fid2, model='fake-model'))
        self.assertEqual('TECHNICAL_ERROR', result2['route'])
        # a crash is not a review outcome: status untouched, history kept
        self.assertEqual('unverified', store2.facts[fid2]['support_status'])
        self.assertEqual(1, len(store2.facts[fid2]['reviews']))

    def test_record_review_rejects_bad_input(self):
        store = EvidenceMemory()
        bridge, outcome = ingest(store, FakeCall())
        fid = outcome['fact_ids'][0]
        with self.assertRaises(ValueError):
            store.record_review(fid, 'APPROVED', ['x'])
        with self.assertRaises(ValueError):
            store.record_review(fid, 'DEFER', [])
        with self.assertRaises(ValueError):
            store.record_review('missing', 'DEFER', ['x'])


class BridgePairingTests(unittest.TestCase):
    def test_pairing_finds_target_pair_and_records_change(self):
        store = EvidenceMemory()
        bridge, outcome = ingest(store, FakeCall())
        old_fid, new_fid = outcome['fact_ids']            # Shanghai -> Beijing
        pairs = bridge.candidate_pairs(max_candidates=3)
        ids = {(p['old_fact_id'], p['new_fact_id']) for p in pairs['pairs']}
        self.assertIn((old_fid, new_fid), ids)
        pair = next(p for p in pairs['pairs']
                    if p['old_fact_id'] == old_fid and p['new_fact_id'] == new_fid)
        self.assertEqual('BEFORE', pair['time_order'])
        self.assertTrue(pair['candidate_only'])
        # both sides indexed with derived keys, unknowns preserved
        view = pairs['views'][old_fid]
        self.assertEqual('user', view['subject_key'])
        self.assertEqual('live_in', view['attribute_key'])
        self.assertEqual('UNKNOWN', view['object_key'])

        result = asyncio.run(bridge.judge_and_record(
            pair, views=pairs['views'], model='fake-model',
            review_id='pair-1'))
        self.assertEqual('proposed', result['status'])
        self.assertEqual('CHANGE', result['relation'])
        self.assertFalse(result['applied'])
        self.assertIsNotNone(result['change_id'])
        events = store.trace(old_fid)
        self.assertEqual(1, len(events))
        self.assertEqual('candidate', events[0]['status'])
        self.assertEqual('pending', events[0]['decision'])
        self.assertEqual('CHANGE', events[0]['relation'])
        # symmetric lookup from the new end
        self.assertEqual(events, store.trace(new_fid))
        # review_id retry returns the same change; new review appends
        retry = asyncio.run(bridge.judge_and_record(
            pair, views=pairs['views'], model='fake-model',
            review_id='pair-1'))
        self.assertEqual(result['change_id'], retry['change_id'])
        self.assertEqual(1, len(store.trace(old_fid)))
        recorded = store.trace(old_fid)[0]
        with self.assertRaises(ValueError):     # same id, different decision
            store.add_change(old_fid, new_fid, recorded['relation'],
                             recorded['evidence_id'], recorded['reason'],
                             decision='replace', review_id='pair-1')
        # history survives serialization
        restored = EvidenceMemory.from_dict(store.to_dict())
        self.assertEqual([e['seq'] for e in restored.trace(old_fid)],
                         [e['seq'] for e in store.trace(old_fid)])

    def test_unparseable_session_time_leaves_fact_unindexed(self):
        store = EvidenceMemory()
        call = FakeCall()
        sources = [{'dataset_id': 'd', 'conversation_id': 'c',
                    'session_id': 's', 'turn_id': '0', 'speaker': 'user',
                    'text': 'I live in Shanghai'}]         # no session time
        asyncio.run(EvidenceBridge(store, call).ingest_sources(
            sources, model='fake-model'))
        bridge = EvidenceBridge(store, call)
        pairs = bridge.candidate_pairs(max_candidates=3)
        self.assertEqual([], pairs['pairs'])
        self.assertEqual(1, len(pairs['unindexed_fact_ids']))


class BridgeQueryDiagnosticsTests(unittest.TestCase):
    def test_query_hooks_carry_support_status_and_reviews(self):
        store = EvidenceMemory()
        bridge, outcome = ingest(store, FakeCall())
        old_fid, new_fid = outcome['fact_ids']
        asyncio.run(bridge.review_fact(old_fid, model='fake-model'))

        query = QueryProcessor.__new__(QueryProcessor)
        query.dynamic_memory = SimpleNamespace(evidence_memory=store)
        query.config = SimpleNamespace(
            graph=SimpleNamespace(enable_evidence_memory=True))
        candidates = query._evidence_fact_candidates([{'name': 'Shanghai'}])
        self.assertEqual(1, len(candidates))
        self.assertEqual('review_next', candidates[0]['support_status'])
        self.assertEqual(1, len(candidates[0]['reviews']))
        # change-chain traces exist only once a change links the facts
        self.assertEqual([], query._evidence_candidate_traces([{'name': 'Shanghai'}]))
        pairs = bridge.candidate_pairs(max_candidates=3)
        pair = next(p for p in pairs['pairs']
                    if p['old_fact_id'] == old_fid and p['new_fact_id'] == new_fid)
        asyncio.run(bridge.judge_and_record(
            pair, views=pairs['views'], model='fake-model'))
        bundles = query._evidence_candidate_traces([{'name': 'Shanghai'}])
        self.assertEqual(1, len(bundles))
        reviewed = [f for f in bundles[0]['facts'] if f['id'] == old_fid]
        self.assertEqual('review_next', reviewed[0]['support_status'])
        self.assertEqual(1, len(reviewed[0]['reviews']))
        self.assertEqual(1, len(bundles[0]['changes']))
        # switch off: diagnostics empty, original behavior intact
        query.config.graph.enable_evidence_memory = False
        self.assertEqual([], query._evidence_fact_candidates([{'name': 'Shanghai'}]))
        self.assertEqual([], query._evidence_candidate_traces([{'name': 'Shanghai'}]))


class BridgeCompatibilityTests(unittest.TestCase):
    def test_version3_payload_still_loads(self):
        store = EvidenceMemory()
        _, outcome = ingest(store, FakeCall())
        payload = store.to_dict()
        self.assertEqual(4, payload['version'])
        legacy = dict(payload, version=3)
        for fact in legacy['facts'].values():
            fact.pop('reviews', None)
            fact['support_status'] = 'unverified'
        restored = EvidenceMemory.from_dict(legacy)
        self.assertEqual(2, len(restored.facts))
        self.assertEqual('unverified',
                         restored.facts[outcome['fact_ids'][0]]['support_status'])
        # round trip upgrades back to version 4
        self.assertEqual(4, restored.to_dict()['version'])

    def test_vendored_modules_have_no_platform_locks(self):
        import coregraph.evidence_review as pkg
        import pathlib
        for path in pathlib.Path(pkg.__file__).parent.glob('*.py'):
            self.assertNotIn('import fcntl', path.read_text(encoding='utf-8'),
                             path.name)


if __name__ == '__main__':
    unittest.main()
