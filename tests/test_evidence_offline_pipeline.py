"""Offline end-to-end test of the V2 memory organization.

Runs chunking -> session namespacing -> extraction -> pre-dedup evidence capture
-> graph build -> pickle save/load -> unified query with a FakeLLM, so no online
model call is needed. Also runs the switch-off control to confirm the original
behavior is preserved.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path

from init.config import Config
from coregraph.dynamic_memory import DynamicMemory
from coregraph.evidence_memory import namespace_chunks
from chunking.chunk_processor import ChunkProcessor
from query.query_processor import QueryProcessor
from query.triple_reranker import TripleReranker
from utils.time_statistic import QueryTimeStatistic
from utils.cost_manager import QueryCostManager, TokenCostManager


RESP_S1 = (
    '("entity"|User|person|the user)##'
    '("entity"|Beijing|location|the city)##'
    '("relationship"|2020-01-01|s1|User|Beijing|lives in|5)##END'
)
RESP_S2 = (
    '("entity"|User|person|the user)##'
    '("entity"|Beijing|location|the city)##'
    '("relationship"|2021-06-01|s2|User|Beijing|visited|4)##END'
)


class FakeLLM:
    """Canned stand-in for LLMManager; deterministic and offline."""

    def __init__(self, query_entities=None):
        self.model = 'fake-offline'
        self.enable_concurrent = True
        self.max_concurrent = 4
        self.cost_manager = TokenCostManager(max_budget=100.0)
        self.query_entities = list(query_entities) if query_entities else ['Beijing']

    async def batch_generate(self, prompts, progress_bar=None, **kwargs):
        responses = []
        for prompt in prompts:
            if progress_bar is not None:
                progress_bar.update(1)
            responses.append(RESP_S2 if '"session_id": "s2"' in prompt else RESP_S1)
        return responses

    async def generate(self, prompt, **kwargs):
        return 'OFFLINE-ANSWER'

    async def extract_entities(self, text, session_time=''):
        return [{'entity': name, 'type': 'unknown'} for name in self.query_entities]


def make_config(enable_evidence):
    cfg = Config()
    cfg.chunk.dialogue_input = True
    cfg.graph.force = True
    cfg.graph.add = False
    cfg.graph.enable_evidence_memory = enable_evidence
    cfg.retriever.enable_summary = False
    cfg.retriever.enable_visual = False
    cfg.retriever.enable_CogniRank = False
    cfg.embedding.api_type = 'openai'  # keep EmbeddingManager init offline-safe if constructed
    return cfg


def make_corpus():
    return [
        {'session_id': 's1', 'session_time': '2020-01-01',
         'context': '"user": "I live in Beijing."\n"assistant": "Got it."'},
        {'session_id': 's2', 'session_time': '2021-06-01',
         'context': '"user": "I visited Beijing."\n"assistant": "Nice."'},
    ]


def build_memory(cfg, tmp):
    mem = DynamicMemory(cfg, base_dir=tmp)
    fake = FakeLLM()
    mem.set_extractors(fake, embedding_manager=None)
    mem.chunk_processor = ChunkProcessor(cfg.chunk, data_type=getattr(cfg, 'data_type', 'LongmemEval'))
    chunks = mem.chunk_processor.process_corpus(make_corpus())
    if getattr(cfg.graph, 'enable_evidence_memory', False):
        chunks = namespace_chunks(chunks)
    asyncio.run(mem._build_graph(chunks))
    return mem, fake, chunks


def make_query_processor(cfg, fake, mem):
    qp = QueryProcessor.__new__(QueryProcessor)
    qp.config = cfg
    qp.llm = fake
    qp.dynamic_memory = mem
    qp.embedding_manager = None      # force the offline string-matching fallback
    qp.summary_retriever = None
    qp.visualizer = None
    qp.triple_reranker = TripleReranker(cfg)
    qp.time_manager = QueryTimeStatistic()
    qp.cost_manager = QueryCostManager(max_budget=fake.cost_manager.max_budget)
    return qp


class OfflinePipelineTests(unittest.TestCase):
    def test_pipeline_with_evidence_switch_on(self):
        cfg = make_config(True)
        with tempfile.TemporaryDirectory() as tmp:
            mem, fake, chunks = build_memory(cfg, tmp)

            # 1) chunks: namespaced, source ids preserved, cross-session distinct
            self.assertEqual(2, len(mem.chunk_storage))
            stored_ids = set(mem.chunk_storage)
            for chunk in mem.chunk_storage.values():
                self.assertTrue(str(chunk['chunk_id']).startswith('session_chunk_'))
                self.assertIn('source_chunk_id', chunk)
            self.assertEqual({0, 1}, {c['source_chunk_id'] for c in mem.chunk_storage.values()})

            # 2) facts captured pre-dedup: both relations kept, raw predicates intact
            self.assertTrue(mem.evidence_memory.complete)
            facts = mem.evidence_memory.find_by_entities(['Beijing'])
            self.assertEqual(2, len(facts))
            self.assertEqual({'lives in', 'visited'}, {f['relation'] for f in facts})
            for f in facts:
                self.assertEqual(f['relation'], f['raw']['relation'])
                self.assertIn(f['evidence_id'], mem.evidence_memory.chunks)

            # 3) original DiGraph merges same-pair edges into one, while the
            #    evidence store keeps both source-linked facts (core invariant)
            self.assertIn('Beijing', mem.entity_name_to_index)
            self.assertEqual(1, mem.graph_builder.graph.number_of_edges())
            self.assertEqual(2, len(mem.evidence_memory.find_by_entities(['Beijing'])))

            # 4) save / reload round trip
            pkl = str(Path(tmp) / 'graph.pkl')
            mem.save_graph(pkl)
            mem2 = DynamicMemory(cfg, base_dir=tmp)
            mem2.load_graph(pkl)
            self.assertTrue(mem2.evidence_memory.complete)
            self.assertEqual(2, len(mem2.evidence_memory.find_by_entities(['Beijing'])))
            self.assertEqual(stored_ids, set(mem2.chunk_storage))

            # 5) replay of the whole build is idempotent for the evidence store
            mem2.set_extractors(fake, embedding_manager=None)
            asyncio.run(mem2._build_graph(chunks))
            self.assertEqual(2, len(mem2.evidence_memory.find_by_entities(['Beijing'])))
            self.assertEqual(2, len(mem2.chunk_storage))

            # 6) offline unified query exposes evidence candidates as diagnostics
            qp = make_query_processor(cfg, fake, mem2)
            result = asyncio.run(qp.process_query('Where do I live now?'))
            self.assertEqual('OFFLINE-ANSWER', result['answer'])
            self.assertEqual(2, len(result['evidence_fact_candidates']))
            self.assertTrue(result['chunks'])          # original chunk back-reference works
            self.assertTrue(result['triples'])

    def test_pipeline_with_evidence_switch_off(self):
        cfg = make_config(False)
        with tempfile.TemporaryDirectory() as tmp:
            mem, fake, _ = build_memory(cfg, tmp)

            # original graph path unaffected; offline store flagged incomplete
            self.assertFalse(mem.evidence_memory.complete)
            self.assertEqual([], mem.evidence_memory.find_by_entities(['Beijing']))
            self.assertIn('Beijing', mem.entity_name_to_index)
            self.assertEqual(1, mem.graph_builder.graph.number_of_edges())

            qp = make_query_processor(cfg, fake, mem)
            result = asyncio.run(qp.process_query('Where do I live now?'))
            self.assertEqual('OFFLINE-ANSWER', result['answer'])
            self.assertEqual([], result['evidence_fact_candidates'])
            self.assertEqual([], result['evidence_candidate_traces'])
            # chunk ids stay original (global ints) when the switch is off
            for chunk in mem.chunk_storage.values():
                self.assertNotIn('source_chunk_id', chunk)

    def test_full_batch_then_single_session_replay(self):
        # Repro for review issue: full-batch build of A+B, then replay session B
        # alone through add_single_session. source numbering differs (1 vs 0);
        # the replay must stay idempotent instead of raising a collision.
        cfg = make_config(True)
        with tempfile.TemporaryDirectory() as tmp:
            mem, fake, _ = build_memory(cfg, tmp)
            self.assertEqual(2, len(mem.chunk_storage))
            self.assertEqual(1, len({c['source_chunk_id'] for c in mem.chunk_storage.values()
                                     if c['session_id'] == 's2'}))

            asyncio.run(mem.add_single_session([make_corpus()[1]]))  # must not raise

            self.assertEqual(2, len(mem.chunk_storage))
            self.assertTrue(mem.evidence_memory.complete)
            self.assertEqual(2, len(mem.evidence_memory.find_by_entities(['Beijing'])))
            # session s2 chunk now carries the per-session numbering of the replay run
            b_chunks = [c for c in mem.chunk_storage.values() if c['session_id'] == 's2']
            self.assertEqual(0, b_chunks[0]['source_chunk_id'])

    def test_off_graph_fact_reachable_via_raw_query_entity(self):
        # Repro for review issue: a fact whose entity never entered the graph must
        # still be found when the question mentions that entity (raw query entity
        # path), even though graph entity matching cannot surface it.
        cfg = make_config(True)
        with tempfile.TemporaryDirectory() as tmp:
            mem, _, _ = build_memory(cfg, tmp)
            ev = mem.evidence_memory
            extra = ev.add_chunk({'session_id': 's9', 'chunk_id': 0,
                                  'text': 'User relocated to Shanghai.'})
            fid = ev.add_fact({'src': 'User', 'relation': 'relocated to',
                               'tgt': 'Shanghai'}, extra)
            self.assertNotIn('Shanghai', mem.entity_name_to_index)  # off-graph entity

            fake = FakeLLM(query_entities=['Shanghai'])
            qp = make_query_processor(cfg, fake, mem)
            result = asyncio.run(qp.process_query('Where does the user live now?'))
            ids = {f['id'] for f in result['evidence_fact_candidates']}
            self.assertIn(fid, ids)
            self.assertEqual('OFFLINE-ANSWER', result['answer'])


if __name__ == '__main__':
    unittest.main()
