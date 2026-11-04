import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import networkx as nx
from coregraph.dynamic_memory import DynamicMemory
from coregraph.evidence_memory import EvidenceMemory
from query.query_processor import QueryProcessor

class IntegrationTests(unittest.TestCase):
    def make_memory(self):
        obj=DynamicMemory.__new__(DynamicMemory)
        obj.graph_builder=SimpleNamespace(graph=nx.DiGraph())
        obj.chunk_storage={}; obj.entity_name_to_index={}; obj.evidence_memory=EvidenceMemory()
        return obj
    def test_capture_separate_relationships_and_persist(self):
        mem=self.make_memory()
        chunks=[{'session_id':'s1','chunk_id':'1','text':'works in Beijing'}, {'session_id':'s2','chunk_id':'2','text':'visits Beijing'}]
        rels=[{'src':'user','relation':'works in','tgt':'Beijing','chunk_id':'1'}, {'src':'user','relation':'visits','tgt':'Beijing','chunk_id':'2'}]
        mem._capture_evidence_relationships(chunks,rels,{})
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'graph.pkl'); mem.save_graph(path)
            new=self.make_memory(); new.load_graph(path)
            self.assertEqual(2,len(new.evidence_memory.find_by_entities(['Beijing'])))
            self.assertTrue(new.evidence_memory.complete)
            query=QueryProcessor.__new__(QueryProcessor); query.dynamic_memory=new
            query.config=SimpleNamespace(graph=SimpleNamespace(enable_evidence_memory=True))
            self.assertEqual(2,len(query._evidence_fact_candidates([{'name':'Beijing'}])))
            query.config.graph.enable_evidence_memory=False
            self.assertEqual([],query._evidence_fact_candidates([{'name':'Beijing'}]))
    def test_old_graph_marked_incomplete(self):
        import pickle
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'legacy.pkl')
            with open(path,'wb') as f: pickle.dump({'graph':nx.DiGraph(),'chunk_storage':{},'entity_name_to_index':{}},f)
            mem=self.make_memory(); mem.load_graph(path)
            self.assertFalse(mem.evidence_memory.complete)

    def test_old_format_pickle_marked_incomplete(self):
        import pickle
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'oldformat.pkl')
            with open(path,'wb') as f: pickle.dump(nx.DiGraph(),f)
            mem=self.make_memory(); mem.load_graph(path)
            self.assertFalse(mem.evidence_memory.complete)

    def test_capture_survives_relationship_dedup(self):
        from coregraph.dialogue_extractor import DialogueExtractor
        mem=self.make_memory()
        chunks=[{'session_id':'s1','chunk_id':'c1','text':'I live in Shanghai'},
                {'session_id':'s1','chunk_id':'c2','text':'I live in Shanghai now'}]
        rels=[{'src':'I','relation':'live in','tgt':'Shanghai','chunk_id':'c1'},
              {'src':'I','relation':'live in','tgt':'Shanghai','chunk_id':'c2'}]
        mem._capture_evidence_relationships(chunks,rels,{})
        self.assertEqual(2,len(mem.evidence_memory.find_by_entities(['Shanghai'])))
        ext=DialogueExtractor(llm_manager=None)
        deduped=ext.deduplicate_relationships([dict(r) for r in rels],{},0.9)
        self.assertEqual(1,len(deduped))                      # graph side merges
        self.assertEqual(2,len(mem.evidence_memory.find_by_entities(['Shanghai'])))  # store keeps both

    def test_fact_recall_when_graph_edge_absent(self):
        mem=self.make_memory()
        chunks=[{'session_id':'s1','chunk_id':'c1','text':'user lives in Shanghai'}]
        rels=[{'src':'user','relation':'lives in','tgt':'Shanghai','chunk_id':'c1'}]
        mem._capture_evidence_relationships(chunks,rels,{})
        mem.graph_builder.graph.add_node('user')              # edge merged away from the graph
        mem.graph_builder.graph.add_node('Shanghai')
        found=mem.evidence_memory.find_by_entities(['Shanghai'])
        self.assertEqual(1,len(found))
        self.assertEqual('lives in',found[0]['relation'])     # raw predicate preserved

    def test_canonical_alias_recall(self):
        mem=self.make_memory()
        chunks=[{'session_id':'s1','chunk_id':'c1','text':'t'}]
        rels=[{'src':'user','relation':'lives in','tgt':'Shanghai','chunk_id':'c1'}]
        mem._capture_evidence_relationships(chunks,rels,{'user':'User'})
        self.assertEqual(1,len(mem.evidence_memory.find_by_entities(['User'])))
        self.assertEqual(1,len(mem.evidence_memory.find_by_entities(['user'])))

    def test_complete_flag_follows_switch(self):
        from init.config import Config
        cfg_off=Config(); cfg_off.graph.enable_evidence_memory=False
        mem_off=DynamicMemory(cfg_off)
        self.assertFalse(mem_off.evidence_memory.complete)
        cfg_on=Config(); cfg_on.graph.enable_evidence_memory=True
        mem_on=DynamicMemory(cfg_on)
        self.assertTrue(mem_on.evidence_memory.complete)

    def test_change_history_survives_pickle(self):
        mem=self.make_memory()
        chunks=[{'session_id':'s1','chunk_id':'c1','text':'I live in Shanghai'},
                {'session_id':'s2','chunk_id':'c2','text':'I moved to Beijing'}]
        rels=[{'src':'I','relation':'live in','tgt':'Shanghai','chunk_id':'c1'},
              {'src':'I','relation':'live in','tgt':'Beijing','chunk_id':'c2'}]
        mem._capture_evidence_relationships(chunks,rels,{})
        ev=mem.evidence_memory
        by_tgt={f['tgt']:f for f in ev.find_by_entities(['Shanghai','Beijing'])}
        x,y=by_tgt['Shanghai']['id'],by_tgt['Beijing']['id']
        ev.add_change(x,y,'change',by_tgt['Beijing']['evidence_id'],'moved',decision='keep')
        ev.add_change(x,y,'change',by_tgt['Beijing']['evidence_id'],'moved',decision='replace')
        ev.add_change(x,y,'change',by_tgt['Beijing']['evidence_id'],'moved',decision='keep')
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'graph.pkl'); mem.save_graph(path)
            new=self.make_memory(); new.load_graph(path)
            events=new.evidence_memory.trace(x)
            self.assertEqual(3,len(events))
            self.assertEqual([0,1,2],[e['seq'] for e in events])
            self.assertEqual('keep',events[-1]['decision'])
            self.assertEqual(events,new.evidence_memory.trace(y))

    def test_summary_not_updated_when_chunk_collision_fails(self):
        # 问题6回归：chunk 冲突必须在校验阶段就失败，摘要等副作用不得先发生
        import asyncio
        from coregraph.dynamic_memory import DynamicMemory
        from coregraph.evidence_memory import EvidenceMemory, namespace_chunks

        mem = DynamicMemory.__new__(DynamicMemory)
        mem.config = SimpleNamespace(
            retriever=SimpleNamespace(enable_summary=True),
            graph=SimpleNamespace(enable_evidence_memory=True,
                                  entity_merge_threshold=0.85,
                                  relationship_merge_threshold=1.0),
            chunk=SimpleNamespace(dialogue_input=True))
        # 预置冲突：同 session 同序号、不同内容
        prior = namespace_chunks([{'session_id':'s1','chunk_id':0,'text':'old text'}])
        mem.chunk_storage = {str(prior[0]['chunk_id']): prior[0]}
        mem.evidence_memory = EvidenceMemory()
        mem.chunk_processor = SimpleNamespace(
            process_corpus=lambda corpus: [{'session_id':'s1','chunk_id':0,'text':'new text'}])
        summary_calls = []
        async def fake_summary(session_data):
            summary_calls.append(session_data['session_id'])
        mem._update_summary_for_session = fake_summary
        mem.llm_manager = SimpleNamespace(
            model='m',
            cost_manager=SimpleNamespace(get_costs=lambda: SimpleNamespace(
                total_prompt_tokens=0, total_completion_tokens=0)))
        mem.cost_manager = SimpleNamespace(update_summary_generation_cost=lambda *a: None)

        corpus = [{'session_id':'s1','session_time':'t','context':'...'}]
        with self.assertRaises(ValueError):
            asyncio.run(mem.add_single_session(corpus))
        self.assertEqual([], summary_calls)   # 冲突失败时摘要未被触碰

if __name__=='__main__': unittest.main()
