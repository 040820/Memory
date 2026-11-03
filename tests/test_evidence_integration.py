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

if __name__=='__main__': unittest.main()
