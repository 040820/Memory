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

if __name__=='__main__': unittest.main()
