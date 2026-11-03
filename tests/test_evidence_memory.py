import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('evidence_memory',Path(__file__).resolve().parents[1]/'coregraph'/'evidence_memory.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class EvidenceMemoryTests(unittest.TestCase):
    def test_distinct_chunks_facts_and_idempotency(self):
        s=m.EvidenceMemory()
        a=s.add_chunk({'session_id':'s1','chunk_id':0,'text':'I work in Beijing.'})
        b=s.add_chunk({'session_id':'s2','chunk_id':0,'text':'I visited Beijing.'})
        self.assertNotEqual(a,b)
        x=s.add_fact({'src':'I','relation':'work in','tgt':'Beijing'},a)
        y=s.add_fact({'src':'I','relation':'visited','tgt':'Beijing'},b)
        self.assertEqual(x,s.add_fact({'src':'I','relation':'work in','tgt':'Beijing'},a))
        self.assertEqual({x,y},{f['id'] for f in s.find_by_entities(['Beijing'])})
        self.assertNotEqual(x,y)
    def test_chunk_collision_rejected(self):
        s=m.EvidenceMemory(); s.add_chunk({'session_id':'s','chunk_id':0,'text':'a'})
        with self.assertRaises(ValueError): s.add_chunk({'session_id':'s','chunk_id':0,'text':'b'})
    def test_change_bidirectional_and_reload(self):
        s=m.EvidenceMemory(); a=s.add_chunk({'session_id':'a','chunk_id':0,'text':'Shanghai'})
        b=s.add_chunk({'session_id':'b','chunk_id':0,'text':'I moved to Beijing'})
        x=s.add_fact({'src':'I','relation':'live in','tgt':'Shanghai'},a)
        y=s.add_fact({'src':'I','relation':'live in','tgt':'Beijing'},b)
        s.add_change(x,y,'change',b,'moved',status='candidate')
        t=m.EvidenceMemory.from_dict(s.to_dict())
        self.assertEqual(t.trace(x),t.trace(y))
        self.assertEqual('candidate',t.trace(x)[0]['status'])
        self.assertEqual('I moved to Beijing',t.chunks[b]['text'])
        bundle=t.evidence_bundle(x)
        self.assertEqual({x,y},{f['id'] for f in bundle['facts']})
        self.assertEqual({a,b},set(bundle['chunks']))
    def test_chunk_namespace_is_stable_across_sessions(self):
        chunks=[{'session_id':'a','chunk_id':0,'text':'x'}, {'session_id':'b','chunk_id':1,'text':'y'}]
        initial=m.namespace_chunks(chunks)
        incremental=m.namespace_chunks([{'session_id':'b','chunk_id':0,'text':'y'}])
        self.assertNotEqual(initial[0]['chunk_id'],initial[1]['chunk_id'])
        self.assertEqual(initial[1]['chunk_id'],incremental[0]['chunk_id'])

    def test_bad_source_rejected(self):
        s=m.EvidenceMemory()
        with self.assertRaises(ValueError): s.add_fact({'src':'I','relation':'live','tgt':'x'},'missing')

if __name__=='__main__': unittest.main()
