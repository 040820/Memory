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

    def test_same_pair_different_relations_kept(self):
        s=m.EvidenceMemory(); a=s.add_chunk({'session_id':'s1','chunk_id':0,'text':'t'})
        x=s.add_fact({'src':'I','relation':'live in','tgt':'Shanghai'},a)
        y=s.add_fact({'src':'I','relation':'work in','tgt':'Shanghai'},a)
        self.assertNotEqual(x,y)
        self.assertEqual({x,y},{f['id'] for f in s.find_by_entities(['Shanghai'])})

    def test_same_relation_two_sources_kept(self):
        s=m.EvidenceMemory()
        a=s.add_chunk({'session_id':'s1','chunk_id':0,'text':'I live in Shanghai'})
        b=s.add_chunk({'session_id':'s2','chunk_id':0,'text':'still in Shanghai'})
        x=s.add_fact({'src':'I','relation':'live in','tgt':'Shanghai'},a)
        y=s.add_fact({'src':'I','relation':'live in','tgt':'Shanghai'},b)
        self.assertNotEqual(x,y)
        self.assertEqual(2,len(s.find_by_entities(['Shanghai'])))

    def test_change_decision_field_and_status_constraint(self):
        s=m.EvidenceMemory(); a=s.add_chunk({'session_id':'a','chunk_id':0,'text':'old'})
        b=s.add_chunk({'session_id':'b','chunk_id':0,'text':'new'})
        x=s.add_fact({'src':'I','relation':'live in','tgt':'Shanghai'},a)
        y=s.add_fact({'src':'I','relation':'live in','tgt':'Beijing'},b)
        cid=s.add_change(x,y,'change',b,'moved')
        self.assertEqual('pending',s.changes[cid]['decision'])
        self.assertEqual('candidate',s.changes[cid]['status'])
        cid2=s.add_change(x,y,'supersede',b,'reviewed',decision='update')
        self.assertEqual('update',s.changes[cid2]['decision'])
        t=m.EvidenceMemory.from_dict(s.to_dict())
        self.assertEqual('update',t.changes[cid2]['decision'])
        self.assertEqual('pending',t.changes[cid]['decision'])
        with self.assertRaises(ValueError): s.add_change(x,y,'change',b,'moved',status='confirmed')

    def test_legacy_change_without_decision_loads(self):
        s=m.EvidenceMemory(); a=s.add_chunk({'session_id':'a','chunk_id':0,'text':'old'})
        b=s.add_chunk({'session_id':'b','chunk_id':0,'text':'new'})
        x=s.add_fact({'src':'I','relation':'live in','tgt':'Shanghai'},a)
        y=s.add_fact({'src':'I','relation':'live in','tgt':'Beijing'},b)
        cid=s.add_change(x,y,'change',b,'moved')
        data=s.to_dict()
        for change in data['changes'].values(): change.pop('decision',None)
        t=m.EvidenceMemory.from_dict(data)
        self.assertIn(cid,t.changes)
        self.assertEqual('pending',t.changes[cid]['decision'])

    def test_namespace_source_id_and_replay_idempotent(self):
        raw=[{'session_id':'s1','chunk_id':7,'text':'a'},{'session_id':'s1','chunk_id':8,'text':'b'}]
        n1=m.namespace_chunks(raw); n2=m.namespace_chunks(raw)
        self.assertEqual([c['chunk_id'] for c in n1],[c['chunk_id'] for c in n2])
        self.assertEqual(7,n1[0]['source_chunk_id']); self.assertEqual(8,n1[1]['source_chunk_id'])
        n3=m.namespace_chunks(n1)
        self.assertEqual([c['chunk_id'] for c in n1],[c['chunk_id'] for c in n3])
        self.assertEqual(7,n3[0]['source_chunk_id'])

    def test_cross_session_same_source_id_distinct(self):
        chunks=[{'session_id':'sA','chunk_id':0,'text':'x'},{'session_id':'sB','chunk_id':0,'text':'y'}]
        n=m.namespace_chunks(chunks)
        self.assertNotEqual(n[0]['chunk_id'],n[1]['chunk_id'])
        s=m.EvidenceMemory(); i1=s.add_chunk(n[0]); i2=s.add_chunk(n[1])
        self.assertNotEqual(i1,i2); self.assertEqual(2,len(s.chunks))

    def test_session_replay_append_and_content_change(self):
        s=m.EvidenceMemory()
        c0={'session_id':'s1','chunk_id':0,'text':'first'}
        c1={'session_id':'s1','chunk_id':1,'text':'second'}
        s.add_chunk(c0); s.add_chunk(c1)
        s.add_chunk(c0); s.add_chunk(c1)          # identical full replay: idempotent
        self.assertEqual(2,len(s.chunks))
        s.add_chunk({'session_id':'s1','chunk_id':2,'text':'third'})  # new ordinal append: accepted
        self.assertEqual(3,len(s.chunks))
        with self.assertRaises(ValueError):        # changed content at an existing ordinal: rejected
            s.add_chunk({'session_id':'s1','chunk_id':1,'text':'rewritten'})

if __name__=='__main__': unittest.main()
