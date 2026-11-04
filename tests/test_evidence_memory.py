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

    def test_v1_change_record_loads_and_rekeys(self):
        # a genuine version-1 payload: change id WITHOUT decision in the hash
        s=m.EvidenceMemory(); a=s.add_chunk({'session_id':'a','chunk_id':0,'text':'old'})
        b=s.add_chunk({'session_id':'b','chunk_id':0,'text':'new'})
        x=s.add_fact({'src':'I','relation':'live in','tgt':'Shanghai'},a)
        y=s.add_fact({'src':'I','relation':'live in','tgt':'Beijing'},b)
        legacy_cid=m._id('change_',[x,y,'change',b,'moved'])
        payload={'version':1,'chunks':s.chunks,'facts':s.facts,'complete':True,
                 'changes':{legacy_cid:{'id':legacy_cid,'old_fact_id':x,'new_fact_id':y,
                                        'relation':'change','evidence_id':b,'reason':'moved',
                                        'status':'candidate'}}}
        t=m.EvidenceMemory.from_dict(payload)
        new_cid=list(t.changes)[0]
        self.assertNotEqual(legacy_cid,new_cid)          # re-keyed to version-2 id
        self.assertEqual('pending',t.changes[new_cid]['decision'])
        self.assertEqual(t.trace(x),t.trace(y))
        # round trip as version 2 afterwards
        t2=m.EvidenceMemory.from_dict(t.to_dict())
        self.assertEqual(t.changes,t2.changes)

    def test_decision_revert_kept_and_review_retry_idempotent(self):
        s=m.EvidenceMemory(); a=s.add_chunk({'session_id':'a','chunk_id':0,'text':'old'})
        b=s.add_chunk({'session_id':'b','chunk_id':0,'text':'new'})
        x=s.add_fact({'src':'I','relation':'live in','tgt':'Shanghai'},a)
        y=s.add_fact({'src':'I','relation':'live in','tgt':'Beijing'},b)
        # keep -> replace -> keep: all three events survive, ordered, final = keep
        k1=s.add_change(x,y,'change',b,'moved',decision='keep')
        k2=s.add_change(x,y,'change',b,'moved',decision='replace')
        k3=s.add_change(x,y,'change',b,'moved',decision='keep')   # revert
        self.assertEqual(3,len({k1,k2,k3}))
        events=s.trace(x)
        self.assertEqual([0,1,2],[e['revision'] for e in events])
        self.assertEqual([0,1,2],[e['seq'] for e in events])
        self.assertEqual('keep',events[-1]['decision'])           # latest identified
        # review_id is the retry idempotency key
        r1=s.add_change(x,y,'change',b,'moved',decision='replace',review_id='rev-1')
        r2=s.add_change(x,y,'change',b,'moved',decision='replace',review_id='rev-1')  # retry
        self.assertEqual(r1,r2)
        self.assertEqual(4,len(s.trace(x)))                      # retry adds nothing
        with self.assertRaises(ValueError):                      # contradictory retry
            s.add_change(x,y,'change',b,'moved',decision='keep',review_id='rev-1')
        r3=s.add_change(x,y,'change',b,'moved',decision='keep',review_id='rev-2')     # new review
        self.assertNotEqual(r1,r3)
        self.assertEqual(5,len(s.trace(x)))
        # history survives serialization
        t=m.EvidenceMemory.from_dict(s.to_dict())
        self.assertEqual([e['seq'] for e in t.trace(x)],[e['seq'] for e in s.trace(x)])
        self.assertEqual('keep',t.trace(x)[-1]['decision'])

    def test_v2_change_record_loads_and_rekeys(self):
        # a genuine version-2 payload: id = hash(core + decision), no order metadata
        s=m.EvidenceMemory(); a=s.add_chunk({'session_id':'a','chunk_id':0,'text':'old'})
        b=s.add_chunk({'session_id':'b','chunk_id':0,'text':'new'})
        x=s.add_fact({'src':'I','relation':'live in','tgt':'Shanghai'},a)
        y=s.add_fact({'src':'I','relation':'live in','tgt':'Beijing'},b)
        legacy_cid=m._id('change_',[x,y,'change',b,'moved','keep'])
        payload={'version':2,'chunks':s.chunks,'facts':s.facts,'complete':True,
                 'changes':{legacy_cid:{'id':legacy_cid,'old_fact_id':x,'new_fact_id':y,
                                        'relation':'change','evidence_id':b,'reason':'moved',
                                        'status':'candidate','decision':'keep'}}}
        t=m.EvidenceMemory.from_dict(payload)
        new_cid=list(t.changes)[0]
        self.assertNotEqual(legacy_cid,new_cid)
        record=t.changes[new_cid]
        self.assertEqual('keep',record['decision'])
        self.assertEqual(0,record['revision']); self.assertEqual(0,record['seq'])
        self.assertEqual(t.trace(x),t.trace(y))

    def test_full_batch_then_session_replay_no_collision(self):
        # full-batch numbering vs per-session numbering: only source_chunk_id differs
        full=m.namespace_chunks([{'session_id':'A','chunk_id':0,'text':'xa'},
                                 {'session_id':'B','chunk_id':1,'text':'xb'}])
        self.assertEqual(1,full[1]['source_chunk_id'])   # B numbered across the corpus
        replay=m.namespace_chunks([{'session_id':'B','chunk_id':0,'text':'xb'}])
        self.assertEqual(0,replay[0]['source_chunk_id'])  # per-session numbering restarts
        self.assertEqual(full[1]['chunk_id'],replay[0]['chunk_id'])  # generated id stable
        s=m.EvidenceMemory()
        for c in full: s.add_chunk(c)
        for c in replay: s.add_chunk(c)                  # must not raise
        self.assertEqual(2,len(s.chunks))
        # genuine content change is still rejected
        bad=m.namespace_chunks([{'session_id':'B','chunk_id':0,'text':'rewritten'}])
        with self.assertRaises(ValueError): s.add_chunk(bad[0])
        # chunk_conflicts ignores run-scoped provenance only
        a={'chunk_id':'x','text':'t','source_chunk_id':0}
        b_={'chunk_id':'x','text':'t','source_chunk_id':9}
        c_={'chunk_id':'x','text':'other','source_chunk_id':0}
        self.assertFalse(m.chunk_conflicts(a,b_)); self.assertTrue(m.chunk_conflicts(a,c_))

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

    def test_readd_fact_preserves_review_history(self):
        # 问题2回归：同一事实重复写入不得清空已有审查记录
        s = m.EvidenceMemory(); a = s.add_chunk({'session_id':'s1','chunk_id':0,'text':'I live in Shanghai'})
        raw = {'src':'I','relation':'live in','tgt':'Shanghai'}
        x = s.add_fact(raw, a)
        s.record_review(x, 'REVIEW_NEXT', ['ok'], evidence={'evidence_spans':[{'start':0,'end':5}]})
        y = s.add_fact(dict(raw), a)          # 幂等重放
        self.assertEqual(x, y)
        self.assertEqual('review_next', s.facts[x]['support_status'])
        self.assertEqual(1, len(s.facts[x]['reviews']))
        self.assertEqual([{'start':0,'end':5}], s.facts[x]['reviews'][0]['evidence']['evidence_spans'])
        # 序列化往返同样保留
        t = m.EvidenceMemory.from_dict(s.to_dict())
        self.assertEqual('review_next', t.facts[x]['support_status'])
        self.assertEqual(1, len(t.facts[x]['reviews']))

    def test_namespace_with_stable_ordinal_key(self):
        # 问题3回归：同会话分批输入，用稳定 turn_id 做序号不冲突
        batch1 = m.namespace_chunks(
            [{'session_id':'s1','chunk_id':'0','text':'first'},
             {'session_id':'s1','chunk_id':'1','text':'second'}],
            ordinal_key='chunk_id')
        batch2 = m.namespace_chunks(
            [{'session_id':'s1','chunk_id':'2','text':'third'}],
            ordinal_key='chunk_id')
        self.assertNotEqual(batch1[0]['chunk_id'], batch2[0]['chunk_id'])
        self.assertNotEqual(batch1[1]['chunk_id'], batch2[0]['chunk_id'])
        s = m.EvidenceMemory()
        for c in batch1 + batch2: s.add_chunk(c)
        self.assertEqual(3, len(s.chunks))
        # 重放第一批仍幂等
        for c in m.namespace_chunks(
                [{'session_id':'s1','chunk_id':'0','text':'first'},
                 {'session_id':'s1','chunk_id':'1','text':'second'}],
                ordinal_key='chunk_id'):
            s.add_chunk(c)
        self.assertEqual(3, len(s.chunks))
        # 缺稳定序号直接报错，不退回批次位置
        with self.assertRaises(ValueError):
            m.namespace_chunks([{'session_id':'s1','text':'x'}], ordinal_key='chunk_id')

    def test_record_review_and_change_persist_evidence(self):
        # 问题4回归：审查结论与关系事件都要持久化原文依据
        s = m.EvidenceMemory()
        a = s.add_chunk({'session_id':'a','chunk_id':0,'text':'I live in Shanghai'})
        b = s.add_chunk({'session_id':'b','chunk_id':0,'text':'I moved to Beijing'})
        x = s.add_fact({'src':'I','relation':'live in','tgt':'Shanghai'}, a)
        y = s.add_fact({'src':'I','relation':'live in','tgt':'Beijing'}, b)
        s.record_review(x, 'DEFER', ['SCOPE_UNCLEAR'],
                        evidence={'evidence_spans':[{'start':0,'end':17}],
                                  'component_evidence_check':'PASS'})
        cid = s.add_change(x, y, 'change', b, 'moved',
                           evidence={'citations':[{'start':0,'end':10}]})
        t = m.EvidenceMemory.from_dict(s.to_dict())
        review = t.facts[x]['reviews'][0]
        self.assertEqual([{'start':0,'end':17}], review['evidence']['evidence_spans'])
        self.assertEqual('PASS', review['evidence']['component_evidence_check'])
        change = t.changes[cid]
        self.assertEqual([{'start':0,'end':10}], change['evidence']['citations'])

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
