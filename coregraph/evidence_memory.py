"""V2 evidence memory: independent fact occurrences and proposed change links.

This store does not judge textual support or commit updates. The entity graph is
an index; this store retains raw predicates and evidence without graph merging.
"""
import copy
import hashlib
import json


def _id(prefix, values):
    return prefix + hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def namespace_chunks(chunks):
    """Assign stable IDs per session and ordinal; collision checks reject changed inputs.

    The pre-namespace id is preserved as ``source_chunk_id`` so original dataset
    ids stay traceable. Re-running on already-namespaced input is idempotent
    because the generated id depends only on (session_id, ordinal).
    """
    counts = {}
    result = []
    for chunk in chunks:
        sid = str(chunk.get('session_id', '')).strip()
        if not sid or sid.lower() == 'unknown':
            raise ValueError('evidence memory needs a stable session_id')
        index = counts.get(sid, 0)
        counts[sid] = index + 1
        item = copy.deepcopy(chunk)
        if 'source_chunk_id' not in item:
            item['source_chunk_id'] = chunk.get('chunk_id')
        item['chunk_id'] = _id('session_chunk_', [sid, index])
        result.append(item)
    return result


def chunk_conflicts(a, b):
    """True only when chunk payloads genuinely differ.

    ``source_chunk_id`` is run-scoped provenance (full-batch processing numbers
    across the corpus, per-session processing restarts at 0), so it is excluded
    from identity comparison: the same session replayed under a different
    processing mode must not be rejected as a collision.
    """
    keys = set(a) | set(b)
    keys.discard('source_chunk_id')
    return any(a.get(k) != b.get(k) for k in keys)


class EvidenceMemory:
    def __init__(self):
        self.chunks = {}
        self.facts = {}
        self.changes = {}
        self.complete = True
        self._entities = {}
        self._links = {}
        self._change_keys = {}   # change_key -> [change_id, ...] in event order
        self._reviews = {}       # hash(change_key, review_id) -> change_id (retry dedup)

    def add_chunk(self, chunk):
        sid = str(chunk.get('session_id', '')).strip()
        if not sid or sid.lower() == 'unknown':
            raise ValueError('evidence memory needs a stable session_id')
        if chunk.get('chunk_id') is None or not chunk.get('text'):
            raise ValueError('chunk_id and text are required')
        cid = _id('chunk_', [sid, str(chunk['chunk_id'])])
        data = copy.deepcopy(chunk)
        if cid in self.chunks and chunk_conflicts(self.chunks[cid], data):
            raise ValueError('chunk identity collision; source preserved')
        self.chunks[cid] = data  # equal payload: refresh run-scoped provenance
        return cid

    def add_fact(self, raw, evidence_id):
        if evidence_id not in self.chunks:
            raise ValueError('missing evidence')
        src, rel, tgt = [str(raw.get(k, '')).strip() for k in ('src', 'relation', 'tgt')]
        if not all((src, rel, tgt)):
            raise ValueError('missing subject, predicate or object')
        fid = _id('fact_', [src, rel, tgt, evidence_id])
        self.facts[fid] = {'id': fid, 'src': src, 'relation': rel, 'tgt': tgt,
                           'evidence_id': evidence_id, 'raw': copy.deepcopy(raw),
                           'support_status': 'unverified'}
        for entity in (src, tgt, raw.get('canonical_src', ''), raw.get('canonical_tgt', '')):
            if entity: self._entities.setdefault(str(entity).casefold(), set()).add(fid)
        return fid

    def find_by_entities(self, entities):
        ids = set()
        for entity in entities: ids.update(self._entities.get(str(entity).casefold(), ()))
        return [copy.deepcopy(self.facts[x]) for x in sorted(ids)]

    def add_change(self, old_id, new_id, relation, evidence_id, reason,
                   status='candidate', decision='pending', review_id=None):
        """Record one review event between two facts.

        Each call appends an event: ``revision`` increases per change link (so
        a reverted decision like keep -> replace -> keep keeps all three) and
        ``seq`` orders events globally (so the latest decision is identifiable).
        ``review_id`` is the retry idempotency key: repeating the same
        review_id returns the existing record, while reusing a review_id with a
        different decision is rejected. Without review_id every call is a new
        event (history-first; review pipelines should pass review_id to stay
        replay-safe). Only status='candidate' is allowed: unreviewed links
        never mutate fact state.
        """
        if old_id not in self.facts or new_id not in self.facts or old_id == new_id:
            raise ValueError('two distinct facts required')
        if evidence_id not in self.chunks or not relation or not reason or status != 'candidate':
            raise ValueError('only sourced candidate decisions are supported')
        decision = str(decision or '').strip() or 'pending'
        core = [old_id, new_id, relation, evidence_id, reason]
        change_key = _id('change_key_', core)
        if review_id is not None:
            review_id = str(review_id)
            review_key = _id('review_', core + [review_id])
            prior = self._reviews.get(review_key)
            if prior is not None:
                if self.changes[prior]['decision'] != decision:
                    raise ValueError('review_id reused with a different decision')
                return prior
        revision = len(self._change_keys.get(change_key, ()))
        cid = _id('change_', core + [revision])
        self.changes[cid] = {'id':cid,'old_fact_id':old_id,'new_fact_id':new_id,'relation':relation,
                             'evidence_id':evidence_id,'reason':reason,'status':status,
                             'decision':decision,'change_key':change_key,'revision':revision,
                             'seq':len(self.changes),'review_id':review_id}
        self._change_keys.setdefault(change_key, []).append(cid)
        if review_id is not None:
            self._reviews[review_key] = cid
        for fid in (old_id,new_id): self._links.setdefault(fid,set()).add(cid)
        return cid

    def trace(self, fact_id):
        """Linked change events in review order (global seq)."""
        return [copy.deepcopy(self.changes[x]) for x in
                sorted(self._links.get(fact_id, ()), key=lambda x: self.changes[x]['seq'])]

    def record_review(self, fact_id, route, reasons, review_id=None):
        """Attach one advisory single-fact review outcome (never a state change).

        Unlike add_change there are no two endpoints: a semantic review judges
        ONE candidate against its source. The fact identity never changes;
        only support_status and an append-only review history are updated.
        ``review_id`` is the retry idempotency key: repeating the same id with
        the same outcome returns the existing record, while reusing it with a
        different route/reasons is rejected. TECHNICAL_ERROR leaves
        support_status untouched (a crash is not a review outcome).
        """
        if fact_id not in self.facts:
            raise ValueError('unknown fact')
        route = str(route or '').strip()
        if route not in ('DEFER', 'REVIEW_NEXT', 'TECHNICAL_ERROR'):
            raise ValueError('unsupported review route')
        if not isinstance(reasons, (list, tuple)) or not reasons or \
                any(not isinstance(r, str) or not r.strip() for r in reasons):
            raise ValueError('nonempty string reasons required')
        reasons = list(dict.fromkeys(reasons))
        reviews = self.facts[fact_id].setdefault('reviews', [])
        if review_id is not None:
            review_id = str(review_id)
            prior = next((r for r in reviews if r.get('review_id') == review_id), None)
            if prior is not None:
                if prior.get('route') != route or prior.get('reasons') != reasons:
                    raise ValueError('review_id reused with a different outcome')
                return prior
        record = {'route': route, 'reasons': reasons,
                  'review_id': review_id, 'seq': len(reviews)}
        reviews.append(record)
        if route == 'DEFER':
            self.facts[fact_id]['support_status'] = 'review_defer'
        elif route == 'REVIEW_NEXT':
            self.facts[fact_id]['support_status'] = 'review_next'
        return copy.deepcopy(record)

    def evidence_bundle(self, fact_id):
        """Trace both endpoints and source chunks; links remain unverified candidates."""
        if fact_id not in self.facts:
            raise ValueError('unknown fact')
        changes = self.trace(fact_id)
        fact_ids = {fact_id}
        source_ids = set()
        for change in changes:
            fact_ids.update((change['old_fact_id'], change['new_fact_id']))
            source_ids.add(change['evidence_id'])
        facts = [copy.deepcopy(self.facts[fid]) for fid in sorted(fact_ids)]
        source_ids.update(f['evidence_id'] for f in facts)
        return {'facts': facts, 'changes': changes,
                'chunks': {cid: copy.deepcopy(self.chunks[cid]) for cid in sorted(source_ids)}}

    def to_dict(self):
        # version 4: facts may carry advisory review history (support_status/reviews
        # restored on load; identity unchanged). version 3: change records are review
        # events (change_key/revision/seq/review_id); version 2 ids included the
        # decision, version 1 excluded it. Older versions are validated with their
        # original formula and re-keyed as events on load.
        return copy.deepcopy({'version':4,'chunks':self.chunks,'facts':self.facts,
                              'changes':self.changes,'complete':self.complete})

    @classmethod
    def from_dict(cls, data):
        version = data.get('version')
        if version not in (1, 2, 3, 4): raise ValueError('unsupported evidence memory version')
        s=cls(); s.complete = data.get('complete',False)
        for cid, chunk in data['chunks'].items():
            if s.add_chunk(chunk) != cid: raise ValueError('invalid chunk identity')
        for fid, fact in data['facts'].items():
            if s.add_fact(fact['raw'],fact['evidence_id']) != fid: raise ValueError('invalid fact identity')
            # advisory review metadata is not part of fact identity; restore it
            record=s.facts[fid]
            record['support_status']=fact.get('support_status','unverified')
            reviews=fact.get('reviews',[])
            if not isinstance(reviews,list) or any(not isinstance(r,dict) or
                    not isinstance(r.get('route'),str) or not isinstance(r.get('reasons'),list)
                    for r in reviews):
                raise ValueError('invalid review history')
            if [r.get('seq') for r in reviews]!=list(range(len(reviews))):
                raise ValueError('invalid review order metadata')
            if reviews: record['reviews']=copy.deepcopy(reviews)
        for cid, change in data['changes'].items():
            decision = str(change.get('decision') or '').strip() or 'pending'
            core = [change['old_fact_id'],change['new_fact_id'],change['relation'],
                    change['evidence_id'],change['reason']]
            args = (*core, change['status'], decision)
            if version == 1:
                if _id('change_', core) != cid: raise ValueError('invalid legacy change identity')
                s.add_change(*args)  # re-key as a review event
            elif version == 2:
                if _id('change_', core + [decision]) != cid: raise ValueError('invalid legacy change identity')
                s.add_change(*args)  # re-key as a review event
            else:
                new_id = s.add_change(*args, review_id=change.get('review_id'))
                if new_id != cid: raise ValueError('invalid change identity')
                record = s.changes[new_id]
                if (record['revision'] != change.get('revision') or
                        record['seq'] != change.get('seq') or
                        record['change_key'] != change.get('change_key')):
                    raise ValueError('invalid change order metadata')
        return s
