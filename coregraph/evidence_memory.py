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


class EvidenceMemory:
    def __init__(self):
        self.chunks = {}
        self.facts = {}
        self.changes = {}
        self.complete = True
        self._entities = {}
        self._links = {}

    def add_chunk(self, chunk):
        sid = str(chunk.get('session_id', '')).strip()
        if not sid or sid.lower() == 'unknown':
            raise ValueError('evidence memory needs a stable session_id')
        if chunk.get('chunk_id') is None or not chunk.get('text'):
            raise ValueError('chunk_id and text are required')
        cid = _id('chunk_', [sid, str(chunk['chunk_id'])])
        data = copy.deepcopy(chunk)
        if cid in self.chunks and self.chunks[cid] != data:
            raise ValueError('chunk identity collision; source preserved')
        self.chunks[cid] = data
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

    def add_change(self, old_id, new_id, relation, evidence_id, reason, status='candidate', decision='pending'):
        """Record a sourced candidate decision between two facts.

        ``decision`` is the proposed handling action (default 'pending'); it is
        an attribute only and is not part of the record id, so records written
        before this field existed still load. Only status='candidate' is allowed:
        unreviewed links never mutate fact state.
        """
        if old_id not in self.facts or new_id not in self.facts or old_id == new_id:
            raise ValueError('two distinct facts required')
        if evidence_id not in self.chunks or not relation or not reason or status != 'candidate':
            raise ValueError('only sourced candidate decisions are supported')
        decision = str(decision or '').strip() or 'pending'
        cid = _id('change_', [old_id,new_id,relation,evidence_id,reason])
        self.changes[cid] = {'id':cid,'old_fact_id':old_id,'new_fact_id':new_id,'relation':relation,
                             'evidence_id':evidence_id,'reason':reason,'status':status,'decision':decision}
        for fid in (old_id,new_id): self._links.setdefault(fid,set()).add(cid)
        return cid

    def trace(self, fact_id):
        return [copy.deepcopy(self.changes[x]) for x in sorted(self._links.get(fact_id, ()))]

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
        return copy.deepcopy({'version':1,'chunks':self.chunks,'facts':self.facts,
                              'changes':self.changes,'complete':self.complete})

    @classmethod
    def from_dict(cls, data):
        if data.get('version') != 1: raise ValueError('unsupported evidence memory version')
        s=cls(); s.complete = data.get('complete',False)
        for cid, chunk in data['chunks'].items():
            if s.add_chunk(chunk) != cid: raise ValueError('invalid chunk identity')
        for fid, fact in data['facts'].items():
            if s.add_fact(fact['raw'],fact['evidence_id']) != fid: raise ValueError('invalid fact identity')
        for cid, change in data['changes'].items():
            if s.add_change(change['old_fact_id'],change['new_fact_id'],change['relation'],change['evidence_id'],change['reason'],change['status'],change.get('decision','pending')) != cid: raise ValueError('invalid change identity')
        return s
