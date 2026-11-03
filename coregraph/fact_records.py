"""Experimental source-linked fact and decision records; no automatic state updates."""
import copy
import hashlib
import json


def _identifier(prefix, values):
    raw = json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return prefix + hashlib.sha256(raw).hexdigest()


class FactRecords:
    def __init__(self):
        self.chunks = {}
        self.facts = {}
        self.changes = {}
        self._subject_index = {}
        self._change_index = {}

    def add_chunk(self, chunk_id, text):
        if not chunk_id or not isinstance(text, str) or not text:
            raise ValueError('chunk id and text are required')
        if chunk_id in self.chunks and self.chunks[chunk_id] != text:
            raise ValueError('chunk id already belongs to different text')
        self.chunks[chunk_id] = text

    def add_fact(self, subject, relation, value, evidence_id):
        if not all(isinstance(x, str) and x.strip() for x in (subject, relation, value)):
            raise ValueError('subject, relation and value are required')
        if evidence_id not in self.chunks:
            raise ValueError('evidence chunk not found')
        fid = _identifier('fact_', [subject, relation, value, evidence_id])
        self.facts[fid] = {'id': fid, 'subject': subject, 'relation': relation,
                           'value': value, 'evidence_id': evidence_id}
        self._subject_index.setdefault(subject, set()).add(fid)
        self._subject_index.setdefault(value, set()).add(fid)
        return fid

    def find_facts(self, subject, relation=None):
        return [copy.deepcopy(self.facts[fid]) for fid in sorted(self._subject_index.get(subject, set()))
                if relation is None or self.facts[fid]['relation'] == relation]

    def add_change(self, old_id, new_id, relation, evidence_id, reason):
        if old_id not in self.facts or new_id not in self.facts or old_id == new_id:
            raise ValueError('two distinct existing facts are required')
        if evidence_id not in self.chunks or not relation or not reason:
            raise ValueError('change needs a source, relation and reason')
        cid = _identifier('change_', [old_id, new_id, relation, evidence_id, reason])
        self.changes[cid] = {'id': cid, 'old_fact_id': old_id, 'new_fact_id': new_id,
                             'relation': relation, 'evidence_id': evidence_id, 'reason': reason,
                             'decision_status': 'recorded_unverified'}
        for fid in (old_id, new_id): self._change_index.setdefault(fid, set()).add(cid)
        return cid

    def changes_for(self, fact_id):
        return [copy.deepcopy(self.changes[cid]) for cid in sorted(self._change_index.get(fact_id, set()))]

    def to_dict(self):
        return copy.deepcopy({'chunks': self.chunks, 'facts': self.facts, 'changes': self.changes})

    @classmethod
    def from_dict(cls, data):
        store = cls()
        for key, text in data['chunks'].items(): store.add_chunk(key, text)
        for f in data['facts'].values():
            if store.add_fact(f['subject'], f['relation'], f['value'], f['evidence_id']) != f['id']:
                raise ValueError('invalid fact id')
        for c in data['changes'].values():
            if store.add_change(c['old_fact_id'], c['new_fact_id'], c['relation'], c['evidence_id'], c['reason']) != c['id']:
                raise ValueError('invalid change id')
        return store
