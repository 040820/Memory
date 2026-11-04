"""Conservative shadow ingestion; it never authorizes a state update."""
from .store import StateStore, stable_id


class ShadowPipeline:
    def __init__(self, path):
        self.store = StateStore(path)

    def process_source(self, source, candidates=()):
        evidence_id = self.store.put_source(source)
        outcomes = []
        for candidate in candidates:
            try:
                item = dict(candidate)
                item['evidence_id'] = evidence_id
                fact_id = self.store.put_candidate(item)
                outcomes.append({'fact_id': fact_id, 'status': 'unverified',
                                 'action': 'DEFER', 'reason': 'semantic review not available'})
            except (TypeError, ValueError) as exc:
                outcomes.append({'status': 'skipped', 'action': 'DEFER', 'reason': str(exc)})
        trace = {'evidence_id': evidence_id, 'mode': 'shadow', 'applied': False,
                 'candidates': outcomes}
        trace_id = stable_id('trace', trace)
        with self.store._locked():
            snapshot = self.store._load()
            traces = snapshot.setdefault('shadow_traces', {})
            if trace_id not in traces:
                traces[trace_id] = trace
                self.store._write(snapshot)
        return trace


def ensure_graph_mode(mode):
    if mode == 'off':
        return
    if mode in ('shadow', 'active'):
        raise NotImplementedError(f'evolution.mode={mode} is not connected to GraphRAG; use standalone ShadowPipeline only')
    raise ValueError(f'unknown evolution mode: {mode}')
