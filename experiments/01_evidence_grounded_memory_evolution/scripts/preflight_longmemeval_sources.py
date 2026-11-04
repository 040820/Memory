#!/usr/bin/env python3
"""Freeze and check a small, dialogue-only LongMemEval source sample."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'implementation'))
from evidence_memory.longmemeval import adapt_sessions
from evidence_memory.store import EvidenceStore


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--question-id', required=True)
    parser.add_argument('--session-count', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.session_count < 1:
        parser.error('session-count must be positive')
    with args.input.open(encoding='utf8') as stream:
        records = json.load(stream)
    matching = [r for r in records if r.get('question_id') == args.question_id]
    if len(matching) != 1:
        raise ValueError('question id must identify exactly one record')
    row = matching[0]
    if not (len(row['haystack_session_ids']) == len(row['haystack_dates']) == len(row['haystack_sessions'])):
        raise ValueError('full session arrays differ in length')
    count = args.session_count
    if count > len(row['haystack_sessions']):
        raise ValueError('requested sample exceeds available sessions')
    # Only these dialogue fields cross the adapter boundary. No labels or answers.
    sources = adapt_sessions(row['question_id'], row['haystack_session_ids'][:count],
                             row['haystack_dates'][:count], row['haystack_sessions'][:count])
    args.output.mkdir(parents=True, exist_ok=False)
    store_path = args.output / 'evidence.json'
    store = EvidenceStore(store_path)
    ids = [store.put_source(source) for source in sources]
    replay = [store.put_source(source) for source in sources]
    saved = store.sources()
    if ids != replay or len(set(ids)) != len(ids) or len(saved) != len(sources):
        raise AssertionError('identity, replay or uniqueness check failed')
    if any(saved[sid] != source for sid, source in zip(ids, sources)):
        raise AssertionError('source round trip failed')
    checked_quotes = 0
    for sid, source in zip(ids, sources):
        quote = source['text'][:min(30, len(source['text']))]
        spans = store.locate_quote(sid, quote)
        if not spans or any(source['text'][start:end] != quote for start, end in spans):
            raise AssertionError('quote locator failed')
        checked_quotes += 1
    report = {'purpose': 'source adaptation preflight; not a model or accuracy evaluation',
              'input_path': str(args.input.resolve()), 'input_sha256': sha256(args.input),
              'question_id_for_conversation_identity_only': args.question_id,
              'selection': 'first sessions in original order', 'session_count': count,
              'session_ids': row['haystack_session_ids'][:count],
              'source_count': len(sources), 'unique_evidence_ids': len(set(ids)),
              'roles': dict(Counter(s['speaker'] for s in sources)),
              'time_precision': dict(Counter(s['time_precision'] for s in sources)),
              'timezone': 'unknown', 'roundtrip_and_replay': 'passed', 'quote_locator_checked': checked_quotes,
              'script_sha256': sha256(Path(__file__)),
              'output_sha256': sha256(store_path),
              'source_adapter_sha256': sha256(ROOT / 'implementation/evidence_memory/longmemeval.py'),
              'store_sha256': sha256(ROOT / 'implementation/evidence_memory/store.py')}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
