#!/usr/bin/env python3
"""Create a reproducible, blind, human-annotation pilot from three local datasets."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERN = re.compile(r'\b(moved|moving|relocated|married|divorced|graduated|new job|started working|no longer|engaged|left my job|quit my job)\b', re.I)


def read_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def memory_turns(session):
    turns = []
    for turn_id, messages in (session.get('Session_Dialogue') or {}).items():
        for index, message in enumerate(messages):
            turns.append({'turn_id': turn_id, 'message_index': index, 'speaker': message.get('role'), 'text': message.get('content', '')})
    return turns


def annotation():
    return {'annotator': '', 'status': 'unannotated', 'old_evidence': [], 'new_evidence': [],
            'subject': None, 'attribute': None, 'old_value': None, 'new_value': None,
            'old_modality': None, 'new_modality': None, 'event_time': None,
            'relation': None, 'allowed_actions': [], 'reason': '', 'uncertainty': '', 'notes': ''}


def sample(candidates, count, rng):
    return rng.sample(candidates, min(count, len(candidates)))


def mem_candidates(path):
    groups = {x: [] for x in ('residence', 'career', 'relationship', 'static_conflict')}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            person = json.loads(line)
            history = []
            for session in person['Full_Session_Chain']:
                visible = {'session_id': session.get('Session_ID'), 'date': session.get('Date'), 'messages': memory_turns(session)}
                history.append(visible)
                attrs = session.get('Updated_Attributes') or []
                names = {a.get('Attribute') for a in attrs}
                tags = []
                if 'Residence' in names: tags.append('residence')
                if 'Career_Status' in names: tags.append('career')
                if 'Marital_Status' in names: tags.append('relationship')
                if session.get('Static_Conflict_Information'): tags.append('static_conflict')
                for tag in tags:
                    groups[tag].append({'dataset': 'memconflict', 'source_id': str(person['ID']),
                        'target_session_id': session.get('Session_ID'), 'context': list(history),
                        'reference': {'selection_stratum': tag, 'updated_attributes': attrs,
                                      'static_conflicts': session.get('Static_Conflict_Information', []),
                                      'questions': session.get('Session_Questions', [])}})
    return groups


def long_candidates(path):
    candidates = []
    for row in read_json(path):
        if row.get('question_type') != 'knowledge-update' or str(row.get('question_id', '')).endswith('_abs'):
            continue
        ids, dates, sessions = (row.get('haystack_session_ids', []), row.get('haystack_dates', []), row.get('haystack_sessions', []))
        if not len(ids) == len(dates) == len(sessions):
            raise ValueError('LongMemEval session arrays have unequal lengths: ' + str(row.get('question_id')))
        context = []
        for sid, date, turns in zip(ids, dates, sessions):
            context.append({'session_id': sid, 'date': date, 'messages': [
                {'turn_id': str(i), 'speaker': turn.get('role'), 'text': turn.get('content', '')}
                for i, turn in enumerate(turns)]})
        candidates.append({'dataset': 'longmemeval', 'source_id': row['question_id'], 'target_session_id': None,
            'context': context, 'reference': {k: row.get(k) for k in ('question', 'answer', 'question_type', 'question_date', 'answer_session_ids')}})
    return candidates


def locomo_candidates(path):
    result = []
    for row in read_json(path):
        conv = row['conversation']
        keys = sorted((k for k in conv if re.fullmatch(r'session_\d+', k)), key=lambda k: int(k.split('_')[1]))
        history = []
        for key in keys:
            turns = conv[key]
            session = {'session_id': 'D' + key.split('_')[1], 'date': conv.get(key + '_date_time'), 'messages': []}
            for turn in turns:
                message = {'turn_id': turn.get('dia_id'), 'speaker': turn.get('speaker'), 'text': turn.get('text', '')}
                session['messages'].append(message)
                if PATTERN.search(message['text']):
                    snapshot = history + [{**session, 'messages': list(session['messages'])}]
                    result.append({'dataset': 'locomo', 'source_id': row['sample_id'], 'target_session_id': session['session_id'],
                        'target_turn_id': message['turn_id'], 'context': snapshot,
                        'reference': {'selection_method': 'keyword_candidate_not_gold', 'matched_term': PATTERN.search(message['text']).group(0)}})
            history.append(session)
    return result


def make_id(item):
    key = [item['dataset'], item['source_id'], item.get('target_session_id'), item.get('target_turn_id')]
    return item['dataset'] + '-' + hashlib.sha256(json.dumps(key, ensure_ascii=False).encode('utf8')).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memconflict', type=Path, default=ROOT/'dataset/data/MemConflict/Step4_4.jsonl')
    p.add_argument('--longmemeval', type=Path, default=ROOT/'dataset/data/longmemeval_s_cleaned.json')
    p.add_argument('--locomo', type=Path, default=ROOT/'dataset/data/locomo10.json')
    p.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite existing annotations')
    p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--mem-per-stratum', type=int, default=5)
    p.add_argument('--long-count', type=int, default=8)
    p.add_argument('--locomo-count', type=int, default=8)
    args = p.parse_args()
    if any(v < 0 for v in (args.mem_per_stratum, args.long_count, args.locomo_count)):
        p.error('sample sizes must be nonnegative')
    if args.output.exists():
        p.error('output already exists; choose a new directory')
    rng = random.Random(args.seed)
    groups = mem_candidates(args.memconflict)
    chosen, seen, strata = [], set(), {}
    for name, pool in groups.items():
        eligible = [x for x in pool if make_id(x) not in seen]
        selected = sample(eligible, args.mem_per_stratum, rng)
        strata[name] = {'candidates': len(pool), 'eligible_after_dedup': len(eligible), 'selected': len(selected)}
        for item in selected:
            seen.add(make_id(item)); chosen.append(item)
    long_pool, locomo_pool = long_candidates(args.longmemeval), locomo_candidates(args.locomo)
    chosen += sample(long_pool, args.long_count, rng) + sample(locomo_pool, args.locomo_count, rng)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'annotation.jsonl').open('w', encoding='utf-8', newline='\n') as blind, (args.output/'references.jsonl').open('w', encoding='utf-8', newline='\n') as ref:
        for item in chosen:
            item = dict(item)
            reference = item.pop('reference')
            case_id = make_id(item)
            item['case_id'] = case_id
            item['annotation'] = annotation()
            blind.write(json.dumps(item, ensure_ascii=False) + '\n')
            ref.write(json.dumps({'case_id': case_id, 'reference': reference}, ensure_ascii=False) + '\n')
    manifest = {'seed': args.seed, 'counts': dict(Counter(x['dataset'] for x in chosen)), 'mem_strata': strata,
                'long_pool': len(long_pool), 'locomo_keyword_pool': len(locomo_pool),
                'selection_bias': 'MemConflict metadata, LongMemEval question type, LoCoMo keywords; diagnostic sample only',
                'context_policy': 'source order; MemConflict through target session, LoCoMo through target turn, LongMemEval full haystack; no timestamp filtering',
                'inputs': {name: {'path': str(path.resolve()), 'sha256': file_hash(path)} for name, path in [('memconflict', args.memconflict), ('longmemeval', args.longmemeval), ('locomo', args.locomo)]}}
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'counts': manifest['counts'], 'mem_strata': strata}, ensure_ascii=False))


if __name__ == '__main__':
    main()
