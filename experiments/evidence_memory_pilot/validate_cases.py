"""Validate controlled input and separate assessment data; never call a model."""
import argparse
import json
from pathlib import Path

FORBIDDEN = {'answer', 'expected', 'gold', 'relation', 'allowed_actions', 'evidence', 'label'}


def validate(inputs, gold):
    errors = []
    def indexed(rows, name):
        result = {}
        for row in rows:
            key = row.get('case_id')
            if not key or key in result:
                errors.append(f'{name}: missing or duplicate case_id {key}')
            result[key] = row
        return result
    ii, gg = indexed(inputs, 'inputs'), indexed(gold, 'gold')
    if set(ii) != set(gg):
        errors.append('input/gold case IDs do not match')
    for case_id, item in ii.items():
        def check_forbidden(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in FORBIDDEN:
                        errors.append(f'{case_id}: forbidden input field {k}')
                    check_forbidden(v)
            elif isinstance(obj, list):
                for v in obj:
                    check_forbidden(v)
        check_forbidden(item)
        messages = item.get('messages', [])
        msgmap = {m.get('id'): m.get('text', '') for m in messages}
        if len(msgmap) != len(messages) or None in msgmap:
            errors.append(f'{case_id}: missing or duplicate message ID')
        queries = item.get('queries', [])
        query_ids = [q.get('id') for q in queries]
        if len(set(query_ids)) != len(query_ids) or None in query_ids:
            errors.append(f'{case_id}: missing or duplicate query ID')
        for evidence in gg.get(case_id, {}).get('evidence', []):
            quote = evidence.get('quote', '')
            if not quote or quote not in msgmap.get(evidence.get('message_id'), ''):
                errors.append(f'{case_id}: quote does not match message')
        answer_ids = [a.get('query_id') for a in gg.get(case_id, {}).get('answers', [])]
        if len(set(answer_ids)) != len(answer_ids) or set(answer_ids) != set(query_ids):
            errors.append(f'{case_id}: gold query IDs do not match')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('controlled_inputs.json'))
    parser.add_argument('--gold', type=Path, default=Path(__file__).with_name('controlled_gold.json'))
    args = parser.parse_args()
    with args.inputs.open(encoding='utf8') as f:
        inputs = json.load(f)
    with args.gold.open(encoding='utf8') as f:
        gold = json.load(f)
    errors = validate(inputs, gold)
    for error in errors:
        print(error)
    print(f'cases={len(inputs)} gold={len(gold)} errors={len(errors)}')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
