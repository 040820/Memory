#!/usr/bin/env python3
"""Run isolated user-turn candidate extraction, with per-turn checkpoints."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'implementation'))
from evidence_memory.fact_extraction import create_payload, parse_candidates
from evidence_memory.store import EvidenceStore
from evidence_memory.api_config import request_settings


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    with tempfile.NamedTemporaryFile('w', encoding='utf8', dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf8'))
    if cfg['enable_thinking'] is not False or cfg['temperature'] != 0 or cfg['input_roles'] != ['user']:
        raise ValueError('unexpected experiment configuration')
    sources = EvidenceStore(args.sources).sources()
    selected = sorted((eid, s) for eid, s in sources.items() if s['speaker'] == 'user')
    provenance = {'sources_sha256': sha(args.sources), 'config_sha256': sha(args.config),
                  'extractor_sha256': sha(ROOT / 'implementation/evidence_memory/fact_extraction.py'),
                  'runner_sha256': sha(Path(__file__)), 'selected_turns': len(selected),
                  'config': cfg, 'applied': False}
    if args.resume:
        old = json.loads((args.output / 'manifest.json').read_text(encoding='utf8'))
        if old != provenance:
            raise ValueError('resume inputs or code do not match manifest')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'turns').mkdir()
        save_json(args.output / 'manifest.json', provenance)
    total = len(selected)
    for number, (eid, source) in enumerate(selected, 1):
        path = args.output / 'turns' / (eid + '.json')
        if path.exists():
            print(f'[{number}/{total}] cached {eid}', flush=True)
            continue
        started = time.monotonic()
        payload = create_payload(source, model=cfg['model'], max_tokens=cfg['max_tokens'])
        headers, payload = request_settings(cfg, payload)
        record = {'evidence_id': eid, 'source': source, 'request': payload,
                  'created_at': datetime.now(timezone.utc).isoformat(), 'applied': False}
        try:
            response = requests.post(cfg['base_url'].rstrip('/') + '/chat/completions',
                                     headers=headers, json=payload, timeout=None)
            record['http_status'] = response.status_code
            record['raw_http_body'] = response.text
            response.raise_for_status()
            data = response.json()
            record['response'] = data
            choice = data['choices'][0]
            raw = choice['message'].get('content')
            if choice.get('finish_reason') == 'length':
                result = {'status': 'truncated', 'candidates': [], 'skipped': []}
            else:
                result = parse_candidates(raw, source)
            for candidate in result['candidates']:
                quote = candidate['quote']
                spans = []
                start = 0
                while (pos := source['text'].find(quote, start)) >= 0:
                    spans.append([pos, pos + len(quote)]); start = pos + 1
                candidate['evidence_id'] = eid
                candidate['quote_spans'] = spans
            record['result'] = result
        except Exception as exc:
            record['result'] = {'status': 'technical_error', 'candidates': [], 'skipped': [],
                                'error': f'{type(exc).__name__}: {exc}'}
        record['elapsed_seconds'] = round(time.monotonic() - started, 2)
        save_json(path, record)
        usage = record.get('response', {}).get('usage') or {}
        result = record['result']
        print(f"[{number}/{total}] {eid} status={result['status']} candidates={len(result['candidates'])} skipped={len(result['skipped'])} tokens={usage.get('total_tokens', 'not_returned')} elapsed={record['elapsed_seconds']}s", flush=True)
    print('COMPLETED: per-turn records saved; no memory updates applied', flush=True)


if __name__ == '__main__':
    main()
