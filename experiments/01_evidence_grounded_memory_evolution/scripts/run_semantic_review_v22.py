#!/usr/bin/env python3
"""Advisory review of frozen candidates; labels never enter this program."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
import requests
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'implementation'))
from evidence_memory.semantic_review_v22 import review_payload,parse_and_route
from evidence_memory.api_config import request_settings


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,obj):
    with tempfile.NamedTemporaryFile('w',dir=path.parent,encoding='utf8',delete=False) as f:
        tmp=Path(f.name)
        try:
            json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True);raise
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    cases=json.loads(args.input.read_text(encoding='utf8'))
    cfg=yaml.safe_load(args.config.read_text(encoding='utf8'))
    if cfg['enable_thinking'] is not False or cfg['temperature']!=0:raise ValueError('unexpected configuration')
    if len({x['id'] for x in cases})!=len(cases):raise ValueError('duplicate case id')
    manifest={'input_sha256':sha(args.input),'config_sha256':sha(args.config),
              'reviewer_sha256':sha(ROOT/'implementation/evidence_memory/semantic_review_v22.py'),
              'base_v2_sha256':sha(ROOT/'implementation/evidence_memory/semantic_review_v2.py'),'base_v21_sha256':sha(ROOT/'implementation/evidence_memory/semantic_review_v21.py'),'runner_sha256':sha(Path(__file__)),'api_config_sha256':sha(ROOT/'implementation/evidence_memory/api_config.py'),'case_count':len(cases),'config':cfg,
              'review_is_advisory':True,'extractor_model':cfg.get('extractor_model','unknown'),'reviewer_model':cfg['model'],'applied':False}
    if args.resume:
        if json.loads((args.output/'manifest.json').read_text(encoding='utf8'))!=manifest:raise ValueError('resume manifest mismatch')
    else:
        args.output.mkdir(parents=True,exist_ok=False);(args.output/'cases').mkdir();write(args.output/'manifest.json',manifest)
    for idx,case in enumerate(cases,1):
        key=hashlib.sha256(case['id'].encode()).hexdigest();path=args.output/'cases'/f'{key}.json'
        if path.exists():print(f'[{idx}/{len(cases)}] cached {case["id"]}',flush=True);continue
        source=dict(case['source']);source['evidence_id']=case.get('evidence_id',case['id']);candidate=case['candidate']
        result={'case_id':case['id'],'source':source,'candidate':candidate,'applied':False}
        start=time.monotonic()
        try:
            payload=review_payload(source,candidate,model=cfg['model'],max_tokens=cfg['max_tokens'])
            headers,payload=request_settings(cfg,payload)
            result['request']=payload
            resp=requests.post(cfg['base_url'].rstrip('/')+'/chat/completions',headers=headers,json=payload,timeout=None)
            result['http_status']=resp.status_code;result['raw_http_body']=resp.text;resp.raise_for_status()
            data=resp.json();result['response']=data;choice=data['choices'][0]
            if choice.get('finish_reason')=='length':review={'route':'TECHNICAL_ERROR','reasons':['TRUNCATED'],'applied':False}
            else:review=parse_and_route(choice['message'].get('content'),source,candidate)
            result['result']=review
        except Exception as exc:
            result['result']={'route':'TECHNICAL_ERROR','reasons':['REQUEST_ERROR'],'applied':False,'error':f'{type(exc).__name__}: {exc}'}
        result['elapsed_seconds']=round(time.monotonic()-start,2);write(path,result)
        review=result['result'];usage=result.get('response',{}).get('usage') or {}
        print(f'[{idx}/{len(cases)}] {case["id"]} route={review["route"]} reasons={review.get("reasons",[])} tokens={usage.get("total_tokens","not_returned")} elapsed={result["elapsed_seconds"]}s',flush=True)
    print('COMPLETED: advisory review saved; no memory updates applied',flush=True)

if __name__=='__main__':main()
