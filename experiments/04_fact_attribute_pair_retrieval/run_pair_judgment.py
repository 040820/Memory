#!/usr/bin/env python3
"""Review all retrieved fact pairs as non-mutating candidate proposals."""
import argparse,hashlib,json,os,sys,tempfile,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import requests,yaml
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from relation_judge import make_payload,parse_relation

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,data):
 with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=path.parent,delete=False) as f:
  tmp=Path(f.name)
  try:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  except BaseException:tmp.unlink(missing_ok=True);raise
 os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--retrieval',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 sources=json.loads(a.input.read_text())['sources'];source_map={s['evidence_id']:s for s in sources}
 facts=json.loads((a.retrieval/'facts_snapshot.json').read_text())['facts'];fact_map={f['fact_id']:f for f in facts};pairs=json.loads((a.retrieval/'pair_candidates.json').read_text())['pairs'];cfg=yaml.safe_load(a.config.read_text())
 manifest={'input_sha256':sha(a.input),'facts_sha256':sha(a.retrieval/'facts_snapshot.json'),'pairs_sha256':sha(a.retrieval/'pair_candidates.json'),'config_sha256':sha(a.config),'module_sha256':sha(ROOT/'src/relation_judge.py'),'runner_sha256':sha(Path(__file__)),'pair_count':len(pairs),'model':cfg['model'],'question_or_gold_used':False,'applied':False}
 if a.resume:
  if json.loads((a.output/'manifest.json').read_text())!=manifest:raise ValueError('resume mismatch')
 else:a.output.mkdir(parents=True,exist_ok=False);(a.output/'pairs').mkdir();write(a.output/'manifest.json',manifest)
 key=Path(cfg['api_key_file']).read_text().strip()
 def work(pair):
  ident=hashlib.sha256((pair['old_fact_id']+'|'+pair['new_fact_id']).encode()).hexdigest();path=a.output/'pairs'/(ident+'.json')
  if path.exists():return json.loads(path.read_text())
  old,new=fact_map[pair['old_fact_id']],fact_map[pair['new_fact_id']];start=time.monotonic();record={'pair':pair,'old_fact':old,'new_fact':new,'applied':False}
  try:
   req=make_payload(old,new,source_map,model=cfg['model'],max_tokens=cfg['max_tokens'],time_order=pair['time_order']);record['request']=req
   response=requests.post(cfg['base_url'].rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=req,timeout=None);record['http_status']=response.status_code;record['raw_http_body']=response.text;response.raise_for_status();data=response.json();record['response']=data;choice=data['choices'][0]
   record['result']={'status':'truncated','relation':None,'action':'DEFER','applied':False} if choice.get('finish_reason')=='length' else parse_relation(choice['message'].get('content'),old,new,source_map)
  except Exception as e:record['result']={'status':'technical_error','relation':None,'action':'DEFER','applied':False,'error':type(e).__name__}
  record['elapsed_seconds']=round(time.monotonic()-start,2);write(path,record);return record
 records=[]
 with ThreadPoolExecutor(max_workers=cfg['max_concurrent_requests']) as pool:
  futures=[pool.submit(work,x) for x in pairs]
  for i,future in enumerate(as_completed(futures),1):
   r=future.result();records.append(r);res=r['result'];usage=r.get('response',{}).get('usage') or {}
   print(f'PAIR {i}/{len(pairs)} status={res["status"]} relation={res.get("relation")} tokens={usage.get("total_tokens","not_returned")}',flush=True)
 summary={'pairs':len(pairs),'statuses':dict(Counter(r['result']['status'] for r in records)),'relations':dict(Counter(r['result'].get('relation') or 'UNKNOWN' for r in records)),'total_tokens':sum(r.get('response',{}).get('usage',{}).get('total_tokens',0) for r in records),'applied_updates':sum(bool(r['applied']) for r in records),'semantic_support_verified':False,'question_or_gold_used':False}
 write(a.output/'summary.json',summary);print('COMPLETED',json.dumps(summary),flush=True)
if __name__=='__main__':main()
