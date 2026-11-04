#!/usr/bin/env python3
"""Execute frozen, label-free relation examples; checkpoint every response."""
import argparse,hashlib,json,os,sys,tempfile,time
from pathlib import Path
import requests,yaml
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from relation import make_payload,parse_proposal

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,obj):
 with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=path.parent,delete=False) as f:
  tmp=Path(f.name);json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 cases=json.loads(a.input.read_text(encoding='utf8'));cfg=yaml.safe_load(a.config.read_text(encoding='utf8'))
 if cfg.get('enable_thinking') is not False or cfg.get('temperature')!=0:raise ValueError('invalid config')
 ids=[x['id'] for x in cases]
 if len(ids)!=len(set(ids)):raise ValueError('duplicate case id')
 manifest={'input_sha256':sha(a.input),'config_sha256':sha(a.config),'relation_sha256':sha(ROOT/'src/relation.py'),'runner_sha256':sha(Path(__file__)),'cases':len(cases),'model':cfg['model'],'applied':False}
 if a.resume:
  if json.loads((a.output/'manifest.json').read_text())!=manifest:raise ValueError('resume manifest mismatch')
 else:a.output.mkdir(parents=True,exist_ok=False);(a.output/'cases').mkdir();write(a.output/'manifest.json',manifest)
 key=Path(cfg['api_key_file']).expanduser().read_text().strip()
 if not key:raise ValueError('empty API key')
 for i,case in enumerate(cases,1):
  f=a.output/'cases'/(hashlib.sha256(case['id'].encode()).hexdigest()+'.json')
  if f.exists():print(f'[{i}/{len(cases)}] cached {case["id"]}',flush=True);continue
  start=time.monotonic();row={'case_id':case['id'],'old':case['old'],'new':case['new'],'applied':False}
  try:
   payload=make_payload(case,model=cfg['model'],max_tokens=cfg['max_tokens']);row['request']=payload
   r=requests.post(cfg['base_url'].rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload,timeout=None)
   row['http_status']=r.status_code;row['raw_http_body']=r.text;r.raise_for_status();data=r.json();row['response']=data;choice=data['choices'][0]
   if choice.get('finish_reason')=='length':proposal={'status':'truncated','relation':None,'action':'DEFER','applied':False}
   else:proposal=parse_proposal(choice['message'].get('content'),case)
   row['result']=proposal
  except Exception as exc:row['result']={'status':'technical_error','relation':None,'action':'DEFER','applied':False,'error':type(exc).__name__}
  row['elapsed_seconds']=round(time.monotonic()-start,2);write(f,row);res=row['result'];usage=row.get('response',{}).get('usage') or {}
  print(f'[{i}/{len(cases)}] {case["id"]} status={res["status"]} relation={res["relation"]} action={res["action"]} tokens={usage.get("total_tokens","not_returned")} elapsed={row["elapsed_seconds"]}s',flush=True)
 print('COMPLETED: proposals only; no memory updates',flush=True)
if __name__=='__main__':main()
