#!/usr/bin/env python3
"""Same evidence; compare an optional relation proposal. Gold excluded."""
import argparse,hashlib,json,os,sys,tempfile,time
from pathlib import Path
import requests,yaml
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from reader import make_reader_payload,validate_answer

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=p.parent,delete=False) as f:
  tmp=Path(f.name);json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--input',type=Path,required=True);a.add_argument('--relation-run',type=Path,required=True);a.add_argument('--config',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--resume',action='store_true');args=a.parse_args()
 items=json.loads(args.input.read_text());cfg=yaml.safe_load(args.config.read_text());relation_files=sorted((args.relation_run/'cases').glob('*.json'))
 relations={}
 for p in relation_files:
  r=json.loads(p.read_text());relations[r['case_id']]=r
 manifest={'input_sha256':sha(args.input),'config_sha256':sha(args.config),'reader_sha256':sha(ROOT/'src/reader.py'),'runner_sha256':sha(Path(__file__)),'relation_files_sha256':{p.name:sha(p) for p in relation_files},'questions':len(items),'model':cfg['model'],'applied':False}
 if args.resume:
  if json.loads((args.output/'manifest.json').read_text())!=manifest:raise ValueError('resume manifest mismatch')
 else:args.output.mkdir(parents=True,exist_ok=False);(args.output/'cases').mkdir();write(args.output/'manifest.json',manifest)
 key=Path(cfg['api_key_file']).expanduser().read_text().strip()
 for i,item in enumerate(items,1):
  rel=relations[item['pair_id']]['result'];proposal={'relation':rel['relation'],'action':rel['action'],'reason':rel.get('model_output',{}).get('reason','')}
  for mode in ('baseline','relation'):
   path=args.output/'cases'/f'{item["id"]}_{mode}.json'
   if path.exists():print(f'[{i}/{len(items)}] {item["id"]} {mode} cached',flush=True);continue
   row={'question_id':item['id'],'mode':mode,'applied':False};start=time.monotonic()
   try:
    payload=make_reader_payload(item,proposal if mode=='relation' else None,model=cfg['model'],max_tokens=cfg['max_tokens']);row['request']=payload
    response=requests.post(cfg['base_url'].rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload,timeout=None);row['http_status']=response.status_code;row['raw_http_body']=response.text;response.raise_for_status();data=response.json();row['response']=data;choice=data['choices'][0]
    row['result']={'status':'truncated','citation_ids_exist':False,'semantic_support_verified':False} if choice.get('finish_reason')=='length' else validate_answer(choice['message'].get('content'),item)
   except Exception as e:row['result']={'status':'technical_error','citation_ids_exist':False,'semantic_support_verified':False,'error':type(e).__name__}
   row['elapsed_seconds']=round(time.monotonic()-start,2);write(path,row);usage=row.get('response',{}).get('usage') or {}
   print(f'[{i}/{len(items)}] {item["id"]} {mode} status={row["result"]["status"]} citations_exist={row["result"].get("citation_ids_exist")} tokens={usage.get("total_tokens","not_returned")}',flush=True)
 print('COMPLETED: answers are unverified; no memory updates',flush=True)
if __name__=='__main__':main()
