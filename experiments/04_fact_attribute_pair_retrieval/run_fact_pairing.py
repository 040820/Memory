#!/usr/bin/env python3
"""Gold-free session fact extraction, then attribute-key candidate retrieval."""
import argparse,hashlib,json,os,sys,tempfile,time
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import requests,yaml
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from fact_extraction import create_payload,parse_facts
from pair_retrieval import retrieve_pairs

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):
 with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=path.parent,delete=False) as f:
  tmp=Path(f.name)
  try:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  except BaseException:tmp.unlink(missing_ok=True);raise
 os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--retry-failures',action='store_true');a=p.parse_args()
 inventory=json.loads(a.input.read_text(encoding='utf8'));sources=inventory['sources'];cfg=yaml.safe_load(a.config.read_text(encoding='utf8'))
 if cfg.get('enable_thinking') is not False or cfg.get('temperature')!=0:raise ValueError('unexpected config')
 groups=defaultdict(list)
 for src in sources:
  if src['speaker']!='user':raise ValueError('source is not user turn')
  groups[src['session_id']].append(src)
 groups={k:sorted(v,key=lambda x:x['turn_index']) for k,v in groups.items()}
 manifest={'input_sha256':sha(a.input),'config_sha256':sha(a.config),'runner_sha256':sha(Path(__file__)),'modules':{p.name:sha(p) for p in sorted((ROOT/'src').glob('*.py'))},'source_turns':len(sources),'sessions':len(groups),'question_or_gold_used':False,'model':cfg['model'],'applied':False}
 if a.resume:
  if json.loads((a.output/'manifest.json').read_text())!=manifest:raise ValueError('resume manifest mismatch')
 else:
  a.output.mkdir(parents=True,exist_ok=False);(a.output/'sessions').mkdir();write(a.output/'manifest.json',manifest)
 key=Path(cfg['api_key_file']).read_text().strip()
 if not key:raise ValueError('empty API key')
 jobs=[];final=[]
 for sid,batch in sorted(groups.items()):
  name=hashlib.sha256(sid.encode()).hexdigest();old=sorted((a.output/'sessions').glob(name+'.attempt_*.json'))
  if old:
   record=json.loads(old[-1].read_text(encoding='utf8'))
   if record['result']['status']=='parsed' or not a.retry_failures:final.append(record);continue
  jobs.append((name,len(old)+1,batch))
 def work(job):
  name,num,batch=job;start=time.monotonic();record={'session_key':name,'attempt':num,'source_ids':[s['evidence_id'] for s in batch],'applied':False}
  try:
   payload=create_payload(batch,model=cfg['model'],max_tokens=cfg['max_tokens']);record['request']=payload
   r=requests.post(cfg['base_url'].rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload,timeout=None);record['http_status']=r.status_code;record['raw_http_body']=r.text;r.raise_for_status();data=r.json();record['response']=data;choice=data['choices'][0]
   record['result']={'status':'truncated','facts':[],'skipped':[]} if choice.get('finish_reason')=='length' else parse_facts(choice['message'].get('content'),batch)
  except Exception as exc:record['result']={'status':'technical_error','facts':[],'skipped':[],'error':type(exc).__name__}
  record['elapsed_seconds']=round(time.monotonic()-start,2);write(a.output/'sessions'/f'{name}.attempt_{num:03d}.json',record);return record
 done=len(final)
 if done:print('CHECKPOINTS_REUSED',done,'/',len(groups),flush=True)
 with ThreadPoolExecutor(max_workers=cfg['max_concurrent_requests']) as pool:
  futures=[pool.submit(work,j) for j in jobs]
  for future in as_completed(futures):
   record=future.result();final.append(record);done+=1;r=record['result'];usage=record.get('response',{}).get('usage') or {}
   print(f'SESSION {done}/{len(groups)} status={r["status"]} facts={len(r["facts"])} skipped={len(r["skipped"])} tokens={usage.get("total_tokens","not_returned")}',flush=True)
 facts=[f for r in final for f in r['result']['facts']];skip=[x for r in final for x in r['result']['skipped']];ids=[f['fact_id'] for f in facts]
 if len(ids)!=len(set(ids)):raise ValueError('fact id collision across sessions')
 facts.sort(key=lambda f:(f['session_time'],f['session_id'],f['turn_index'],f['fact_id']))
 write(a.output/'facts_snapshot.json',{'facts':facts,'skipped_candidates':skip,'question_or_gold_used':False})
 paired=retrieve_pairs(facts,max_candidates=cfg['max_candidates_per_fact']);write(a.output/'pair_candidates.json',paired)
 summary={'source_turns':len(sources),'sessions':len(groups),'session_statuses':dict(Counter(r['result']['status'] for r in final)),'fact_count':len(facts),'skipped_count':len(skip),'skipped_reasons':dict(Counter(x['reason'] for x in skip)),'unknown_attribute':sum(f['attribute_key'].strip().upper()=='UNKNOWN' for f in facts),'unknown_object':sum(f['object_key'].strip().upper()=='UNKNOWN' for f in facts),'unknown_scope':sum(f['scope_key'].strip().upper()=='UNKNOWN' for f in facts),'pairs':len(paired['pairs']),'unindexed_facts':len(paired['unindexed_fact_ids']),'truncated_candidates':sum(paired['truncated_by_new_fact'].values()),'total_llm_tokens':sum(r.get('response',{}).get('usage',{}).get('total_tokens',0) for r in final),'candidate_only':True,'applied':False}
 write(a.output/'summary.json',summary);print('COMPLETED',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
