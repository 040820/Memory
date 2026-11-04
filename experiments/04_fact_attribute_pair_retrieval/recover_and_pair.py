#!/usr/bin/env python3
"""Recover failed session extraction and propose lexical-normalized fact pairs."""
import argparse,hashlib,json,os,sys,tempfile,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import requests,yaml
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from fact_extraction import create_payload,parse_facts
from normalization import normalize_attribute
from pair_retrieval import retrieve_pairs

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
 with tempfile.NamedTemporaryFile('w',dir=p.parent,encoding='utf8',delete=False) as f:
  tmp=Path(f.name)
  try:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  except BaseException:tmp.unlink(missing_ok=True);raise
 os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 prev=json.loads((a.previous/'manifest.json').read_text());
 if prev['input_sha256']!=sha(a.input) or prev['config_sha256']!=sha(a.config):raise ValueError('previous inputs differ')
 sources=json.loads(a.input.read_text())['sources'];byid={s['evidence_id']:s for s in sources};cfg=yaml.safe_load(a.config.read_text());files=sorted((a.previous/'sessions').glob('*.json'));records=[json.loads(f.read_text()) for f in files]
 if len(records)!=prev['sessions']:raise ValueError('missing original session output')
 manifest={'input_sha256':sha(a.input),'config_sha256':sha(a.config),'previous_manifest_sha256':sha(a.previous/'manifest.json'),'previous_session_hashes':{p.name:sha(p) for p in files},'modules':{p.name:sha(p) for p in sorted((ROOT/'src').glob('*.py'))},'runner_sha256':sha(Path(__file__)),'question_or_gold_used':False,'applied':False}
 if a.resume:
  if json.loads((a.output/'manifest.json').read_text())!=manifest:raise ValueError('resume manifest mismatch')
 else:
  a.output.mkdir(parents=True,exist_ok=False);(a.output/'recovery').mkdir();write(a.output/'manifest.json',manifest)
 key=Path(cfg['api_key_file']).read_text().strip();results=[];jobs=[]
 for record in records:
  status=record['result']['status'];batch=[byid[e] for e in record['source_ids']]
  if status=='parsed':results.append({'origin':'original_session','source_ids':record['source_ids'],'result':record['result']});continue
  if status=='invalid_schema' and 'response' in record:
   output=record['response']['choices'][0]['message'].get('content');parsed=parse_facts(output,batch)
   if parsed['status']=='parsed':
    recovered={'origin':'posthoc_reparse_no_model_call','source_ids':record['source_ids'],'result':parsed};write(a.output/'recovery'/(record['session_key']+'.reparsed.json'),recovered);results.append(recovered);continue
  for src in batch:jobs.append(src)
 def work(src):
  path=a.output/'recovery'/(src['evidence_id']+'.json')
  if path.exists():return json.loads(path.read_text())
  started=time.monotonic();r={'origin':'split_turn_retry','source_ids':[src['evidence_id']],'applied':False}
  try:
   payload=create_payload([src],model=cfg['model'],max_tokens=cfg['max_tokens']);r['request']=payload
   response=requests.post(cfg['base_url'].rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload,timeout=None);r['http_status']=response.status_code;r['raw_http_body']=response.text;response.raise_for_status();data=response.json();r['response']=data;choice=data['choices'][0]
   r['result']={'status':'truncated','facts':[],'skipped':[]} if choice.get('finish_reason')=='length' else parse_facts(choice['message'].get('content'),[src])
  except Exception as e:r['result']={'status':'technical_error','facts':[],'skipped':[],'error':type(e).__name__}
  r['elapsed_seconds']=round(time.monotonic()-started,2);write(path,r);return r
 print('REUSE',sum(x['origin']=='original_session' for x in results),'REPARSED',sum(x['origin']=='posthoc_reparse_no_model_call' for x in results),'SPLIT_RETRY_TURNS',len(jobs),flush=True)
 with ThreadPoolExecutor(max_workers=cfg['max_concurrent_requests']) as pool:
  pending=[pool.submit(work,s) for s in jobs]
  for i,f in enumerate(as_completed(pending),1):
   r=f.result();results.append(r);print(f'RECOVERY {i}/{len(jobs)} status={r["result"]["status"]} facts={len(r["result"]["facts"])}',flush=True)
 facts=[dict(x) for r in results for x in r['result']['facts']];skipped=[x for r in results for x in r['result']['skipped']]
 ids=[f['fact_id'] for f in facts]
 if len(ids)!=len(set(ids)):raise ValueError('duplicate fact IDs')
 for f in facts:
  norm=normalize_attribute(f['attribute_key'],f['object_key']);f['search_attribute_key']=norm['key'];f['attribute_normalization']=norm
 paired=retrieve_pairs(facts,max_candidates=cfg['max_candidates_per_fact'])
 write(a.output/'facts_snapshot.json',{'facts':facts,'skipped_candidates':skipped,'question_or_gold_used':False});write(a.output/'pair_candidates.json',paired)
 summary={'source_turns':len(sources),'original_session_count':len(records),'reused_sessions':sum(r['origin']=='original_session' for r in results),'posthoc_reparsed_sessions':sum(r['origin']=='posthoc_reparse_no_model_call' for r in results),'split_retry_turns':sum(r['origin']=='split_turn_retry' for r in results),'statuses':dict(Counter(r['result']['status'] for r in results)),'facts':len(facts),'skipped_candidates':len(skipped),'pairs':len(paired['pairs']),'pairs_by_path':dict(Counter(x['candidate_path'] for x in paired['pairs'])),'unindexed_facts':len(paired['unindexed_fact_ids']),'truncated_candidates':sum(paired['truncated_by_new_fact'].values()),'normalization_version':'lexical_v1_dev','question_or_gold_used':False,'candidate_only':True,'applied':False}
 write(a.output/'summary.json',summary);print('COMPLETED',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
