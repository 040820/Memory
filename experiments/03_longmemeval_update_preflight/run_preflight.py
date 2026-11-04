#!/usr/bin/env python3
"""One-question development diagnostic, gold-free at every model stage."""
import argparse,hashlib,json,os,sys,tempfile,time
from pathlib import Path
import requests,yaml
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from preflight import extract_sources,select_pairs,query_seeds,choose_pairs
from relation import make_payload,parse_proposal
from answer import payload as answer_payload,validate as validate_answer

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
 with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=p.parent,delete=False) as f:
  tmp=Path(f.name);json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def load(p):return json.loads(p.read_text(encoding='utf8'))
def embeds(url,model,inputs,dims):
 r=requests.post(url.rstrip('/')+'/embeddings',json={'model':model,'input':inputs},timeout=None);r.raise_for_status();data=r.json()['data'];data=sorted(data,key=lambda d:d['index'])
 if len(data)!=len(inputs) or [x['index'] for x in data]!=list(range(len(inputs))):raise ValueError('embedding indices mismatch')
 vectors=[x['embedding'] for x in data]
 if any(len(x)!=dims for x in vectors) or not np.isfinite(np.asarray(vectors)).all():raise ValueError('embedding dimensions or values invalid')
 return vectors

def llm(url,key,request):
 response=requests.post(url.rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=request,timeout=None);status=response.status_code;body=response.text;response.raise_for_status();return status,body,response.json()

def main():
 a=argparse.ArgumentParser();a.add_argument('--input',type=Path,required=True);a.add_argument('--config',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--resume',action='store_true');args=a.parse_args()
 data=load(args.input);cfg=yaml.safe_load(args.config.read_text());
 if data.get('question_id')!=cfg['question_id'] or cfg.get('enable_thinking') is not False:raise ValueError('config mismatch')
 if set(data)-{'question_id','question','question_time','sessions'}:raise ValueError('visible input has unexpected top-level fields')
 manifest={'input_sha256':sha(args.input),'config_sha256':sha(args.config),'runner_sha256':sha(Path(__file__)),'source_sha256':{p.name:sha(p) for p in sorted((ROOT/'src').glob('*.py'))},'question_id':cfg['question_id'],'applied':False}
 if args.resume:
  if load(args.output/'manifest.json')!=manifest:raise ValueError('resume inputs or code changed')
 else:
  args.output.mkdir(parents=True,exist_ok=False)
  for x in ('embedding_batches','relation_cases','answers'): (args.output/x).mkdir()
  write(args.output/'manifest.json',manifest);write(args.output/'visible_input.json',data)
 sources,stats=extract_sources(data);write(args.output/'source_inventory.json',{'sources':sources,'stats':stats,'time_basis':'session level, minute, timezone unknown'})
 print('SOURCE_PREPARED',len(sources),'STATS',stats,flush=True)
 vectors=[];size=cfg['embedding_batch_size']
 for start in range(0,len(sources),size):
  f=args.output/'embedding_batches'/f'{start:05d}.json';batch=sources[start:start+size];expected=[s['evidence_id'] for s in batch]
  if f.exists():
   cached=load(f)
   if cached['evidence_ids']!=expected:raise ValueError('embedding checkpoint source mismatch')
   vec=cached['vectors']
  else:
   vec=embeds(cfg['embedding_url'],cfg['embedding_model'],[s['text'] for s in batch],cfg['embedding_dimensions']);write(f,{'evidence_ids':expected,'vectors':vec})
  vectors.extend(vec);print(f'EMBEDDINGS {len(vectors)}/{len(sources)}',flush=True)
 pairs=select_pairs(sources,vectors,min_similarity=cfg['candidate_min_similarity']);write(args.output/'global_candidate_pairs.json',{'pairs':pairs,'construction_uses_question_text':False,'candidate_not_proof':True})
 qfile=args.output/'question_embedding.json'
 if qfile.exists():qvector=load(qfile)['vector']
 else:qvector=embeds(cfg['embedding_url'],cfg['embedding_model'],[data['question']],cfg['embedding_dimensions'])[0];write(qfile,{'vector':qvector})
 seeds=query_seeds(vectors,qvector,cfg['query_top_k']);selected=choose_pairs(pairs,seeds,cfg['relation_max_pairs']);indices=sorted(set(seeds+[p['old_index'] for p in selected]+[p['new_index'] for p in selected]));shown=[sources[i] for i in indices]
 write(args.output/'retrieval.json',{'seed_indices':seeds,'selected_pairs':selected,'shown_source_indices':indices,'shown_sources':shown,'pair_similarity_is_not_proof':True})
 print('RETRIEVAL seeds',len(seeds),'global_pairs',len(pairs),'selected_pairs',len(selected),'shown_sources',len(shown),flush=True)
 key=Path(cfg['api_key_file']).read_text().strip();suggestions=[]
 for i,pair in enumerate(selected,1):
  file=args.output/'relation_cases'/f'{i:03d}.json';item={'id':str(i),'old':pair['old'],'new':pair['new']}
  if file.exists():record=load(file)
  else:
   started=time.monotonic();record={'pair':pair,'applied':False}
   try:
    req=make_payload(item,model=cfg['llm_model'],max_tokens=cfg['max_tokens']);record['request']=req
    status,body,response=llm(cfg['llm_base_url'],key,req);record.update(http_status=status,raw_http_body=body,response=response)
    choice=response['choices'][0];record['result']={'status':'truncated','relation':None,'action':'DEFER','applied':False} if choice.get('finish_reason')=='length' else parse_proposal(choice['message'].get('content'),item)
   except Exception as exc:record['result']={'status':'technical_error','relation':None,'action':'DEFER','applied':False,'error':type(exc).__name__}
   record['elapsed_seconds']=round(time.monotonic()-started,2);write(file,record)
  r=record['result'];print(f'RELATION {i}/{len(selected)} status={r["status"]} relation={r["relation"]} action={r["action"]}',flush=True)
  if r['status']=='proposed':suggestions.append({'old_evidence_id':pair['old']['evidence_id'],'new_evidence_id':pair['new']['evidence_id'],'relation':r['relation'],'action':r['action'],'reason':r['model_output']['reason'],'citations':r['citations']})
 for mode in ('baseline','relation'):
  file=args.output/'answers'/f'{mode}.json'
  if file.exists():record=load(file)
  else:
   started=time.monotonic();record={'mode':mode,'applied':False}
   try:
    req=answer_payload(data['question'],data['question_time'],shown,suggestions if mode=='relation' else None,model=cfg['llm_model'],max_tokens=cfg['max_tokens']);record['request']=req
    status,body,response=llm(cfg['llm_base_url'],key,req);record.update(http_status=status,raw_http_body=body,response=response);choice=response['choices'][0]
    record['result']={'status':'truncated','citations_exist':False} if choice.get('finish_reason')=='length' else validate_answer(choice['message'].get('content'),shown)
   except Exception as exc:record['result']={'status':'technical_error','citations_exist':False,'error':type(exc).__name__}
   record['elapsed_seconds']=round(time.monotonic()-started,2);write(file,record)
  print('ANSWER',mode,'status',record['result']['status'],'citations_exist',record['result'].get('citations_exist'),flush=True)
 print('COMPLETED gold-free development run; no memory updated',flush=True)
if __name__=='__main__':main()
