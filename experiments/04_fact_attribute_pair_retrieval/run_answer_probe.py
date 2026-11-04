#!/usr/bin/env python3
"""Compare one fact-pair hint against the frozen experiment 03 retrieved context."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import requests,yaml
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'03_longmemeval_update_preflight/src'))
from answer import payload,validate

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline-run',type=Path,required=True);p.add_argument('--relations',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 inp=json.loads((a.baseline_run/'visible_input.json').read_text());retr=json.loads((a.baseline_run/'retrieval.json').read_text());sources=retr['shown_sources'];ids={s['evidence_id'] for s in sources};records=json.loads(a.relations.read_text());cfg=yaml.safe_load(a.config.read_text())
 selected=[r for r in records if r['pair']['old_evidence_id'] in ids and r['pair']['new_evidence_id'] in ids and r['result']['status']=='proposed']
 hints=[{'old_evidence_id':r['pair']['old_evidence_id'],'new_evidence_id':r['pair']['new_evidence_id'],'old_fact':{k:r['old_fact'][k] for k in ('value','quote','polarity','modality')},'new_fact':{k:r['new_fact'][k] for k in ('value','quote','polarity','modality')},'relation':r['result']['relation'],'action':r['result']['action'],'reason':r['result']['model_reason'],'caveats':r['result']['caveats']} for r in selected]
 a.output.mkdir(parents=True,exist_ok=False)
 manifest={'baseline_input_sha256':sha(a.baseline_run/'visible_input.json'),'retrieval_sha256':sha(a.baseline_run/'retrieval.json'),'baseline_answer_sha256':sha(a.baseline_run/'answers/baseline.json'),'relation_results_sha256':sha(a.relations),'config_sha256':sha(a.config),'runner_sha256':sha(Path(__file__)),'reader_sha256':sha(ROOT.parent/'03_longmemeval_update_preflight/src/answer.py'),'hint_count':len(hints),'selection':'both pair evidence IDs appear in frozen retrieved sources, with proposed relation; no gold','applied':False}
 (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 req=payload(inp['question'],inp['question_time'],sources,hints,model=cfg['model'],max_tokens=cfg['max_tokens']);start=time.monotonic();record={'request':req,'applied':False}
 try:
  key=Path(cfg['api_key_file']).read_text().strip();r=requests.post(cfg['base_url'].rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=req,timeout=None);record['http_status']=r.status_code;record['raw_http_body']=r.text;r.raise_for_status();d=r.json();record['response']=d;choice=d['choices'][0]
  record['result']={'status':'truncated','citations_exist':False} if choice.get('finish_reason')=='length' else validate(choice['message'].get('content'),sources)
 except Exception as exc:record['result']={'status':'technical_error','citations_exist':False,'error':type(exc).__name__}
 record['elapsed_seconds']=round(time.monotonic()-start,2);(a.output/'answer.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print('selected_hints',len(hints),'status',record['result']['status'],'answer',record['result'].get('answer'),'tokens',record.get('response',{}).get('usage',{}).get('total_tokens','not_returned'))
if __name__=='__main__':main()
