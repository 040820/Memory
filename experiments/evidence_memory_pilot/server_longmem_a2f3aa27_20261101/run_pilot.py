import asyncio,datetime,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/cwf/Code/LiCoMemory-main'); EXP=ROOT/'experiments/evidence_memory_pilot'; OUT=EXP/'server_longmem_a2f3aa27_20261101'; DATA=ROOT/'dataset/data/longmemeval_s_cleaned.json'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(EXP))
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:os.environ.pop(k,None)
os.environ['NO_PROXY']='127.0.0.1,localhost';os.environ['no_proxy']=os.environ['NO_PROXY']
import torch
from remote_embedding import RemoteEmbeddingClient,install_manager_adapter
from qwen_adapter import install_non_thinking_adapter
from base.embeddings import EmbeddingManager
from base.llm import LLMManager
from init.config import Config
from init.graph_rag import GraphRAG
from dataset.longmem import build_context

def save(data):
 (OUT/'status.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf8')
async def main():
 start=time.monotonic();status={'status':'started','question_id':'a2f3aa27','gold_loaded_into_system':False,'input_protocol':'complete original-order haystack; no timestamp filtering','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};save(status)
 try:
  with DATA.open(encoding='utf8') as f:raw=json.load(f)
  item=next(x for x in raw if x.get('question_id')==status['question_id'])
  ids,dates,sessions=item['haystack_session_ids'],item['haystack_dates'],item['haystack_sessions']
  if not len(ids)==len(dates)==len(sessions):raise ValueError('misaligned sessions')
  corpus=[]; input_sessions=[]
  for sid,date,turns in zip(ids,dates,sessions):
   clean=[{'role':t.get('role',''),'content':t.get('content','')} for t in turns]
   input_sessions.append({'session_id':sid,'session_time':date,'messages':clean})
   corpus.append({'session_id':sid,'session_time':date,'context':build_context(clean)})
  visible={'question_id':item['question_id'],'question':item['question'],'question_time':item['question_date'],'sessions':input_sessions}
  del raw,item
  (OUT/'visible_input.json').write_text(json.dumps(visible,ensure_ascii=False,indent=2),encoding='utf8')
  status.update(session_count=len(corpus),message_count=sum(len(s['messages']) for s in input_sessions));save(status)
  config=Config.parse(EXP/'baseline_qwen.yaml');config.data_type='LongmemEval';config.index_name='longmem_graph';config.llm.base_url='http://127.0.0.1:8011/v1';config.query_llm.base_url=config.llm.base_url
  install_non_thinking_adapter(LLMManager,config.llm.model);install_manager_adapter(EmbeddingManager,RemoteEmbeddingClient('http://127.0.0.1:8015/v1','bge-m3',1024),torch)
  graph=GraphRAG(config,str(OUT));await graph.insert(corpus)
  status['graph_stats']=graph.core.graph.get_graph_stats();status['status']='graph_complete';save(status)
  answer=await graph.query(visible['question'],question_time=visible['question_time']);status['response']=answer
  text=answer.get('answer','') if isinstance(answer,dict) else str(answer)
  status['status']='completed_unscored' if text and not text.startswith(('Error processing query:','Unable to generate answer')) else 'failed'
 except Exception as e:status.update(status='error',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
 status['elapsed_seconds']=round(time.monotonic()-start,2);save(status)
 print('PILOT_STATUS',status['status'],'SECONDS',status['elapsed_seconds'],flush=True)
 if status['status']!='completed_unscored':raise SystemExit(1)
asyncio.run(main())
