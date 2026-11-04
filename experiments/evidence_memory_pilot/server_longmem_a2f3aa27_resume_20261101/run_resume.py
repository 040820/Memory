import asyncio,datetime,hashlib,json,os,shutil,sys,time,traceback
from pathlib import Path
from checkpoint import atomic_json,save_checkpoint,load_checkpoint
ROOT=Path('/home/cwf/Code/LiCoMemory-main');EXP=ROOT/'experiments/evidence_memory_pilot';OUT=Path(__file__).resolve().parent;OLD=EXP/'server_longmem_a2f3aa27_20261101';CACHE=OUT/'checkpoints';CACHE.mkdir(exist_ok=True)
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

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest()
def fhash(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_status(s):atomic_json(OUT/'status.json',s)
async def main():
 status={'status':'initializing','case_id':'a2f3aa27','gold_loaded_into_model':False,'previous_run_reused':'44 session summaries only; prior extraction was not persisted','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};write_status(status);started=time.monotonic()
 try:
  visible=json.loads((OLD/'visible_input.json').read_text(encoding='utf8'))
  summaries=json.loads((OLD/'session_summaries.json').read_text(encoding='utf8'))
  session_ids=[s['session_id'] for s in visible['sessions']]
  if len(summaries)!=len(session_ids) or {s['session_id'] for s in summaries}!=set(session_ids):raise ValueError('summary sessions mismatch')
  config_file=EXP/'baseline_qwen.yaml';files=[OLD/'visible_input.json',OLD/'session_summaries.json',config_file,ROOT/'base/llm.py',ROOT/'coregraph/dialogue_extractor.py',ROOT/'coregraph/dynamic_memory.py',ROOT/'chunking/dialog_chunk_processor.py',EXP/'qwen_adapter.py',EXP/'remote_embedding.py',OUT/'run_resume.py',OUT/'checkpoint.py']
  manifest={'input_and_code_sha256':{str(p):fhash(p) for p in files},'llm':'qwen3-8b','llm_url':'http://127.0.0.1:8011/v1','thinking':False,'embedding':'bge-m3','embedding_url':'http://127.0.0.1:8015/v1','embedding_dimensions':1024,'protocol':'full source-order visible input, no gold, summary reuse, per-chunk checkpoint'}
  mf=OUT/'manifest.json'
  if mf.exists() and json.loads(mf.read_text(encoding='utf8'))!=manifest:raise ValueError('resume manifest mismatch; use a new directory')
  if not mf.exists():atomic_json(mf,manifest)
  summary_copy=OUT/'session_summaries.json'
  if not summary_copy.exists():shutil.copyfile(OLD/'session_summaries.json',summary_copy)
  elif fhash(summary_copy)!=fhash(OLD/'session_summaries.json'):shutil.copyfile(OLD/'session_summaries.json',summary_copy)
  config=Config.parse(config_file);config.data_type='LongmemEval';config.index_name='longmem_graph';config.llm.base_url='http://127.0.0.1:8011/v1';config.query_llm.base_url=config.llm.base_url
  install_non_thinking_adapter(LLMManager,config.llm.model);install_manager_adapter(EmbeddingManager,RemoteEmbeddingClient('http://127.0.0.1:8015/v1','bge-m3',1024),torch)
  corpus=[{'session_id':s['session_id'],'session_time':s['session_time'],'context':build_context(s['messages'])} for s in visible['sessions']]
  graph=GraphRAG(config,str(OUT));chunks=graph.core.chunk_processor.process_corpus(corpus)
  if len({str(c['chunk_id']) for c in chunks})!=len(chunks):raise ValueError('duplicate chunk IDs')
  extractor=graph.core.graph.dialogue_extractor;llm=graph.core.llm_manager;sem=asyncio.Semaphore(4)
  async def one(index,chunk):
   key=digest(chunk);path=CACHE/('%04d.json'%index)
   if path.exists():return load_checkpoint(path,key)
   async with sem:
    enhanced={'text':chunk.get('text',''),'session_time':str(chunk.get('session_time','unknown')),'session_id':str(chunk.get('session_id','unknown'))}
    prompt=extractor.extraction_prompt.format(text=json.dumps(enhanced))
    response=await llm._generate_internal(prompt)
    save_checkpoint(path,key,response)
    print('CHECKPOINT',index+1,'OF',len(chunks),'completed',len(list(CACHE.glob('*.json'))),flush=True)
    return response
  status.update(status='extracting',total_chunks=len(chunks),previous_summary_count=len(summaries),completed_chunks=sum((CACHE/('%04d.json'%i)).exists() for i in range(len(chunks))));write_status(status)
  responses=[]
  for start in range(0,len(chunks),4):
   responses.extend(await asyncio.gather(*(one(i,chunks[i]) for i in range(start,min(start+4,len(chunks))))))
   status['completed_chunks']=len(responses);write_status(status)
  async def replay(selected,progress_bar=None):
   if len(selected)!=len(chunks) or any(digest(a)!=digest(b) for a,b in zip(selected,chunks)):raise ValueError('replay chunk mismatch')
   all_entities=[];all_relationships=[]
   for response,chunk in zip(responses,chunks):
    entities,relationships=extractor._parse_dialogue_response(response)
    for ent in entities:
     ent['chunk_id']=chunk.get('chunk_id',0);text=chunk.get('text','');ent['source_text']=text[:100]+'...' if len(text)>100 else text
    for rel in relationships:rel['chunk_id']=chunk.get('chunk_id',0)
    all_entities.extend(entities);all_relationships.extend(relationships)
   if progress_bar:progress_bar.update(len(chunks))
   return all_entities,all_relationships
  extractor.extract_from_chunks=replay
  status['status']='constructing_graph';write_status(status)
  await graph.core.graph.build_graph(chunks,force=True,add=False)
  graph.core.graph.save_graph(str(OUT/'longmem_graph.pkl'))
  status.update(status='querying',graph_stats=graph.core.graph.get_graph_stats());write_status(status)
  answer=await graph.query(visible['question'],question_time=visible['question_time']);status['response']=answer
  text=answer.get('answer','') if isinstance(answer,dict) else str(answer)
  status['status']='completed_unscored' if text and not text.startswith(('Error processing query:','Unable to generate answer')) else 'failed'
 except Exception as e:
  status.update(status='error',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
 status['elapsed_seconds']=round(time.monotonic()-started,2);write_status(status)
 print('RESUME_STATUS',status['status'],'SECONDS',status['elapsed_seconds'],flush=True)
 if status['status']!='completed_unscored':raise SystemExit(1)
asyncio.run(main())
