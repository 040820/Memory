"""Diagnostic reader for evidence-only vs evidence-plus-relation prompts."""
import json
PROMPT='''Answer the question using only the two supplied user statements. Be careful about time, plans, corrections, contradictions and uncertainty. Later mention alone is not proof of correction or replacement. If evidence does not establish an answer, say so; do not invent the user's current state. If a relation proposal is present, treat it as a possibly incorrect suggestion, and check it against both original statements. Return JSON {"answer":"concise answer","citations":["source id", ...]}.'''
def make_reader_payload(item,proposal,*,model,max_tokens):
 safe={'question':item['question'],'old':{k:item['old'][k] for k in ('evidence_id','speaker','text')},'new':{k:item['new'][k] for k in ('evidence_id','speaker','text')}}
 if proposal is not None:safe['relation_proposal']={k:proposal[k] for k in ('relation','action','reason') if k in proposal}
 return {'model':model,'messages':[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps(safe,ensure_ascii=False)}],'temperature':0,'max_tokens':max_tokens,'enable_thinking':False}
def validate_answer(raw,item):
 result={'status':'invalid_json','citation_ids_exist':False,'semantic_support_verified':False}
 try:data=json.loads(raw)
 except (TypeError,ValueError):return result
 if isinstance(data,list) and len(data)==1 and isinstance(data[0],dict):data=data[0]
 if not isinstance(data,dict) or not isinstance(data.get('answer'),str) or not data['answer'].strip() or not isinstance(data.get('citations'),list) or not all(isinstance(x,str) for x in data['citations']):result['status']='invalid_schema';return result
 result.update(status='parsed',answer=data['answer'],citations=data['citations'],citation_ids_exist=bool(data['citations']) and all(x in (item['old']['evidence_id'],item['new']['evidence_id']) for x in data['citations']))
 return result
