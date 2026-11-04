"""Diagnostic answer generation from the same retrieved sources."""
import json
PROMPT='''Answer the question only from the provided user statements. Consider conversation times and any uncertainty; do not treat the last sentence as automatically proving a correction. Distinguish current facts, past facts, plans and approximate values. When evidence is insufficient say so. If relation suggestions are provided, they are fallible and not facts; check them against the sources. Return JSON {"answer":"concise answer", "citations":["evidence ID"]}.'''

def payload(question,question_time,sources,suggestions,*,model,max_tokens):
 data={'question':question,'question_time':question_time,'sources':[{k:s[k] for k in ('evidence_id','text','session_time')} for s in sources]}
 if suggestions is not None:data['relation_suggestions']=suggestions
 return {'model':model,'messages':[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps(data,ensure_ascii=False)}],'temperature':0,'max_tokens':max_tokens,'enable_thinking':False}

def validate(raw,sources):
 try:data=json.loads(raw)
 except (TypeError,ValueError):return {'status':'invalid_json','citations_exist':False}
 if isinstance(data,list) and len(data)==1 and isinstance(data[0],dict):data=data[0]
 if not isinstance(data,dict) or not isinstance(data.get('answer'),str) or not data['answer'].strip() or not isinstance(data.get('citations'),list) or not all(isinstance(x,str) for x in data['citations']):return {'status':'invalid_schema','citations_exist':False}
 ids={s['evidence_id'] for s in sources};return {'status':'parsed','answer':data['answer'],'citations':data['citations'],'citations_exist':bool(data['citations']) and all(c in ids for c in data['citations']),'semantic_support_verified':False}
