"""V2.1 source restoration and component comparison, advisory only."""
import json
from . import semantic_review_v2 as base

COMPONENTS=('subject','predicate','value','polarity','time','modality','scope')
EXTRA = '''Before judging the candidate, provide source_restatement: a short description of the original utterance that preserves every relevant subject, negation, uncertainty, modality, numerical qualifier and scope. This restatement is your interpretation, not new evidence. Provide source_qualifiers: a list of verbatim short phrases from the source such as "hard to", "at most", "might", "plan", "not", or approximations; use [] if none.
Provide comparisons: an object with exactly these components: subject, predicate, value, polarity, time, modality, scope. For each use MATCH, MISMATCH, UNKNOWN or NOT_APPLICABLE comparing source and candidate. Consider the full candidate: SUPPORTED requires all relevant components supported. When the source does not establish a fact, use INSUFFICIENT. Use CONTRADICTED only if source explicitly rules out the same claim for the same subject, property, scope and time. Difficult or hard is not impossible; likely or approximate is not exact. A plan or possibility does not prove or disprove present residence. Do not infer the user shares the sister's occupation. If terms such as table capacity are ambiguous and the source only says more than four people are hard to fit, the capacity claim is INSUFFICIENT, not a definite contradiction. If a capacity is explicitly at most four and the candidate explicitly says at least six under the same circumstances, it is CONTRADICTED. Cite original text, never restatement, as evidence.'''


def review_payload(source,candidate,*,model,max_tokens):
 payload=base.review_payload(source,candidate,model=model,max_tokens=max_tokens)
 payload['messages'][0]['content']+='\n\n'+EXTRA
 return payload


def parse_and_route(raw,source,candidate):
 try:
  data=json.loads(raw)
  if isinstance(data,list) and len(data)==1:data=data[0]
  if not isinstance(data,dict):raise ValueError('invalid root')
  for wrapper in ('judgment','fields'):
   if wrapper not in data:continue
   nested=data[wrapper]
   if not isinstance(nested,dict) or any(k in data and data[k]!=v for k,v in nested.items()):raise ValueError('conflicting nested fields')
   data=dict(data,**nested)
  comps=data.get('comparisons')
  if isinstance(comps,dict):
   comps={k:('MISMATCH' if v=='CONTRADICTED' else v) for k,v in comps.items()}
   data['comparisons']=comps
 except (ValueError,TypeError):
  return {'schema_version':'semantic_review_v2_1','route':'TECHNICAL_ERROR','applied':False,'reasons':['INVALID_OR_CONFLICTING_SCHEMA'],'evidence_spans':[]}
 result=base.parse_and_route(json.dumps(data,ensure_ascii=False),source,candidate)
 result['schema_version']='semantic_review_v2_1'
 if result['route']=='TECHNICAL_ERROR':return result
 try:
  comps=data.get('comparisons');qualifiers=data.get('source_qualifiers');restatement=data.get('source_restatement')
  valid=isinstance(comps,dict) and all(isinstance(comps.get(k),str) and comps[k] in ('MATCH','MISMATCH','UNKNOWN','NOT_APPLICABLE') for k in COMPONENTS)
  valid=valid and isinstance(qualifiers,list) and all(isinstance(x,str) and bool(x.strip()) for x in qualifiers) and isinstance(restatement,str) and bool(restatement.strip())
  if not valid:
   result['route']='TECHNICAL_ERROR';result['reasons'].append('INVALID_V21_SCHEMA');return result
 except (ValueError,TypeError,AttributeError):
  result['route']='TECHNICAL_ERROR';result['reasons'].append('INVALID_V21_SCHEMA');return result
 result['model_review']['source_restatement']=restatement
 result['model_review']['source_qualifiers']=qualifiers
 result['model_review']['comparisons']={k:comps[k] for k in COMPONENTS}
 if any(q not in source.get('text','') for q in qualifiers):
  result['reasons'].append('QUALIFIER_NOT_IN_SOURCE');result['route']='DEFER'
 if data['semantic_support']=='SUPPORTED' and any(comps[k] != 'MATCH' for k in COMPONENTS):
  result['reasons'].append('SUPPORTED_WITH_UNMATCHED_COMPONENT');result['route']='DEFER'
 result['reasons']=list(dict.fromkeys(result['reasons']))
 return result
