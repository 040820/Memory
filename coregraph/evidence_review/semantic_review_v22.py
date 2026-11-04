"""Advisory component-level evidence links; source existence is not proof."""
import json
from . import semantic_review_v21 as base

COMPONENTS=('subject','relation','value')
EXTRA='''For each candidate component, provide component_evidence as an object with subject, relation, value. Each must be {"evidence_id": original ID, "anchor": verbatim words in source, "quote": exact contiguous source passage containing that anchor}. The value anchor must identify the specific person, place or value, not merely an anaphor like "there" or "last time". When a claim depends on pronouns or omitted referents, quote enough context to include both the referent and the phrase that refers to it; do not invent a missing referent. For uncertain or unavailable evidence use an empty object, not guesses. Provide reference_resolution as an array of objects {"expression": exact phrase, "antecedent": exact source words, "evidence_id": original ID, "quote": contiguous passage containing both}; provide [] only if no reference resolution is needed. These links document what you relied on and do not by themselves prove the claim.'''


def review_payload(source,candidate,*,model,max_tokens):
 payload=base.review_payload(source,candidate,model=model,max_tokens=max_tokens)
 payload['messages'][0]['content']+='\n\n'+EXTRA
 return payload


def parse_and_route(raw,source,candidate):
 result=base.parse_and_route(raw,source,candidate)
 result['schema_version']='semantic_review_v2_2';result['component_evidence_check']='UNKNOWN';result['component_evidence_spans']=[]
 if result['route']=='TECHNICAL_ERROR':return result
 try:
  data=json.loads(raw)
  if isinstance(data,list) and len(data)==1:data=data[0]
  for wrapper in ('judgment','fields'):
   if isinstance(data.get(wrapper),dict):
    for key in ('component_evidence','reference_resolution'):
     if key in data[wrapper] and key in data and data[wrapper][key]!=data[key]:raise ValueError('conflicting wrapper')
     if key in data[wrapper]:data[key]=data[wrapper][key]
  components=data.get('component_evidence');refs=data.get('reference_resolution')
 except (TypeError,ValueError,AttributeError):
  result['route']='TECHNICAL_ERROR';result['reasons'].append('INVALID_COMPONENT_SCHEMA');return result
 failures=[];spans=[];text=source.get('text','');eid=source.get('evidence_id')
 for name in COMPONENTS:
  item=components.get(name) if isinstance(components,dict) else None
  if not isinstance(item,dict) or not item:failures.append('COMPONENT_EVIDENCE_MISSING:'+name);continue
  quote=item.get('quote');anchor=item.get('anchor')
  if item.get('evidence_id')!=eid or not isinstance(quote,str) or not quote or quote not in text:
   failures.append('COMPONENT_QUOTE_INVALID:'+name);continue
  if not isinstance(anchor,str) or not anchor or anchor not in quote:
   failures.append('COMPONENT_ANCHOR_NOT_IN_QUOTE:'+name);continue
  if name=='value' and anchor.strip().casefold() in {'there','here','it','that','this','then','last time','那里','这里','它','那个','那时','上次'}:
   failures.append('BARE_ANAPHOR_VALUE_ANCHOR')
  start=0
  while (idx:=text.find(quote,start))>=0:
   spans.append({'component':name,'evidence_id':eid,'anchor':anchor,'quote':quote,'start':idx,'end':idx+len(quote)});start=idx+1
 if not isinstance(refs,list):failures.append('REFERENCE_RESOLUTION_MISSING')
 else:
  for ref in refs:
   if not isinstance(ref,dict) or ref.get('evidence_id')!=eid or not isinstance(ref.get('quote'),str) or not ref['quote'] or ref['quote'] not in text:
    failures.append('REFERENCE_QUOTE_INVALID');continue
   if not isinstance(ref.get('expression'),str) or not ref['expression'] or ref['expression'] not in ref['quote']:failures.append('REFERENCE_EXPRESSION_NOT_IN_QUOTE')
   if not isinstance(ref.get('antecedent'),str) or not ref['antecedent'] or ref['antecedent'] not in ref['quote']:failures.append('REFERENCE_ANTECEDENT_NOT_IN_QUOTE')
 result['component_evidence_spans']=spans
 result['component_evidence_check']='FAIL' if failures else 'PASS'
 result['model_review']['component_evidence']=components
 result['model_review']['reference_resolution']=refs
 if failures:result['route']='DEFER';result['reasons']=list(dict.fromkeys(result['reasons']+failures))
 return result
