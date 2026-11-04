#!/usr/bin/env python3
"""Post-hoc compare controlled predictions with separately stored expectations."""
import argparse,json
from collections import Counter
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--run',required=True,type=Path)
p.add_argument('--expected',type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
rows=[json.loads(x.read_text(encoding='utf8')) for x in sorted((a.run/'cases').glob('*.json'))]
status=Counter(r['result']['status'] for r in rows)
recs=Counter(r['result']['recommendation'] for r in rows)
summary={'case_count':len(rows),'statuses':dict(status),'recommendations':dict(recs),'applied_updates':sum(bool(r['applied']) for r in rows),'limitation':'model reviews are advisory and are not independent ground truth'}
if a.expected:
 expected=json.loads(a.expected.read_text(encoding='utf8'))
 counts={k:0 for k in ('subject_support','temporal_modality','semantic_support')}; details=[];whole=0
 if set(expected)!={r['case_id'] for r in rows}:raise ValueError('expected IDs differ from run')
 for r in rows:
  exp=expected[r['case_id']];pred=r['result'].get('review',{});checks={k:pred.get(k)==v for k,v in exp.items()}
  for k,v in checks.items():counts[k]+=int(v)
  whole+=int(all(checks.values()))
  details.append({'case_id':r['case_id'],'expected':exp,'predicted':{k:pred.get(k) for k in exp},'matches':checks,'recommendation':r['result']['recommendation']})
 summary.update({'expected_labels':'researcher-authored synthetic controls, not human-validated benchmark','per_field_correct':counts,'denominator':len(rows),'all_three_fields_correct':whole,'details':details})
if a.output.exists():raise FileExistsError(a.output)
a.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k!='details'},ensure_ascii=False,indent=2))
