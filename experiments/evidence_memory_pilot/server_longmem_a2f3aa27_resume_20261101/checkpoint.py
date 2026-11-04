import json, os, tempfile
from pathlib import Path

def atomic_json(path, data):
 path=Path(path)
 with tempfile.NamedTemporaryFile(mode='w',encoding='utf8',dir=path.parent,prefix='.tmp_',delete=False) as f:
  tmp=f.name;json.dump(data,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def save_checkpoint(path,chunk_hash,response):
 if not isinstance(response,str) or not response.strip():raise ValueError('empty response')
 atomic_json(path,{'chunk_hash':chunk_hash,'response':response})

def load_checkpoint(path,chunk_hash):
 d=json.loads(Path(path).read_text(encoding='utf8'))
 if d.get('chunk_hash')!=chunk_hash or not isinstance(d.get('response'),str) or not d['response'].strip():raise ValueError('invalid or mismatched checkpoint')
 return d['response']
