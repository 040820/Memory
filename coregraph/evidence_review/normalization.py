"""Versioned lexical candidate-key proposal; never overwrites source facts."""
import re

PLATFORMS={'instagram','twitter','facebook','linkedin','tiktok','youtube','x'}
VERSION='lexical_v1_dev'

def normalize_attribute(raw,object_key):
 if not isinstance(raw,str) or not raw.strip() or raw.strip().upper()=='UNKNOWN':return {'raw':raw,'key':'UNKNOWN','rules':[],'status':'candidate_only','version':VERSION}
 key=re.sub(r'[^a-z0-9]+','_',raw.strip().lower()).strip('_');rules=[]
 tokens=key.split('_')
 if tokens and tokens[-1] in ('estimate','estimated','approximation'):
  tokens=tokens[:-1];rules.append('remove_estimate_qualifier')
 if len(tokens)>=3 and tokens[:2]==['social','media'] and isinstance(object_key,str) and object_key.lower() in PLATFORMS:
  tokens=tokens[2:];rules.append('remove_social_media_prefix_with_platform')
 if 'followers' in tokens:
  tokens=['follower' if x=='followers' else x for x in tokens];rules.append('singularize_followers')
 key='_'.join(tokens) or 'UNKNOWN'
 return {'raw':raw,'key':key,'rules':rules,'status':'candidate_only','version':VERSION}
