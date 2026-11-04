"""Prepare optional authentication without adding secrets to saved requests."""
from pathlib import Path


def request_settings(config, payload):
    headers = {}
    body = dict(payload)
    key_file = config.get('api_key_file')
    if key_file:
        key = Path(key_file).expanduser().read_text(encoding='utf8').strip()
        if not key: raise ValueError('API key file is empty')
        headers['Authorization'] = 'Bearer ' + key
    mode = config.get('thinking_parameter', 'vllm')
    if mode == 'top_level':
        body.pop('chat_template_kwargs', None)
        body['enable_thinking'] = False
    elif mode != 'vllm':
        raise ValueError('unknown thinking parameter mode')
    return headers, body
