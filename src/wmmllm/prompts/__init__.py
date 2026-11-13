"""Exact packaged prompt bytes and JSON data."""
from __future__ import annotations
import hashlib,json
from functools import lru_cache
from importlib.resources import files
class PromptChanged(RuntimeError):pass
@lru_cache(maxsize=1)
def manifest():return json.loads(files(__package__).joinpath('PROMPTS.json').read_text(encoding='utf-8'))
def _bytes(name):
    if name not in manifest():raise ValueError('Unknown prompt resource: '+str(name))
    data=files(__package__).joinpath(name).read_bytes()
    if hashlib.sha256(data).hexdigest()!=manifest()[name]['sha256']:raise PromptChanged('Prompt data changed: '+name)
    return data
def read_text(name):return _bytes(name).decode('utf-8')
def read_json_data(name):return json.loads(_bytes(name))
