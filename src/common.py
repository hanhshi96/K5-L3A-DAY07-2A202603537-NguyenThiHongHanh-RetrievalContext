"""Shared helpers for Day 7: env, OpenAI calls (stdlib), normalization, hit checks, context packing."""
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'day7'
RESULTS = ROOT / 'results'


def load_env():
    """Read KEY=VALUE lines from ROOT/.env without extra dependencies."""
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()
EMBED_MODEL = os.getenv('EMBEDDING_MODEL') or 'text-embedding-3-small'
EMBED_DIM = int(os.getenv('EMBEDDING_DIM') or 1536)
AGENT_MODEL = os.getenv('AGENT_MODEL') or 'gpt-4o-mini'
# USD per 1M tokens. Check the provider's current price list and edit .env if different.
PRICE_EMBED = float(os.getenv('PRICE_EMBED_PER_1M') or 0.02)
PRICE_IN = float(os.getenv('PRICE_INPUT_PER_1M') or 0.15)
PRICE_OUT = float(os.getenv('PRICE_OUTPUT_PER_1M') or 0.60)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def norm(text):
    """Same rule as Day 6: NFC, casefold, remove all whitespace (PDF text splits Vietnamese syllables)."""
    return re.sub(r'\s+', '', unicodedata.normalize('NFC', text or '')).casefold()


def evidence_in(text, gold):
    return any(norm(g['evidence']) in norm(text) for g in gold)


def chunk_hit(chunk, gold):
    """Hit = right source and page, and the whole evidence inside this one chunk."""
    return any(chunk['source_id'] == g['source_id'] and int(chunk['page']) == int(g['page'])
               and norm(g['evidence']) in norm(chunk['text']) for g in gold)


def context_hit(context_chunks, gold):
    """Evidence fully present in the final context: in one chunk, or across two
    consecutive chunks of the same page (overlap removed using start/end_char)."""
    if any(chunk_hit(c, gold) for c in context_chunks):
        return True
    by_page = {}
    for c in context_chunks:
        by_page.setdefault((c['source_id'], int(c['page'])), []).append(c)
    for (sid, page), group in by_page.items():
        group = sorted(group, key=lambda c: c['metadata'].get('start_char', 0))
        for a, b in zip(group, group[1:]):
            a_end, b_start = a['metadata'].get('end_char'), b['metadata'].get('start_char')
            if a_end is None or b_start is None or b_start > a_end:
                continue
            merged = {'source_id': sid, 'page': page, 'text': a['text'] + b['text'][a_end - b_start:]}
            if chunk_hit(merged, gold):
                return True
    return False


def pack_context(chunks, budget_chars):
    """Keep chunks in the given order, skip duplicates, stop adding when the budget is exceeded."""
    blocks, kept, seen = [], [], set()
    for c in chunks:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        block = f"[{c['id']}] (nguồn: {c['source_id']}, trang {c['page']})\n{c['text']}"
        if len('\n\n'.join(blocks + [block])) > budget_chars:
            continue
        blocks.append(block)
        kept.append(c)
    return '\n\n'.join(blocks), kept


class Usage:
    """Token and cost tracker for one run."""
    def __init__(self):
        self.embed_tokens = self.prompt_tokens = self.completion_tokens = 0

    def cost(self):
        return (self.embed_tokens * PRICE_EMBED + self.prompt_tokens * PRICE_IN
                + self.completion_tokens * PRICE_OUT) / 1_000_000

    def as_dict(self):
        return {'embed_tokens': self.embed_tokens, 'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens, 'cost_usd': round(self.cost(), 6)}


def _post(url, payload):
    key = os.getenv('OPENAI_API_KEY', '').strip()
    if not key:
        raise RuntimeError('Thiếu OPENAI_API_KEY trong .env')
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), method='POST',
                                 headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', 'replace')[:500]
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f'OpenAI HTTP {e.code}: {body}') from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:  # gồm cả read timeout
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f'Lỗi mạng OpenAI sau 3 lần thử: {type(e).__name__}: {e}') from None


def embed_query(text, usage):
    data = _post('https://api.openai.com/v1/embeddings',
                 {'model': EMBED_MODEL, 'input': [text], 'dimensions': EMBED_DIM, 'encoding_format': 'float'})
    usage.embed_tokens += data.get('usage', {}).get('total_tokens', 0)
    vec = data['data'][0]['embedding']
    if len(vec) != EMBED_DIM:
        raise ValueError(f'Embedding {len(vec)} chiều, index cần {EMBED_DIM}')
    return vec


def chat(messages, usage, *, tools=None, tool_choice=None, json_mode=False, model=None):
    payload = {'model': model or AGENT_MODEL, 'messages': messages, 'temperature': 0}
    if tools:
        payload['tools'] = tools
        payload['tool_choice'] = tool_choice or 'auto'
    if json_mode:
        payload['response_format'] = {'type': 'json_object'}
    data = _post('https://api.openai.com/v1/chat/completions', payload)
    u = data.get('usage', {})
    usage.prompt_tokens += u.get('prompt_tokens', 0)
    usage.completion_tokens += u.get('completion_tokens', 0)
    return data['choices'][0]['message']
