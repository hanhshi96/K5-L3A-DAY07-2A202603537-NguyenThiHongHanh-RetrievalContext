"""Bước 1: kiểm tra kết nối pgvector Day 6 + embedding model, và index khớp chunks.json.

python src/check_connection.py
"""
from common import AGENT_MODEL, EMBED_DIM, EMBED_MODEL, Usage
from retrievers import STRATEGY, TABLE, DenseRetriever, load_chunks


def main():
    chunks = load_chunks()
    d = DenseRetriever()
    info = d.check([c['id'] for c in chunks])
    print(f'Bảng {TABLE}, strategy {STRATEGY}: {info}')
    if not info['match_chunks_json']:
        raise SystemExit('Chunk trong pgvector KHÔNG khớp data/day7/chunks.json — kiểm tra lại Day 6.')
    u = Usage()
    r = d.search('Nghị định 13/2023 có hiệu lực từ ngày nào?', 3, u)
    print(f'Embedding {EMBED_MODEL} ({EMBED_DIM} chiều) OK; agent model: {AGENT_MODEL}')
    for x in r:
        print(f"  {x['rank']}. {x['id']} | {x['source_id']} trang {x['page']} | score {x['score']:.4f}")
    print('Token embed:', u.embed_tokens)


if __name__ == '__main__':
    main()
