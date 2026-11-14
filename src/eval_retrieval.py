"""Bước 3-5: chạy một cấu hình retrieval trên một split, lưu trace + metrics.

python src/eval_retrieval.py --config dense --split dev
"""
import argparse
import statistics
import time

from common import RESULTS, DATA, Usage, chunk_hit, context_hit, pack_context, read_json, write_json
from configs import CONFIGS
from retrievers import build, load_chunks, with_neighbors


def filters_for(q, cfg):
    # Chỉ dùng filter người dùng cung cấp sẵn trong câu hỏi; không bao giờ suy từ gold.
    return (q.get('filter') or None) if cfg['filter'] == 'user' else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=CONFIGS)
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    if args.split == 'test':
        lock = RESULTS / 'lock.json'
        if not lock.exists():
            raise SystemExit('Chưa chốt cấu hình: chạy src/lock.py trước khi đụng tới test.')
        if args.config not in (read_json(lock)['baseline'], read_json(lock)['chosen']):
            raise SystemExit('Trên test chỉ chạy baseline và cấu hình đã chốt.')

    chunks = load_chunks()
    questions = [q for q in read_json(DATA / 'benchmark.json') if q['split'] == args.split and q['answerable']]
    retriever = build(cfg['method'], chunks)
    usage, traces = Usage(), []
    for q in questions:
        t0 = time.perf_counter()
        error = None
        try:
            results = retriever.search(q['question'], max(3, cfg['top_k']), usage, filters_for(q, cfg))
        except Exception as e:  # ghi lỗi, không thay bằng kết quả giả
            results, error = [], f'{type(e).__name__}: {e}'
        top = results[:cfg['top_k']]
        cand = with_neighbors(top, chunks, cfg['neighbors']) if cfg['neighbors'] else top
        context, kept = pack_context(cand, cfg['budget_chars'])
        ms = (time.perf_counter() - t0) * 1000
        top3 = [{'id': r['id'], 'rank': r['rank'], 'score': round(r['score'], 5), 'source_id': r['source_id'],
                 'page': r['page'], 'hit': chunk_hit(r, q['gold']), 'text': r['text']} for r in results[:3]]
        first = next((i + 1 for i, r in enumerate(top3) if r['hit']), None)
        traces.append({'query_id': q['id'], 'split': q['split'], 'question': q['question'], 'config': args.config,
                       'filter_used': filters_for(q, cfg), 'gold': q['gold'], 'top3': top3,
                       'hit1': bool(top3 and top3[0]['hit']), 'hit3': first is not None,
                       'rr': 1 / first if first else 0.0,
                       'context_ids': [c['id'] for c in kept], 'context_chars': len(context),
                       'context_hit': context_hit(kept, q['gold']), 'context': context,
                       'latency_ms': round(ms, 1), 'error': error})
        print(f"{q['id']}: hit1={traces[-1]['hit1']} hit3={traces[-1]['hit3']} ctx={traces[-1]['context_hit']}"
              + (f' ERROR {error}' if error else ''), flush=True)

    N = len(traces)
    s = {'config': args.config, 'split': args.split, **cfg, 'N': N,
         'hit1': sum(t['hit1'] for t in traces), 'hit3': sum(t['hit3'] for t in traces),
         'mrr3': round(sum(t['rr'] for t in traces) / N, 4),
         'context_evidence': sum(t['context_hit'] for t in traces),
         'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces)),
         'median_latency_ms': round(statistics.median(t['latency_ms'] for t in traces), 1),
         'errors': sum(bool(t['error']) for t in traces), **usage.as_dict()}
    out = RESULTS / 'retrieval' / f'{args.config}_{args.split}'
    write_json(out / 'traces.json', traces)
    write_json(out / 'summary.json', s)
    print(f"\n{args.config}/{args.split}: Hit@1 {s['hit1']}/{N} · Hit@3 {s['hit3']}/{N} · "
          f"Context {s['context_evidence']}/{N} · MRR@3 {s['mrr3']} · median {s['median_latency_ms']} ms · "
          f"${s['cost_usd']} · lỗi {s['errors']}")


if __name__ == '__main__':
    main()
