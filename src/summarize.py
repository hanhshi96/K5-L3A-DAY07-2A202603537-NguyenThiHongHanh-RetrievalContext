"""Bước 8: in bảng markdown cho báo cáo từ các summary và nhãn đã chấm.

python src/summarize.py
"""
import sys
from collections import Counter

from common import DATA, RESULTS, ROOT, read_json, write_json

sys.path.insert(0, str(ROOT / 'starter'))
from agent_metrics import aggregate  # script tổng hợp của đề; chỉ cộng nhãn bạn đã chấm


def pct(n, N):
    return f'{n}/{N} ({n / N:.0%})' if N else '0/0 (không áp dụng)'


def main():
    print('## Retrieval và context\n')
    print('| Cấu hình | Split | Phương pháp, top-k, filter, neighbors, budget | Hit@1 | Evidence Hit@3 | MRR@3 | Context Evidence Rate | Median ms / query | Token embed / LLM | Chi phí USD |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for p in sorted((RESULTS / 'retrieval').glob('*/summary.json')):
        s = read_json(p)
        print(f"| {s['config']} | {s['split']} | {s['method']}, k={s['top_k']}, filter={s['filter']}, ±{s['neighbors']}, {s['budget_chars']} ký tự "
              f"| {pct(s['hit1'], s['N'])} | {pct(s['hit3'], s['N'])} | {s['mrr3']} | {pct(s['context_evidence'], s['N'])} "
              f"| {s['median_latency_ms']} | {s['embed_tokens']} / {s['prompt_tokens'] + s['completion_tokens']} | {s['cost_usd']} |")

    cases = read_json(DATA / 'cases.json')
    print('\n## Agent\n')
    print('| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 (TB /2) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Context 0/1/2 | Lỗi hệ thống | Median ms | Chi phí USD |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for run in sorted((RESULTS / 'agent').glob('*')):
        if not (run / 'reviews.json').exists():
            print(f'| {run.name} | (chưa chấm) |')
            continue
        traces, reviews, auto = read_json(run / 'traces.json'), read_json(run / 'reviews.json'), read_json(run / 'summary_auto.json')
        split = auto['split']
        sub = [c for c in cases if c['split'] == split]
        try:
            m = aggregate(sub, traces, reviews)
        except ValueError as e:
            print(f'| {run.name} | (chấm chưa xong: {e}) |')
            continue
        write_json(run / 'summary_labels.json', m)
        g = m['grounded_answer']
        ctx = Counter(r['context_grade'] for r in reviews if 'context_grade' in r)
        mean = f"{g['mean']:.2f}" if g['mean'] is not None else '-'
        print(f"| {run.name} | {m['tool_use']['n']}/{m['tool_use']['N']} | {auto['searches_distribution']} "
              f"| {g['0']}/{g['1']}/{g['2']} ({mean}) | {pct(m['citation_validity']['n'], m['citation_validity']['N'])} "
              f"| {pct(m['answer_citation_coverage']['n'], m['answer_citation_coverage']['N'])} "
              f"| {pct(m['abstention']['n'], m['abstention']['N'])} | {pct(m['false_abstention']['n'], m['false_abstention']['N'])} "
              f"| {ctx[0]}/{ctx[1]}/{ctx[2]} | {m['provider_errors']} | {auto['median_latency_ms']} | {auto['cost_usd']} |")

    print('\n## Từng câu (để chép vào bảng Query ID trong báo cáo)\n')
    print('| Run | Query ID | Loại câu | Tool call | Câu trả lời và citation | Nhãn, lý do |')
    print('|---|---|---|---|---|---|')
    for run in sorted((RESULTS / 'agent').glob('*_test')):
        if not (run / 'reviews.json').exists():
            continue
        rv = {r['query_id']: r for r in read_json(run / 'reviews.json')}
        for t in read_json(run / 'traces.json'):
            r = rv.get(t['query_id'], {})
            calls = '; '.join(f"search(\"{e['query']}\")" for e in t['events'] if e['event'] == 'search')
            label = (f"answer {r.get('answer_grade')}, context {r.get('context_grade')}" if t['answerable']
                     else f"abstention_correct={r.get('abstention_correct')}")
            ans = (t['answer'] or t.get('error') or '').replace('|', '/').replace('\n', ' ')[:200]
            print(f"| {run.name} | {t['query_id']} | {'có đáp án' if t['answerable'] else 'thiếu evidence'} | {calls} "
                  f"| {ans} | {label}; {r.get('reason', '')} |")


if __name__ == '__main__':
    main()
