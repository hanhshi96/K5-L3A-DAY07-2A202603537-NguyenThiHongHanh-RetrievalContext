"""Bước 2: tạo data/day7/chunks.json và data/day7/benchmark.json từ đầu ra Day 6.

python src/prepare_data.py --day6 <đường_dẫn_tới_thư_mục_starter_Day6>
"""
import argparse
from pathlib import Path

from common import DATA, chunk_hit, norm, read_json, write_json


def to_chunk(c):
    return {'id': c['chunk_id'], 'text': c['text'], 'source_id': c['document_id'],
            'source': c['source'], 'page': int(c['page']), 'metadata': c.get('metadata', {})}


def gold_of(q):
    gold = [{'source_id': q['gold_source_id'], 'page': int(q['gold_page']), 'evidence': q['evidence_quote']}]
    for a in q.get('alternative_gold') or []:
        gold.append({'source_id': a.get('source_id') or a.get('gold_source_id'),
                     'page': int(a.get('page') or a.get('gold_page')),
                     'evidence': a.get('evidence_quote') or a.get('evidence')})
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day6', required=True, type=Path, help='thư mục starter/ của repo Day 6')
    args = ap.parse_args()
    d6 = args.day6
    chunks = [to_chunk(c) for c in read_json(d6 / 'results/lab06_api/chunks_A.json')]
    pages = {(p['document_id'], int(p['page'])): p['text'] for p in read_json(d6 / 'results/lab06_api/pages.json')}
    if len({c['id'] for c in chunks}) != len(chunks):
        raise ValueError('chunk ID trùng')

    questions = []
    # Day 6 questions: all become DEV (their old test was already seen in Day 6).
    for q in read_json(d6 / 'benchmark.json'):
        questions.append({'id': q['id'], 'split': 'dev', 'question': q['question'], 'answerable': True,
                          'expected_answer': q['expected_answer'], 'gold': gold_of(q),
                          'filter': q.get('filter') or {}, 'origin': f"day6-{q['split']}"})
    for q in read_json(DATA / 'new_questions.json'):
        q = dict(q)
        q.setdefault('filter', {})
        q['origin'] = 'day7-new'
        questions.append(q)

    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('question ID trùng')
    problems = []
    for q in questions:
        if q['answerable']:
            for g in q['gold']:
                if norm(g['evidence']) not in norm(pages.get((g['source_id'], g['page']), '')):
                    raise ValueError(f"{q['id']}: evidence KHÔNG có ở {g['source_id']} trang {g['page']}")
            q['evidence_in_single_chunk'] = any(chunk_hit(c, q['gold']) for c in chunks)
            if not q['evidence_in_single_chunk']:
                problems.append(q['id'])
        else:
            for k in ('expected_behavior', 'absence_reason'):
                if not q.get(k):
                    raise ValueError(f"{q['id']}: câu thiếu evidence cần trường {k}")

    count = lambda s, a: sum(q['split'] == s and q['answerable'] == a for q in questions)
    print(f"dev: {count('dev', True)} có đáp án + {count('dev', False)} thiếu evidence")
    print(f"test: {count('test', True)} có đáp án + {count('test', False)} thiếu evidence")
    if count('dev', True) < 15 or count('dev', False) < 2 or count('test', True) < 5 or count('test', False) < 2:
        raise ValueError('Chưa đủ số câu tối thiểu theo BENCHMARK.md')
    if problems:
        print('Lưu ý: evidence bị tách qua 2 chunk (không thể Hit trong 1 chunk):', problems)
    write_json(DATA / 'chunks.json', chunks)
    write_json(DATA / 'benchmark.json', questions)
    write_json(DATA / 'cases.json', [{'id': q['id'], 'split': q['split'], 'answerable': q['answerable']} for q in questions])
    print(f'Đã ghi {len(chunks)} chunk và {len(questions)} câu vào {DATA}')


if __name__ == '__main__':
    main()
