"""Bước 7 (chấm): tự chấm câu trả lời agent. Lưu ngay sau mỗi câu; gõ q để dừng, chạy lại để tiếp tục.

python src/review_agent.py --run dense_test
"""
import argparse
import sys

from common import DATA, RESULTS, read_json, write_json


def ask(prompt, valid):
    while True:
        a = input(prompt).strip().lower()
        if a == 'q':
            sys.exit('Đã lưu. Chạy lại để tiếp tục.')
        if a in valid:
            return a


def reason():
    while not (r := input('Lý do: ').strip()):
        print('Lý do là bắt buộc.')
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='tên thư mục trong results/agent, vd dense_test')
    a = ap.parse_args()
    run = RESULTS / 'agent' / a.run
    traces = read_json(run / 'traces.json')
    bench = {q['id']: q for q in read_json(DATA / 'benchmark.json')}
    path = run / 'reviews.json'
    reviews = read_json(path) if path.exists() else []
    done = {r['query_id'] for r in reviews}
    todo = [t for t in traces if t['status'] != 'error' and t['query_id'] not in done]
    print(f'Còn {len(todo)} câu cần chấm (lỗi hệ thống không cần nhãn).')
    for n, t in enumerate(todo, 1):
        q = bench[t['query_id']]
        print('\n' + '=' * 90)
        print(f"({n}/{len(todo)}) {t['query_id']} — {'CÓ ĐÁP ÁN' if q['answerable'] else 'THIẾU EVIDENCE'}")
        print('CÂU HỎI:', q['question'])
        if q['answerable']:
            print('ĐÁP ÁN CHUẨN:', q['expected_answer'])
            print('EVIDENCE GOLD:', [(g['source_id'], g['page'], g['evidence']) for g in q['gold']])
        else:
            print('LÝ DO THIẾU:', q['absence_reason'])
        print('\nTOOL CALLS:')
        for e in t['events']:
            if e['event'] == 'search':
                print(f"  search('{e['query']}') -> {[(r['id'][-8:], r['source_id'], r['page']) for r in e['results']]}")
            else:
                print('  ', e)
        print('\nCONTEXT CUỐI GỬI MODEL:\n' + '-' * 90 + '\n' + (t['context'] or '(rỗng)') + '\n' + '-' * 90)
        print(f"STATUS: {t['status']}\nCÂU TRẢ LỜI: {t['answer']}")
        if t.get('raw_answer') and t['raw_answer'] != t['answer']:
            print('(câu trả lời gốc của model trước khi bị chuyển thành từ chối:', t['raw_answer'], ')')
        print('CITATIONS:', t['citations'])

        r = {'query_id': t['query_id']}
        if q['answerable']:
            r['context_grade'] = int(ask('Context 0=không có evidence | 1=có nhưng thiếu điều kiện/nguồn | 2=đủ: ', {'0', '1', '2'}))
            if t['status'] == 'abstain':
                print('Agent từ chối câu có đáp án -> answer_grade = 0 (theo BENCHMARK).')
                r['answer_grade'] = 0
            else:
                r['answer_grade'] = int(ask('Câu trả lời 0=sai/thiếu evidence | 1=đúng một phần | 2=đúng và đủ: ', {'0', '1', '2'}))
        else:
            r['abstention_correct'] = ask('Agent từ chối đúng, không thêm khẳng định vô căn cứ? (y/n): ', {'y', 'n'}) == 'y'
        r['citation_support'] = []
        for c in t['citations']:
            r['citation_support'].append(ask(f'Citation [{c}] có trong context VÀ hỗ trợ câu trả lời? (y/n): ', {'y', 'n'}) == 'y')
        r['reason'] = reason()
        reviews.append(r)
        write_json(path, reviews)
    print('\nXONG. Tổng hợp:  python src/summarize.py')


if __name__ == '__main__':
    main()
