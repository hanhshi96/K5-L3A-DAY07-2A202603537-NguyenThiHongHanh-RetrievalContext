"""Bước 6: chốt cấu hình TRƯỚC khi chạy test.

python src/lock.py --baseline dense --chosen hybrid_neighbors --criteria "Ưu tiên ..."
"""
import argparse
from datetime import datetime, timezone

from common import RESULTS, read_json, write_json
from configs import CONFIGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, choices=CONFIGS)
    ap.add_argument('--chosen', required=True, choices=CONFIGS)
    ap.add_argument('--criteria', required=True)
    a = ap.parse_args()
    lock = RESULTS / 'lock.json'
    if lock.exists():
        raise SystemExit('Đã chốt rồi (results/lock.json). Không đổi cấu hình sau khi chốt.')
    dev = {}
    for p in sorted((RESULTS / 'retrieval').glob('*_dev/summary.json')):
        dev[p.parent.name] = read_json(p)
    write_json(lock, {'locked_at': datetime.now(timezone.utc).isoformat(), 'baseline': a.baseline,
                      'chosen': a.chosen, 'criteria': a.criteria,
                      'configs': {a.baseline: CONFIGS[a.baseline], a.chosen: CONFIGS[a.chosen]},
                      'dev_retrieval_summaries': dev})
    print('Đã chốt. Commit results/lock.json lên git ngay bây giờ, rồi mới chạy test.')


if __name__ == '__main__':
    main()
