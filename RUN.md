# Bài làm Day 7 — cách chạy

Chạy từ thư mục gốc repo Day 7, Python 3.11+, venv có `psycopg[binary]` (dùng lại venv Day 6 hoặc `pip install "psycopg[binary]"`).
Database pgvector của Day 6 phải đang chạy (`cd <repo Day 6>/starter && docker compose up -d`).

```bash
cp .env.example .env                                   # điền OPENAI_API_KEY
python src/prepare_data.py --day6 <repo Day 6>/starter # tạo data/day7/chunks.json, benchmark.json, cases.json
python src/check_connection.py                         # kiểm tra pgvector + embedding

# Retrieval trên dev (3 phương pháp + 2 cải thiện)
for c in dense bm25 hybrid hybrid_neighbors hybrid_rerank; do python src/eval_retrieval.py --config $c --split dev; done
# Agent trên dev (baseline + ứng viên)
python src/agent.py --config dense --split dev
python src/agent.py --config <ung_vien> --split dev
# Chốt rồi mới chạy test
python src/lock.py --baseline dense --chosen <ung_vien> --criteria "<tiêu chí>"
python src/eval_retrieval.py --config dense --split test
python src/eval_retrieval.py --config <ung_vien> --split test
python src/agent.py --config dense --split test
python src/agent.py --config <ung_vien> --split test
# Tự chấm và tổng hợp
python src/review_agent.py --run dense_test
python src/review_agent.py --run <ung_vien>_test
python src/summarize.py
```

Kết quả: `results/retrieval/<config>_<split>/` (traces.json, summary.json), `results/agent/<config>_<split>/` (traces.json, summary_auto.json, reviews.json, summary_labels.json), `results/lock.json`.
