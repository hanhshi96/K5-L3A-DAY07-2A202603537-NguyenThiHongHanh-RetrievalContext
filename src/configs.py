"""Cấu hình thử nghiệm. Sửa ở đây, không sửa sau khi đã chốt (results/lock.json)."""

CONFIGS = {
    # 3 phương pháp so sánh
    'dense':   {'method': 'dense',  'top_k': 3, 'filter': 'none', 'neighbors': 0, 'budget_chars': 4000,
                'note': 'Baseline: dense pgvector Day 6 (strategy A), query gốc'},
    'bm25':    {'method': 'bm25',   'top_k': 3, 'filter': 'none', 'neighbors': 0, 'budget_chars': 4000,
                'note': 'Lexical BM25 (k1=1.5, b=0.75), tách theo âm tiết'},
    'hybrid':  {'method': 'hybrid', 'top_k': 3, 'filter': 'none', 'neighbors': 0, 'budget_chars': 4000,
                'note': 'RRF k=60 của dense top-20 + BM25 top-20'},
    # cấu hình cải thiện (mỗi cái có giả thuyết riêng, ghi trong báo cáo)
    'hybrid_neighbors': {'method': 'hybrid', 'top_k': 3, 'filter': 'none', 'neighbors': 1, 'budget_chars': 7000,
                'note': 'Cải thiện 1: hybrid + thêm chunk lân cận ±1 vào context (evidence bị cắt ở ranh giới)'},
    'hybrid_rerank': {'method': 'rerank', 'top_k': 3, 'filter': 'none', 'neighbors': 0, 'budget_chars': 4000,
                'note': 'Cải thiện 2: hybrid top-10 → LLM rerank → top-3 (tăng Hit@1)'},
    'dense_filter': {'method': 'dense', 'top_k': 3, 'filter': 'user', 'neighbors': 0, 'budget_chars': 4000,
                'note': 'Tùy chọn: dense + filter language khi người dùng cung cấp trong câu hỏi'},
}
