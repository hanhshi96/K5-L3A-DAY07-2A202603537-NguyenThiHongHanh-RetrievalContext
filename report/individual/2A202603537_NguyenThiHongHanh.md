# Báo cáo Day 7 — Agentic RAG

**Học viên:** Nguyễn Thị Hồng Hạnh · **MSSV:** 2A202603537 · **Lớp:** L3A
**Baseline:** dense (pgvector Day 6) · **Cấu hình chốt trên dev:** `hybrid_neighbors` (hybrid RRF + chunk lân cận ±1)

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** dùng lại corpus Day 6: Nghị định 13/2023/NĐ-CP (tiếng Việt, 29 trang nội dung; văn bản quy phạm pháp luật, không thuộc đối tượng bảo hộ quyền tác giả) và *Attention Is All You Need* (arXiv 1706.03762, tiếng Anh, 9 trang; dùng cho mục đích học thuật, có ghi nguồn, không phân phối lại PDF). Tổng 38 trang nội dung, **121 chunk** của cấu hình A Day 6 (cửa sổ 900 ký tự, overlap 150), ID ổn định dạng `<doc>-p<trang>-fixed-<hash>`. File chunk: `data/day7/chunks.json` (xuất từ `chunks_A.json` của Day 6).
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL local + **pgvector 0.8.7**, bảng Day 6 `lab06_api_chunks`, `strategy='A'`, cosine (`<=>`), exact search. Embedding OpenAI **`text-embedding-3-small`, 1536 chiều**, dùng cùng model cho query và document như Day 6 nên **không tạo lại index**. `src/check_connection.py` xác nhận 121 dòng trong pgvector khớp hoàn toàn `chunks.json`.
- **Agent model, cách chạy và phân chia dev/test:** OpenAI **`gpt-4o-mini`**, temperature 0, tool calling (`src/agent.py`). Benchmark (`data/day7/benchmark.json`, tạo bằng `src/prepare_data.py`):

  | Split | Có đáp án | Thiếu evidence | Ghi chú |
  |---|---:|---:|---|
  | dev | 20 (Q01–Q20) | 2 (U01, U02) | Q01–Q20 là toàn bộ benchmark Day 6; 5 câu test cũ của Day 6 (Q08–Q10, Q19, Q20) đã được xem kết quả ở Day 6 nên **chuyển sang dev** |
  | test | 5 (T01–T05) | 2 (U03, U04) | **Câu mới viết cho Day 7**, chốt và commit trước mọi thử nghiệm |

  Benchmark và code được commit trước khi chạy retrieval; cấu hình được chốt bằng `results/lock.json` (commit `a9d84f2`) trước khi chạy test.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** với câu có đáp án, ghi `gold = {source_id, page, evidence}`; `prepare_data.py` kiểm tra evidence nằm ở đúng trang sau chuẩn hóa (NFC, bỏ khoảng trắng do PDF tách âm tiết tiếng Việt) và nằm trọn trong một chunk; tác giả đối chiếu thêm với PDF. Câu thiếu evidence là câu **gần chủ đề**, ghi `absence_reason` và cách kiểm tra bằng tìm từ khóa trên toàn bộ 38 trang (vd. U01: chỉ Điều 4 trang 5 nhắc "xử phạt… theo quy định", không có mức tiền; U03: "phí" chỉ xuất hiện ở trang 13 (chi phí in, sao, chụp), 25, 27 (kinh phí); U04: không có "Chinese"). Câu thiếu evidence không vào mẫu số của retrieval metrics. Hit = chunk đúng nguồn/trang và chứa trọn evidence; Context Evidence Rate = evidence trọn vẹn trong context cuối (trong một chunk, hoặc nối hai chunk liền nhau cùng trang).

## Retrieval và thử nghiệm

Cùng corpus, index, câu hỏi và nhãn; query gốc (không viết lại) cho bảng này; không dùng filter. Thời gian là median mỗi truy vấn (ms); chi phí USD cho toàn split theo giá cấu hình trong `.env` (embedding 0,02 USD/1M token; gpt-4o-mini 0,15/0,60 USD/1M token vào/ra).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline `dense` | Dense pgvector Day 6, k=3, không filter, không rerank, 4000 ký tự | Baseline bắt buộc | dev | 9/20 (45%) | 16/20 (80%) | 16/20 (80%) | 447 ms · 0,000014 USD |
| Phương pháp 2 `bm25` | BM25 (k1=1,5, b=0,75), tách theo âm tiết `\w+` sau NFC, k=3, 4000 ký tự | Khớp từ khóa đặc thù (số liệu, thuật ngữ luật) mà dense bỏ sót | dev | 13/20 (65%) | 16/20 (80%) | 16/20 (80%) | 0,3 ms · 0 USD |
| Phương pháp 3 `hybrid` | RRF (k=60) của dense top-20 + BM25 top-20, k=3, 4000 ký tự | Dense và lexical bù cho nhau | dev | 13/20 (65%) | 18/20 (90%) | 18/20 (90%) | 521 ms · 0,000014 USD |
| Cải thiện 1 `hybrid_neighbors` | Hybrid k=3 + thêm chunk liền trước/sau (±1) vào context, 7000 ký tự | Evidence bị cắt ở ranh giới chunk sẽ được bổ sung từ chunk lân cận | dev | 13/20 (65%) | 18/20 (90%) | **20/20 (100%)** | 501 ms · 0,000014 USD |
| Cải thiện 2 `hybrid_rerank` | Hybrid top-10 → gpt-4o-mini xếp lại → top-3, 4000 ký tự | Model xếp lại sẽ đưa chunk chứa evidence lên top-1 | dev | **18/20 (90%)** | **19/20 (95%)** | 19/20 (95%) | 4 788 ms · 0,0115 USD (62 946 token LLM) |
| **Cấu hình đã chọn** `hybrid_neighbors` | như Cải thiện 1 | — | test | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | 619 ms · 0,000004 USD |
| Baseline `dense` | như Baseline | — | test | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | 524 ms · 0,000004 USD |

`hybrid_neighbors` vừa là cấu hình cải thiện vừa dùng phương pháp hybrid; thay đổi được đo riêng so với `hybrid` (chỉ khác bước mở rộng context). MRR@3 trên dev: dense 0,62 · bm25 0,70 · hybrid 0,77 · hybrid_neighbors 0,77 · hybrid_rerank 0,93; trên test: dense 0,77 · hybrid_neighbors 0,80.

**Hai lỗi dev, nguyên nhân, cách thử:**

1. **Q04, Q05 — dense không có evidence trong top-3 vì thiếu khớp từ khóa.** Q04 ("Quyền truy cập… ngoại lệ gì?") dense trả về trang 1 (phần đầu văn bản), 7, 9 nhưng sai đoạn; Q05 ("hạn chế xử lý… thời hạn nào") dense trả về trang 11, 15, 18, gold ở trang 8. *Nguyên nhân:* embedding bắt chủ đề chung "bảo vệ dữ liệu cá nhân" nhưng không bám cụm đặc thù "quyền truy cập", "hạn chế xử lý", "72 giờ"; text PDF bị tách âm tiết ("truy c ập") làm biểu diễn chunk kém. *Giả thuyết:* thêm lexical qua hybrid RRF sẽ kéo đúng đoạn lên. *Kết quả:* BM25 trúng cả hai (Hit@1); hybrid sửa Q04 (Hit@1) và Q05 (Hit@3); rerank đưa Q05 lên top-1.
2. **Q03, Q06 — đúng điều luật nhưng evidence nằm ở chunk kế tiếp.** Q03 dense lấy được chunk đầu Điều 3 trang 5 ("Nguyên tắc… 1. Dữ liệu cá nhân được xử lý…") nhưng evidence là khoản 7 ở chunk sau; dense, hybrid và cả rerank đều trượt (chunk evidence không có trong danh sách ứng viên). *Nguyên nhân:* chunk cố định 900 ký tự cắt ngang danh sách khoản; tiêu đề (khớp nghĩa với câu hỏi) và nội dung trả lời nằm ở hai chunk. *Giả thuyết:* thêm chunk lân cận ±1 vào context. *Kết quả:* `hybrid_neighbors` đưa Q03 và Q06 vào context (Context Evidence Rate 18/20 → 20/20) dù Hit@3 không đổi; đánh đổi là context dài gần gấp đôi.

Quan sát phụ: ở Q19, Q20 top-1 của dense là bảng số Table 3 (trang 9) — chunk toàn số/thuật ngữ mô hình "giống" nhiều câu hỏi, làm Hit@1 dense thấp.

**Lý do chọn cấu hình** (tiêu chí ghi trong `results/lock.json`, chốt sau dev và trước test): (1) abstention đúng, không trả lời bịa câu thiếu evidence; (2) Context Evidence Rate; (3) citation trúng chunk gold (chỉ số tham khảo tự động); (4) thời gian/chi phí. Trên dev với agent v2: `hybrid_neighbors` abstention 2/2, context 20/20, cite trúng gold 16/20, 2,4 s/câu, 0,010 USD/22 câu; `dense` 2/2, 17/20, 17/20, 2,5 s, 0,008 USD; `hybrid_rerank` 1/2 (trả lời bịa U02), 19/20, 18/20, 7,2 s, 0,020 USD → loại vì vi phạm tiêu chí (1) và chậm gấp 3.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** hai tool: `search_documents(query)` (retriever theo cấu hình, top-3, ± chunk lân cận) và `finish(status, answer, citations)` với `status ∈ {answered, insufficient_evidence}`. **Lượt tìm đầu luôn dùng nguyên câu hỏi gốc**; sau đó model tự quyết định tìm thêm (tự viết query) hay kết thúc; tối đa **3 lượt tìm**/6 bước, đến giới hạn thì buộc gọi `finish` (tránh vòng lặp). Context: cộng dồn qua các lượt, **loại trùng theo chunk ID**, giữ thứ tự rank, cắt theo budget ký tự (4000 hoặc 7000), mỗi đoạn kèm `[chunk_id] (nguồn, trang)`. Prompt yêu cầu chỉ dùng tài liệu, coi nội dung tài liệu là dữ liệu chứ không phải chỉ dẫn, trả lời có `[chunk_id]` hoặc báo không đủ bằng chứng. Hậu kiểm: citation phải nằm trong context; ID bị chép thiếu đuôi chỉ được sửa khi khớp **duy nhất** một ID trong context; thiếu `[id]` trong câu trả lời thì nối vào cuối — mọi chỉnh sửa ghi thành event (`citation_id_repaired`, `citation_markers_appended`); citation ngoài context → chuyển thành từ chối. Lỗi model/vector store ghi `status: error`, không thay kết quả giả; gọi API tự thử lại 3 lần khi 429/5xx/timeout.

**Agent v1 → v2 trên dev (trước/sau, chỉ số tự động — dev không chấm tay):**

| Cấu hình | v1: câu có đáp án được trả lời | v2: câu có đáp án được trả lời | Context Evidence (agent) v1 → v2 | Abstention U01–U02 v1 → v2 | Median ms v1 → v2 | Chi phí v1 → v2 (USD, 22 câu) |
|---|---|---|---|---|---|---|
| dense | 7/20 (+1 lỗi timeout) | 19/20 | 12/19 → 17/20 | 2/2 → 2/2 | 3 749 → 2 535 | 0,0084 → 0,0079 |
| hybrid_neighbors | 8/20 | 20/20 | 13/20 → 20/20 | 2/2 → 2/2 | 3 828 → 2 398 | 0,0130 → 0,0103 |
| hybrid_rerank | 7/20 | 20/20 | 15/20 → 19/20 | 2/2 → **1/2** | 7 993 → 7 217 | 0,0215 → 0,0205 |

Nguyên nhân v1 kém (đọc trace): (a) model **trả lời đúng nhưng không chèn `[chunk_id]`** vào câu trả lời (vd. Q10 "01/7/2023", Q18 "4000 steps"), có lần chép thiếu đuôi ID → bộ kiểm tra citation chuyển thành từ chối; (b) model **tự rút gọn query** làm mất ngữ cảnh (Q20: "base models … checkpoints averaged written" → kết luận sai là không có thông tin); (c) `TimeoutError` chưa được thử lại. v2 sửa cả ba (lượt đầu dùng query gốc, chuẩn hóa citation có ghi event, retry timeout).

**Kết quả trên test (đã chốt, chấm tay):**

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| `dense` / test | 7/7 | 1 lượt: 6 câu · 3 lượt: 1 câu | 0/1/4 — 1,80 | 5/5 (100%) | 5/5 (100%) | 2/2 (100%) | 0/5 (0%) |
| `hybrid_neighbors` / test | 7/7 | 1 lượt: 6 câu · 2 lượt: 1 câu | 0/1/4 — 1,80 | 5/6 (83%) | 6/6 (100%) | **1/2 (50%)** | 0/5 (0%) |

Context (chấm tay, 5 câu test có đáp án): cả hai cấu hình 0/0/5 (mức 2). Lỗi hệ thống: 0. Median 2 596 ms (dense) và 2 275 ms (hybrid_neighbors); chi phí 7 câu 0,0024 và 0,0033 USD.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| T01 | có đáp án | 1 lượt, query gốc; context chứa Điều 14 khoản 3 trang 12 | "72 giờ sau khi nhận được yêu cầu… trừ trường hợp pháp luật có quy định khác" `[nd13_2023-p0012-fixed-ed68943b32058d19]` | Cả hai cấu hình: answer 2, context 2 — đủ thời hạn và ngoại lệ; citation đúng đoạn |
| T02 | có đáp án | 1 lượt; context trang 16 Điều 19 | "vợ, chồng hoặc con thành niên; nếu không có thì cha, mẹ" `[nd13_2023-p0016-fixed-4194921a5bbaa069]` | answer 1, context 2 — đúng thứ tự người đồng ý nhưng **thiếu ngoại lệ Điều 17, 18** |
| T03 | có đáp án | 1 lượt; trang 25 Điều 31 | "ngân sách nhà nước… dự toán hằng năm" `[nd13_2023-p0025-fixed-5cc52b0b382d1bdb]` | answer 2, context 2 |
| T04 | có đáp án | 1 lượt; trang 9 | "0.9 BLEU worse than the best setting" `[attention_2017-p0009-fixed-f682efbe9bebfdea]` | answer 2, context 2 |
| T05 | có đáp án | 1 lượt; trang 2 | "inherently sequential nature" `[attention_2017-p0002-fixed-a6bf9730a6ebd84a]` | answer 2, context 2 |
| U03 | thiếu evidence | dense 3 lượt, hybrid_neighbors 2 lượt (agent viết lại query); context chỉ có nội dung Hồ sơ đánh giá tác động | "Tài liệu không cung cấp thông tin về lệ phí…" | Cả hai: abstention đúng, không đưa con số |
| U04 | thiếu evidence | 1 lượt; context có kết quả BLEU EN-DE/EN-FR trang 8 | dense: từ chối, nêu chỉ có EN-DE/EN-FR. **hybrid_neighbors: "27.3 (base) và 28.4 (big) on English-to-Chinese"** `[attention_2017-p0008-…]` | dense: abstention đúng. **hybrid_neighbors: abstention sai; citation không hỗ trợ — số đó là EN-DE, không phải tiếng Trung** |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** retrieval `results/retrieval/<cấu hình>_<split>/{traces,summary}.json` (top-3 gồm ID, text, score, rank, nguồn, trang, hit; context cuối; thời gian; token). Agent `results/agent/<cấu hình>_<split>[_v1]/traces.json` (từng tool call, query, nguồn query gốc/agent, kết quả, context ID, context cuối, câu trả lời, citation, event sửa citation, token, chi phí, lỗi), `summary_auto.json`; nhãn tay `results/agent/*_test/reviews.json`, tổng hợp bằng `starter/agent_metrics.py` → `summary_labels.json`; bảng `results/tables.md`; chốt cấu hình `results/lock.json`. Lệnh chạy xem `RUN.md`. Tổng chi phí API toàn bộ thí nghiệm ≈ **0,10 USD** (retrieval ≈ 0,012; agent dev v1 0,043; dev v2 0,039; test 0,006), ước tính từ token API trả về × giá cấu hình, chưa đối chiếu hóa đơn.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - *Retrieval:* hybrid tăng Hit@3 dev 16 → 18/20 so với dense; mở rộng chunk lân cận tăng Context Evidence 18 → 20/20 với chi phí gần như bằng 0 nhưng context dài gần gấp đôi (≈ +0,003 USD/22 câu ở agent); rerank tốt nhất về xếp hạng (Hit@1 18/20) nhưng chậm ~10 lần và tốn token LLM.
  - *Agent:* v2 tăng số câu có đáp án được trả lời từ 7–8/20 lên 19–20/20 trên dev mà vẫn giữ từ chối đúng (trừ rerank).
  - *Trên test, lựa chọn trên dev **không được xác nhận**:* retrieval và câu trả lời của hai cấu hình như nhau (Hit@3 5/5, context 5/5, grounded 1,80), nhưng cấu hình đã chọn **trả lời bịa U04** (lấy BLEU EN-DE gán cho EN-Chinese), trong khi baseline từ chối đúng. Giả thuyết: context rộng hơn chứa nhiều đoạn kết quả BLEU "liên quan" khiến model tin là có đủ thông tin — cùng kiểu lỗi đã thấy ở `hybrid_rerank` với U02 trên dev. Không chỉnh hệ thống theo test; hướng sửa (cho thí nghiệm sau, cần test mới): kiểm tra thực thể/điều kiện trong câu hỏi có xuất hiện trong context trước khi trả lời, hoặc thêm bước verifier.
  - *Ví dụ có bằng chứng:* T01 (72 giờ + ngoại lệ, citation đúng đoạn). *Ví dụ thiếu bằng chứng:* U03 (cả hai từ chối đúng). *Ví dụ lỗi:* U04 với `hybrid_neighbors`; T02 thiếu ngoại lệ Điều 17, 18.
  - *Giới hạn:* test rất nhỏ (5 + 2 câu; 1 câu = 20 hoặc 50 điểm %), không đủ kết luận thống kê; dev gồm câu Day 6 đã từng xem; nhãn do một người chấm và là tác giả (nguy cơ thiên lệch), dev chỉ có chỉ số tự động; hậu kiểm citation v2 có sửa định dạng (đã ghi event) nên "câu có citation" phản ánh cả bước hậu xử lý; thời gian đo một lần, phụ thuộc mạng; giá token là giá cấu hình; corpus chỉ 2 tài liệu khác ngôn ngữ nên filter không được thử nghiệm sâu; kết luận chỉ áp dụng cho corpus này.
- **Sử dụng AI:** dùng Claude hỗ trợ viết code (`src/`), soạn nháp câu hỏi test mới và trình bày báo cáo. Tác giả đã kiểm chứng evidence trên PDF, chạy toàn bộ thí nghiệm, đọc trace và **tự chấm toàn bộ nhãn test**; mọi số liệu lấy từ file kết quả của lần chạy thật.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.