# Individual Reflection — Lab 18 Production RAG

**Tên:** K34 student (thay bằng họ tên trước khi nộp)  
**Phạm vi:** M1–M5 integration và evaluation

## 1. Mapping lecture concepts vào code

| Lecture concept | Module | Hàm cụ thể | Observation từ lần chạy |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng cosine embedding khi model local có sẵn; fallback vẫn nhóm câu để pipeline offline không rỗng. |
| Parent-child retrieval | M1 | `chunk_hierarchical()` | 100 child chunks được tạo từ 26 documents; child giữ đúng `parent_id` của parent. |
| BM25 tiếng Việt + RRF | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | `nghỉ_phép` được tách lại thành hai token; RRF trả `method="hybrid"`. Dense chưa chạy vì model/Qdrant thiếu. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Có contract top-k giảm dần; runtime chuyển lexical fallback khi không có `sentence_transformers`. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Báo cáo gắn `lexical_fallback`, không nhầm proxy với RAGAS vì thiếu `datasets`/`ragas`. |
| Contextual embeddings/HyQA | M5 | `_enrich_single_call()`, `generate_hypothesis_questions()` | Combined mode enrich 100 chunk với một call/chunk; fallback vẫn thêm source context + câu hỏi khi thiếu key/API. |

## 2. Khó khăn và cách giải quyết

- **Exact error:** `ModuleNotFoundError: No module named 'rank_bm25'`. Root cause là
  Python runtime chưa cài dependency dù nó có trong `requirements.txt`. Tôi giữ
  `rank_bm25` là implementation ưu tiên và thêm lexical fallback có scope hẹp để test
  và pipeline không vỡ.
- **Exact runtime error:** `No module named 'sentence_transformers'` và Docker báo
  `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`.
  Tôi không che lỗi này: DenseSearch trả collection rỗng có warning, hybrid vẫn chạy
  BM25, và report nêu rõ degradation.
- **Exact console error:** `UnicodeEncodeError: 'charmap' codec can't encode character`
  khi in emoji tiếng Việt trong CP1252. Chạy `python -X utf8` khắc phục tại process
  boundary mà không làm bẩn source/checker.
- **Exact evaluation error:** `No module named 'datasets'`. M4 trả metric proxy có
  `evaluation_mode: lexical_fallback`; nhờ vậy bottom failures vẫn được phân tích,
  nhưng không được gọi là RAGAS score.

## 3. Đọc kết quả và tối ưu ưu tiên

Production tăng context recall proxy lên **0.7700**, nhưng faithfulness proxy chỉ
**0.0952**. Bottom failures phần lớn là `Không tìm thấy.`, do đó lỗi đầu tiên nằm ở
evidence retrieval availability chứ không phải sáng tạo câu trả lời. Tối ưu tiếp theo:

1. Cài đúng Python 3.11+ environment: `rank-bm25`, `sentence-transformers`, `ragas`,
   `datasets`; bật Docker/Qdrant và download bge-m3/reranker.
2. Rebuild dense index, chạy lại cùng test set bằng RAGAS thật; lưu latency M1–M5.
3. Với lỗi multi-hop/numeric còn lại, chunk section/table-aware và query decomposition
   trước RRF; xác nhận M3 có cải thiện context precision.
4. Chỉ khi context đã có bằng chứng mới tune prompt answer và temperature.

## 4. Action plan áp dụng cho project cá nhân

### Project: Vietnamese policy assistant

### Hiện tại

- Pipeline cần phục vụ câu hỏi policy, version và numeric rule bằng tiếng Việt.
- Known issues: synonym mismatch, query đa bước và dependency/service readiness.

### Plan áp dụng

1. [ ] **Chunking:** parent-child + structure-aware theo heading/table để retrieve nhỏ,
   answer có context đủ.
2. [ ] **Search:** BM25 Vietnamese + bge-m3 dense qua RRF, monitor empty-result rate.
3. [ ] **Reranking:** dùng bge-reranker-v2-m3 cho top-20 → top-3 sau khi benchmark latency.
4. [ ] **Evaluation:** RAGAS bốn metric với test set version/negation/numeric; fallback
   metrics chỉ dùng cho local diagnostics.
5. [ ] **Enrichment:** combined contextual prepend + HyQA cho policy terms trước embed.

### Timeline

- **Tuần 1:** Chuẩn hóa environment/Qdrant, index corpus, baseline + RAGAS thật.
- **Tuần 2:** Parent-child/structure-aware + hybrid retrieval, theo dõi recall và empty rate.
- **Tuần 3:** Rerank, query decomposition, failure-analysis loop cho bottom-10.
