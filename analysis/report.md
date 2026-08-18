# Group Report — Lab 18 Production RAG

**Tên**: Lương Thị Linh

## Module và verification

| Module | Deliverable kiểm chứng | Kết quả test hẹp |
|---|---|---:|
| M1 | Semantic, hierarchical parent/child, structure-aware | 13/13 |
| M2 | Vietnamese BM25, dense Qdrant `query_points`, RRF | 5/5 |
| M3 | CrossEncoder lazy-load và lexical fallback top-k | 5/5 |
| M4 | Bốn metric/fallback và Diagnostic Tree | 4/4 |
| M5 | Combined enrichment và fallback không API key | 12/12 |

**Tổng:** 39/39 pytest tests pass trong lần chạy cuối trước pipeline.

## Kết quả runtime

| Metric | Naive | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.0000 | 0.0952 | +0.0952 |
| Answer relevancy | 0.0246 | 0.4617 | +0.4372 |
| Context precision | 0.0000 | 0.4612 | +0.4612 |
| Context recall | 0.0000 | 0.7700 | +0.7700 |

Các số trên thuộc `lexical_fallback`, vì RAGAS thật không khả dụng trong môi trường
chạy. Production đã cải thiện mọi proxy metric, đặc biệt context recall; chưa được
dùng để khẳng định ngưỡng RAGAS rubric.

## Key findings

1. **Biggest improvement:** RRF + enrichment tăng context recall từ 0 lên 0.7700 và
   answer relevancy từ 0.0246 lên 0.4617 so với dense-only baseline bị mất encoder.
2. **Biggest challenge:** Docker Desktop không chạy, đồng thời thiếu
   `sentence-transformers`, `ragas` và `datasets`; dense search/RAGAS vì thế chuyển
   sang fallback có gắn nhãn thay vì làm gãy pipeline.
3. **Surprise finding:** Nhiều failure là `Không tìm thấy.` dù câu hỏi lookup rõ.
   Điều đó chỉ ra readiness của M2 (model + Qdrant) là bottleneck lớn hơn prompt/rerank.

## Case study và bước tiếp theo

Case `laptop 30 triệu` cần evidence từ cả procurement và CNTT. Error tree cho thấy
context rỗng, nên sửa prompt sẽ không tạo được đáp án đúng. Bước đầu tiên là bật
Qdrant + bge-m3, tiếp theo là query decomposition, cuối cùng mới đo lại reranker và
RAGAS. `analysis/failure_analysis.md` ghi chi tiết bottom-5 và fix theo từng case.
