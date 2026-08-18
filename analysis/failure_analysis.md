# Failure Analysis — Lab 18 Production RAG

**Run nguồn:** `python -X utf8 main.py`, 18/08/2026 12:32–12:38 ICT
**Chế độ đánh giá:** `lexical_fallback` — đây là proxy lexical có gắn nhãn,
không phải điểm RAGAS LLM-as-judge. RAGAS không chạy vì môi trường thiếu
`datasets`/`ragas`.

## So sánh kết quả đã chạy

| Metric | Naive baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.0000 | 0.0952 | +0.0952 |
| Answer relevancy | 0.0246 | 0.4617 | +0.4372 |
| Context precision | 0.0000 | 0.4612 | +0.4612 |
| Context recall | 0.0000 | 0.7700 | +0.7700 |

Production tăng recall mạnh, nhưng faithfulness proxy rất thấp. Điều này phù hợp
với observation thực tế: khi dense encoder/Qdrant không sẵn có, một số query không
lấy được candidate BM25 hữu ích và hệ thống trả `Không tìm thấy.`. Vì proxy lexical
coi token trong câu trả lời phải xuất hiện trong context, nó phạt mạnh fallback answer
này; không được diễn giải con số đó là bằng chứng LLM hallucination.

## Bottom-5 failures

### 1. Laptop 30 triệu cho nhân viên mới

- **Expected:** Director phê duyệt; CNTT xác nhận cấu hình; đính kèm ít nhất ba báo giá.
- **Got:** `Không tìm thấy.`
- **Worst metric / score:** answer_relevancy / 0.1670.
- **Diagnosis:** Không có context được trả về cho câu hỏi multi-hop procurement + IT.
- **Error Tree:** Output sai → context đúng? **Không, rỗng** → query/retrieval OK?
  **Không đủ** → root cause: dense retrieval không hoạt động (`sentence_transformers`
  và Docker/Qdrant không sẵn), BM25 không bắc được hai vocabulary domains.
- **Suggested fix:** Khởi động Qdrant và cài bge-m3; sau đó thêm query decomposition
  (`approval threshold` + `IT technical confirmation`) trước RRF.

### 2. Lương thử việc Junior cao nhất

- **Expected:** 17.000.000 VNĐ/tháng (85% của 20.000.000 VNĐ).
- **Got:** `Không tìm thấy.`
- **Worst metric / score:** answer_relevancy / 0.2286.
- **Diagnosis:** Candidate salary/range không tới được generation context.
- **Error Tree:** Output sai → context đúng? **Không** → candidate chứa `Junior/P1`
  có được retrieve? **Không chứng minh được** → root cause: thiếu dense fallback cho
  biến thể `Junior mức cao nhất` và không có arithmetic answer step.
- **Suggested fix:** Bổ sung dense index, HyQA cho salary band, và prompt tính toán chỉ
  từ số có trong context.

### 3. Thiết bị 55 triệu cần ai phê duyệt

- **Expected:** CEO phê duyệt vì giá trị trên 50 triệu.
- **Got:** `Không tìm thấy.`
- **Worst metric / score:** answer_relevancy / 0.2395.
- **Diagnosis:** Retrieval không đưa được bảng threshold mua sắm vào top context.
- **Error Tree:** Output sai → context đúng? **Không** → query OK? **Có, lookup rõ**
  → root cause: recall theo numeric threshold chưa được dense index hỗ trợ.
- **Suggested fix:** Index section/table-aware chunk với metadata `amount_range`; add
  numeric query expansion (`55 triệu` → `trên 50 triệu`) trước rerank.

### 4. Tự xử lý malware

- **Expected:** Không tự xử lý; báo CNTT trong một giờ qua helpdesk/hotline.
- **Got:** `Không tìm thấy.`
- **Worst metric / score:** faithfulness proxy / 0.2751.
- **Diagnosis:** Answer fallback không phục hồi bằng chứng malware từ context. Nhãn
  faithfulness ở đây là tín hiệu proxy, không kết luận hallucination.
- **Error Tree:** Output sai → context đúng? **Không đủ** → lexical alias (`malware`,
  `sự cố`, `helpdesk`) OK? **Chưa** → root cause: chunk/query synonym mismatch.
- **Suggested fix:** Enrichment thêm HyQA/synonym cho incident response và để BM25+dense
  fusion chạy đầy đủ.

### 5. Nghỉ khi kết hôn

- **Expected:** Ba ngày làm việc có lương, không trừ phép năm.
- **Got:** `Không tìm thấy.`
- **Worst metric / score:** faithfulness proxy / 0.3426.
- **Diagnosis:** Câu hỏi policy ngắn vẫn không có context; đây là retrieval availability
  failure, không phải failure của answer prompt.
- **Error Tree:** Output sai → context đúng? **Rỗng** → query OK? **Có** → root cause:
  dense index bị vô hiệu, BM25 candidate không đủ ở runtime fallback.
- **Suggested fix:** Ưu tiên phục hồi dense/Qdrant và thêm retrieval assertion: nếu cả
  dense lẫn BM25 rỗng, log query/chunk coverage thay vì chỉ trả fallback answer.

## Quyết định tối ưu tiếp theo

Ưu tiên **M2 operational readiness** trước prompt tuning: cài `sentence-transformers`,
khởi động Docker/Qdrant, rebuild bge-m3 index, rồi chạy lại RAGAS thật. Khi retrieval
có bằng chứng, đánh giá lại M3 reranking; chỉ sau đó mới tối ưu LLM answer.
