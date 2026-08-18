# Production RAG Pipeline Design

## Goal

Hoàn thiện năm mô-đun RAG để pipeline có thể chạy với các dịch vụ/model thật khi
chúng sẵn có, nhưng vẫn trả về cấu trúc kết quả hợp lệ khi không có API key, model
cache, hoặc Qdrant.

## Architecture

Pipeline giữ thứ tự `chunking -> enrichment -> hybrid retrieval -> reranking ->
answer -> evaluation`. Mỗi mô-đun có một implementation chính phù hợp yêu cầu lab
và một fallback cục bộ, quyết định tại runtime. Fallback không giả mạo kết quả LLM;
nó chỉ giữ contract và tạo dữ liệu có thể kiểm tra được.

## Module Contracts

### M1: Chunking

- `chunk_semantic` tách câu, dùng embedding MiniLM và cosine similarity khi model
  khả dụng; nếu không, dùng nhóm câu/paragraph xác định để vẫn sinh `Chunk`.
- `chunk_hierarchical` tạo parent theo paragraph, gán `metadata.parent_id` duy nhất
  cho parent và cùng giá trị vào `Chunk.parent_id` của mọi child.
- `chunk_structure_aware` phân đoạn markdown theo heading, giữ heading ở đầu text và
  ghi `section` trong metadata.

### M2: Retrieval

- `segment_vietnamese` gọi underthesea, sau đó thay `_` thành khoảng trắng; lỗi thư
  viện sẽ fallback lowercase whitespace tokenization.
- BM25 giữ `SearchResult(method="bm25")`; dense dùng Qdrant `query_points` và giữ
  `SearchResult(method="dense")`.
- RRF cộng `1/(k + rank + 1)` theo text, trả `SearchResult(method="hybrid")`.

### M3: Reranking

- Reranker nạp `sentence_transformers.CrossEncoder` lazily.
- Khi model không sẵn có, điểm lexical overlap được dùng để giữ thứ tự xác định và
  contract `RerankResult`; mọi output được sắp giảm dần theo `rerank_score`.

### M4: Evaluation

- Khi RAGAS hoạt động, trả bốn aggregate metrics và `EvalResult` theo từng câu.
- Khi RAGAS/API không sẵn có, trả bốn số float hợp lệ (0.0) cùng danh sách per
  question rỗng. Failure analysis xếp theo trung bình bốn metric và map metric thấp
  nhất tới diagnosis/fix cụ thể.

### M5: Enrichment

- Các kỹ thuật riêng lẻ dùng OpenAI nếu có key, fallback extractive/deterministic nếu
  không. Combined mode có đúng một API call/chunk, và khi lỗi trả cấu trúc rỗng để
  `enrich_chunks` bảo toàn text/source metadata.

## Error Handling

Model hoặc dịch vụ bên ngoài không được làm vỡ unit test hoặc pipeline: exception
được bắt tại biên integration, warning ngắn được in, và fallback trả đúng dataclass
hoặc collection rỗng. Input rỗng phải trả collection rỗng hợp lệ.

## Verification and Reports

Mỗi module được chạy test hẹp ngay sau khi implement. Sau đó chạy baseline và
pipeline end-to-end. Các file JSON và markdown được tạo từ số liệu/lỗi chạy thực tế,
ghi rõ metric thấp nhất và bước tối ưu kế tiếp. Reflection được ghi với tên trung
tính `reflection_K34.md` vì chưa có họ tên cá nhân.
