# Production RAG Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện M1–M5, chạy từng kiểm thử hẹp, rồi tạo báo cáo RAG thực tế.

**Architecture:** Các module giữ API scaffold hiện có. External model/service được lazy-load và mọi biên external có fallback cục bộ đúng dataclass; pipeline do đó không đổ vỡ khi offline hoặc thiếu API key.

**Tech Stack:** Python 3.11, pytest, underthesea, rank-bm25, Qdrant, sentence-transformers, RAGAS, OpenAI SDK.

**Spec:** `docs/superpowers/specs/2026-08-18-production-rag-design.md`

## Global Constraints

- Không chỉnh sửa thay đổi có sẵn của người dùng trong `requirements.txt`.
- Dùng `query_points()` cho Qdrant dense search.
- Fallback không được gọi API hoặc tải model khi điều kiện không đáp ứng.
- Mỗi module phải có một test hẹp RED trước code và một test hẹp GREEN sau code.
- Báo cáo chỉ dùng metric và failure thu được từ lệnh chạy thực tế.

---

### Task 1: M1 Chunking

**Files:**
- Modify: `src/m1_chunking.py`
- Test: `tests/test_m1.py`

**Interfaces:**
- Produces: `chunk_semantic(...) -> list[Chunk]`, `chunk_hierarchical(...) -> tuple[list[Chunk], list[Chunk]]`, `chunk_structure_aware(...) -> list[Chunk]`.

- [ ] **Step 1: Run the existing M1 behavioral tests as RED**

Run: `python -m pytest tests/test_m1.py -q`

Expected: semantic, hierarchical, and structure-aware tests fail because TODO implementations return empty output.

- [ ] **Step 2: Implement the three chunking strategies**

```python
# Semantic: encode sentences when available, otherwise preserve sentence grouping.
# Hierarchical: parents have metadata["parent_id"] and each child.parent_id matches it.
# Structure-aware: emitted text starts with the Markdown heading and metadata["section"] holds it.
```

- [ ] **Step 3: Run M1 GREEN verification**

Run: `python -m pytest tests/test_m1.py -q`

Expected: all M1 tests pass.

### Task 2: M2 Hybrid Search

**Files:**
- Modify: `src/m2_search.py`
- Test: `tests/test_m2.py`

**Interfaces:**
- Consumes: chunk dicts containing `text` and `metadata`.
- Produces: `SearchResult` with `method` set to `bm25`, `dense`, or `hybrid`.

- [ ] **Step 1: Run the existing M2 behavioral tests as RED**

Run: `python -m pytest tests/test_m2.py -q`

Expected: BM25 and RRF tests fail because the methods return empty lists.

- [ ] **Step 2: Implement segmentation, BM25, dense Qdrant, and RRF**

```python
tokens = segment_vietnamese(text).lower().split()
score += 1.0 / (k + rank + 1)
return SearchResult(text=text, score=score, metadata=metadata, method="hybrid")
```

- [ ] **Step 3: Run M2 GREEN verification**

Run: `python -m pytest tests/test_m2.py -q`

Expected: all M2 tests pass without requiring a Qdrant connection.

### Task 3: M3 Reranking

**Files:**
- Modify: `src/m3_rerank.py`
- Test: `tests/test_m3.py`

**Interfaces:**
- Consumes: query string and document dicts with `text`, optional `score`, optional `metadata`.
- Produces: decreasing `list[RerankResult]`, at most `top_k` results.

- [ ] **Step 1: Run the existing M3 behavioral tests as RED**

Run: `python -m pytest tests/test_m3.py -q`

Expected: reranking tests fail because `rerank` returns an empty list.

- [ ] **Step 2: Implement lazy CrossEncoder and lexical fallback**

```python
pairs = [(query, document["text"]) for document in documents]
scored = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
```

- [ ] **Step 3: Run M3 GREEN verification**

Run: `python -m pytest tests/test_m3.py -q`

Expected: all M3 tests pass and the leave document sorts ahead of unrelated documents.

### Task 4: M4 Evaluation and Diagnostics

**Files:**
- Modify: `src/m4_eval.py`
- Test: `tests/test_m4.py`

**Interfaces:**
- Produces: dict with four float metrics and `per_question`; diagnosis dicts always contain `diagnosis` and `suggested_fix`.

- [ ] **Step 1: Run the existing M4 behavioral tests as RED**

Run: `python -m pytest tests/test_m4.py -q`

Expected: failure-analysis tests fail because it returns an empty list.

- [ ] **Step 2: Implement RAGAS try/except plus diagnostic tree**

```python
average = sum(metric_values) / 4
worst_metric = min(metric_values, key=metric_values.get)
diagnosis, suggested_fix = diagnostic_tree[worst_metric]
```

- [ ] **Step 3: Run M4 GREEN verification**

Run: `python -m pytest tests/test_m4.py -q`

Expected: all M4 tests pass with or without an OpenAI key.

### Task 5: M5 Enrichment

**Files:**
- Modify: `src/m5_enrichment.py`
- Test: `tests/test_m5.py`

**Interfaces:**
- Produces: `EnrichedChunk` that preserves `original_text`, returns contextual enriched text, and preserves incoming metadata.

- [ ] **Step 1: Run the existing M5 behavioral tests as RED**

Run: `python -m pytest tests/test_m5.py -q`

Expected: contextual prepend test fails because the scaffold returns unmodified text.

- [ ] **Step 2: Implement combined OpenAI enrichment plus deterministic fallbacks**

```python
prefix = f"Trích từ {document_title}. " if document_title else ""
return f"{prefix}{text}"
```

- [ ] **Step 3: Run M5 GREEN verification**

Run: `python -m pytest tests/test_m5.py -q`

Expected: all M5 tests pass without an API key.

### Task 6: End-to-end evaluation and authored deliverables

**Files:**
- Create/modify: `reports/naive_baseline_report.json`, `reports/ragas_report.json`
- Modify: `analysis/failure_analysis.md`, `analysis/group_report.md`
- Create: `analysis/reflections/reflection_K34.md`

**Interfaces:**
- Consumes: report JSON emitted by `main.py` and `failure_analysis` output.
- Produces: complete JSON and markdown deliverables grounded in actual run output.

- [ ] **Step 1: Run full test suite and format checks**

Run: `python -m pytest tests -q; python check_lab.py`

Expected: all unit tests pass before attempting the heavier pipeline run.

- [ ] **Step 2: Run baseline and production pipeline**

Run: `python main.py`

Expected: reports are emitted; if an external dependency is unavailable, capture the exact exception and fallback metrics in the reports.

- [ ] **Step 3: Fill markdown deliverables from the generated JSON**

```markdown
The next optimization targets <lowest metric>, because the bottom failures map to
<diagnostic root cause>; apply <suggested fix> before tuning generation.
```

- [ ] **Step 4: Final verification**

Run: `python -m pytest tests -q; python check_lab.py; rg -n '# TODO' src/m*.py`

Expected: tests pass, checker validates required artifacts, and no TODO markers remain.
