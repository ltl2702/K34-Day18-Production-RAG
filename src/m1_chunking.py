from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if sentence.strip()]
    if not sentences:
        return []

    embeddings = None
    try:
        from sentence_transformers import SentenceTransformer

        # Avoid an unexpected multi-hundred-MB download during offline pipeline runs.
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        embeddings = model.encode(sentences)
    except Exception as exc:
        print(f"  ⚠️  Semantic embedding unavailable; using sentence fallback: {exc}")

    groups: list[list[str]] = [[sentences[0]]]
    for index, sentence in enumerate(sentences[1:], start=1):
        is_new_topic = False
        if embeddings is not None:
            from numpy import dot
            from numpy.linalg import norm

            similarity = dot(embeddings[index - 1], embeddings[index]) / (
                norm(embeddings[index - 1]) * norm(embeddings[index]) + 1e-9
            )
            is_new_topic = similarity < threshold
        elif len(groups[-1]) >= 3:
            # Deterministic fallback keeps related neighbouring sentences together.
            is_new_topic = True

        if is_new_topic:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(
            text=" ".join(group),
            metadata={**metadata, "chunk_index": index, "strategy": "semantic"},
        )
        for index, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if not text.strip() or parent_size <= 0 or child_size <= 0:
        return [], []

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    parent_texts: list[str] = []
    current: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        added_length = len(paragraph) + (2 if current else 0)
        if current and current_length + added_length > parent_size:
            parent_texts.append("\n\n".join(current))
            current, current_length = [], 0
        current.append(paragraph)
        current_length += len(paragraph) + (2 if len(current) > 1 else 0)
    if current:
        parent_texts.append("\n\n".join(current))

    parents: list[Chunk] = []
    children: list[Chunk] = []
    source_prefix = re.sub(r"\W+", "_", str(metadata.get("source", "document"))).strip("_") or "document"
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{source_prefix}_parent_{parent_index}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": parent_id, "chunk_index": parent_index},
        ))
        words = parent_text.split()
        child_parts: list[str] = []
        child_length = 0
        for word in words:
            added_length = len(word) + (1 if child_parts else 0)
            if child_parts and child_length + added_length > child_size:
                children.append(Chunk(
                    text=" ".join(child_parts),
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=parent_id,
                ))
                child_parts, child_length = [], 0
            child_parts.append(word)
            child_length += len(word) + (1 if len(child_parts) > 1 else 0)
        if child_parts:
            children.append(Chunk(
                text=" ".join(child_parts),
                metadata={**metadata, "chunk_type": "child"},
                parent_id=parent_id,
            ))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    current_header = ""
    content_lines: list[str] = []

    def emit_section() -> None:
        content = "\n".join(content_lines).strip()
        if not content and not current_header:
            return
        section = re.sub(r"^#{1,6}\s+", "", current_header).strip() or "Mở đầu"
        chunk_text = f"{current_header}\n{content}".strip() if current_header else content
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_index": len(chunks), "section": section, "strategy": "structure"},
            ))

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+.+$", line):
            emit_section()
            current_header = line.strip()
            content_lines = []
        else:
            content_lines.append(line)
    emit_section()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
