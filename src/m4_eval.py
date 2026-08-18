from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        frame = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        ).to_pandas()
        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=float(row.get("faithfulness", 0.0)),
                answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                context_precision=float(row.get("context_precision", 0.0)),
                context_recall=float(row.get("context_recall", 0.0)),
            )
            for _, row in frame.iterrows()
        ]
        return {
            "faithfulness": _average_metric(per_question, "faithfulness"),
            "answer_relevancy": _average_metric(per_question, "answer_relevancy"),
            "context_precision": _average_metric(per_question, "context_precision"),
            "context_recall": _average_metric(per_question, "context_recall"),
            "per_question": per_question,
            "evaluation_mode": "ragas",
        }
    except Exception as exc:
        print(f"  ⚠️  RAGAS evaluation unavailable; using labelled lexical fallback: {exc}")
        per_question = [
            _fallback_result(question, answer, context, ground_truth)
            for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": _average_metric(per_question, "faithfulness"),
            "answer_relevancy": _average_metric(per_question, "answer_relevancy"),
            "context_precision": _average_metric(per_question, "context_precision"),
            "context_recall": _average_metric(per_question, "context_recall"),
            "per_question": per_question,
            "evaluation_mode": "lexical_fallback",
        }


def _terms(text: str) -> set[str]:
    """Normalize a short Vietnamese text into stable lexical terms."""
    import re

    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _ratio(numerator: set[str], denominator: set[str]) -> float:
    return len(numerator.intersection(denominator)) / len(denominator) if denominator else 0.0


def _fallback_result(question: str, answer: str, contexts: list[str], ground_truth: str) -> EvalResult:
    """Transparent, dependency-free proxy scores used only when RAGAS cannot run."""
    answer_terms = _terms(answer)
    question_terms = _terms(question)
    context_terms = _terms(" ".join(contexts))
    ground_truth_terms = _terms(ground_truth)
    context_matches = [_ratio(_terms(context), ground_truth_terms) for context in contexts]
    return EvalResult(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        faithfulness=_ratio(answer_terms, context_terms),
        answer_relevancy=_ratio(answer_terms, question_terms),
        context_precision=sum(context_matches) / len(context_matches) if context_matches else 0.0,
        context_recall=_ratio(context_terms, ground_truth_terms),
    )


def _average_metric(results: list[EvalResult], name: str) -> float:
    return sum(float(getattr(result, name)) for result in results) / len(results) if results else 0.0


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("Câu trả lời có nội dung không được context hỗ trợ.", "Siết prompt chỉ-dựa-context và giảm temperature."),
        "context_recall": ("Retriever thiếu chunk chứa bằng chứng cần thiết.", "Rà lại chunking, tăng hybrid recall hoặc thêm query expansion."),
        "context_precision": ("Context có quá nhiều chunk không liên quan.", "Tăng trọng số reranking hoặc lọc theo metadata/section."),
        "answer_relevancy": ("Câu trả lời không tập trung trực tiếp vào câu hỏi.", "Cải thiện answer prompt để nêu đáp án trước, rồi giải thích ngắn."),
    }
    analyzed = []
    for result in eval_results:
        metrics = {
            "faithfulness": float(result.faithfulness),
            "answer_relevancy": float(result.answer_relevancy),
            "context_precision": float(result.context_precision),
            "context_recall": float(result.context_recall),
        }
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "worst_metric": worst_metric,
            "score": round(sum(metrics.values()) / len(metrics), 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return sorted(analyzed, key=lambda item: item["score"])[:max(bottom_n, 0)]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
