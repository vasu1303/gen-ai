from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import requests

from src.rag_pipeline import answer_question, initialize_rag


OUTPUT_DIR = Path("report")
OUTPUT_JSON = OUTPUT_DIR / "evaluation_results.json"


@dataclass
class EvaluationQuestion:
    id: int
    question: str
    expected_keywords: list[str]
    expected_sources: list[str]
    is_followup: bool = False


@dataclass
class QuestionResult:
    id: int
    question: str
    answer: str
    sources: list[dict[str, Any]]
    retrieval_time: float
    llm_time: float
    keyword_accuracy: float
    relevance: float
    contextual_awareness: float
    response_quality: float
    overall: float


def check_ollama_available(base_url: str = "http://localhost:11434") -> None:
    try:
        response = requests.get(
            f"{base_url}/api/tags",
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            "Ollama is not reachable at http://localhost:11434.\n"
            "Start Ollama before running evaluation.\n"
            "Example:\n"
            "  ollama serve\n"
            "  ollama pull qwen2.5:1.5b\n"
            "  ollama pull qwen3-vl:4b-instruct"
        ) from error


def build_predefined_questions() -> list[EvaluationQuestion]:
    return [
        EvaluationQuestion(
            id=1,
            question="What is the core idea introduced in the Transformer paper?",
            expected_keywords=["attention", "sequence", "transformer"],
            expected_sources=["1706.03762.pdf"],
        ),
        EvaluationQuestion(
            id=2,
            question="Why does BERT use masked language modeling and next sentence prediction?",
            expected_keywords=["masked language model", "next sentence", "bidirectional"],
            expected_sources=["1810.04805.pdf"],
        ),
        EvaluationQuestion(
            id=3,
            question="How does retrieval improve language generation in Retrieval-Augmented Generation models?",
            expected_keywords=["retrieval", "generation", "knowledge"],
            expected_sources=["2005.14165.pdf"],
        ),
        EvaluationQuestion(
            id=4,
            question="What challenge does RoBERTa address compared with BERT training?",
            expected_keywords=["training", "longer", "data"],
            expected_sources=["1907.11692.pdf"],
        ),
        EvaluationQuestion(
            id=5,
            question="Explain what DistilBERT keeps and what it removes from BERT.",
            expected_keywords=["distillation", "smaller", "faster"],
            expected_sources=["1910.10683.pdf"],
        ),
        EvaluationQuestion(
            id=6,
            question="In Transformer, what is the role of positional encoding?",
            expected_keywords=["position", "order", "encoding"],
            expected_sources=["1706.03762.pdf"],
        ),
        EvaluationQuestion(
            id=7,
            question="How does RAG compare parametric memory with non-parametric memory?",
            expected_keywords=["parametric", "non-parametric", "retriever"],
            expected_sources=["2005.14165.pdf"],
        ),
        EvaluationQuestion(
            id=8,
            question="Summarize major architectural differences among Transformer, BERT, and DistilBERT.",
            expected_keywords=["encoder", "distillation", "attention"],
            expected_sources=["1706.03762.pdf", "1810.04805.pdf", "1910.10683.pdf"],
        ),
        EvaluationQuestion(
            id=9,
            question="Based on our previous discussion, which model is best when low latency is important?",
            expected_keywords=["distilbert", "faster", "latency"],
            expected_sources=["1910.10683.pdf", "1907.11692.pdf"],
            is_followup=True,
        ),
        EvaluationQuestion(
            id=10,
            question="Considering your previous answer, what trade-off should we expect if we pick that model?",
            expected_keywords=["trade-off", "accuracy", "speed"],
            expected_sources=["1910.10683.pdf"],
            is_followup=True,
        ),
    ]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def keyword_accuracy(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    normalized = normalize_text(answer)
    hit_count = sum(
        1 for keyword in expected_keywords
        if normalize_text(keyword) in normalized
    )
    return round(hit_count / len(expected_keywords), 4)


def relevance_score(
    answer: str,
    question: str,
    sources: list[dict[str, Any]],
    expected_sources: list[str],
) -> float:
    answer_norm = normalize_text(answer)
    question_words = {
        token for token in re.findall(r"[a-zA-Z]{4,}", question.lower())
    }
    overlap = 0.0
    if question_words:
        overlap_hits = sum(
            1 for token in question_words if token in answer_norm
        )
        overlap = overlap_hits / len(question_words)

    if expected_sources:
        retrieved = {item.get("source", "") for item in sources}
        source_hit = (
            len(set(expected_sources).intersection(retrieved))
            / len(set(expected_sources))
        )
    else:
        source_hit = 1.0

    return round((0.6 * overlap) + (0.4 * source_hit), 4)


def contextual_awareness_score(
    answer: str,
    is_followup: bool,
) -> float:
    normalized = normalize_text(answer)
    if not is_followup:
        return 1.0
    contextual_tokens = {
        "previous", "earlier", "trade-off", "latency",
        "distilbert", "based on"
    }
    hits = sum(1 for token in contextual_tokens if token in normalized)
    return round(min(1.0, hits / 2), 4)


def response_quality_score(answer: str) -> float:
    word_count = len(answer.split())
    sentence_count = len(
        [s for s in re.split(r"[.!?]+", answer) if s.strip()]
    )
    length_score = 1.0 if 40 <= word_count <= 260 else 0.6
    sentence_score = 1.0 if sentence_count >= 2 else 0.7
    return round((length_score + sentence_score) / 2, 4)


def compute_overall_score(
    keyword: float,
    relevance: float,
    contextual: float,
    quality: float,
) -> float:
    return round(
        (0.35 * keyword)
        + (0.30 * relevance)
        + (0.20 * contextual)
        + (0.15 * quality),
        4,
    )


def evaluate_questions() -> dict[str, Any]:
    check_ollama_available()

    (
        embedding_model,
        index,
        metadata,
        image_index,
        image_metadata,
    ) = initialize_rag()

    questions = build_predefined_questions()
    results: list[QuestionResult] = []

    for item in questions:
        print(f"\n[{item.id}/10] {item.question}")
        rag_result = answer_question(
            question=item.question,
            embedding_model=embedding_model,
            index=index,
            metadata=metadata,
            image_index=image_index,
            image_metadata=image_metadata,
        )
        answer = rag_result["answer"]
        sources = rag_result.get("sources", [])

        score_keyword = keyword_accuracy(answer, item.expected_keywords)
        score_relevance = relevance_score(
            answer=answer,
            question=item.question,
            sources=sources,
            expected_sources=item.expected_sources,
        )
        score_context = contextual_awareness_score(
            answer=answer,
            is_followup=item.is_followup,
        )
        score_quality = response_quality_score(answer)
        overall = compute_overall_score(
            keyword=score_keyword,
            relevance=score_relevance,
            contextual=score_context,
            quality=score_quality,
        )

        result = QuestionResult(
            id=item.id,
            question=item.question,
            answer=answer,
            sources=sources,
            retrieval_time=rag_result["retrieval_time"],
            llm_time=rag_result["llm_time"],
            keyword_accuracy=score_keyword,
            relevance=score_relevance,
            contextual_awareness=score_context,
            response_quality=score_quality,
            overall=overall,
        )
        results.append(result)
        print(
            f"  Scores -> keyword:{score_keyword:.2f}, "
            f"relevance:{score_relevance:.2f}, "
            f"context:{score_context:.2f}, "
            f"quality:{score_quality:.2f}, "
            f"overall:{overall:.2f}"
        )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "question_count": len(results),
        "aggregate": {
            "keyword_accuracy": round(mean(r.keyword_accuracy for r in results), 4),
            "relevance": round(mean(r.relevance for r in results), 4),
            "contextual_awareness": round(mean(r.contextual_awareness for r in results), 4),
            "response_quality": round(mean(r.response_quality for r in results), 4),
            "overall": round(mean(r.overall for r in results), 4),
            "avg_retrieval_time_sec": round(mean(r.retrieval_time for r in results), 4),
            "avg_generation_time_sec": round(mean(r.llm_time for r in results), 4),
        },
        "questions": [asdict(item) for item in results],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run predefined 10-question evaluation for the RAG system."
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_JSON),
        help="Path to save evaluation JSON output.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = evaluate_questions()

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(f"\nSaved evaluation results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
