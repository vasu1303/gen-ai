from __future__ import annotations

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PATH = PROJECT_ROOT / "report" / "evaluation_results.json"
OUTPUT_PATH = PROJECT_ROOT / "report" / "final_report.pdf"


def chunk_lines(text: str, max_chars: int = 95) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        add_len = len(word) if not current else len(word) + 1
        if current_len + add_len > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += add_len
    if current:
        lines.append(" ".join(current))
    return lines


def add_block(
    page: pymupdf.Page,
    lines: Iterable[str],
    y: float,
    font_size: float = 11,
    left_margin: float = 50,
    line_height: float = 15,
) -> float:
    for line in lines:
        page.insert_text(
            pymupdf.Point(left_margin, y),
            line,
            fontsize=font_size,
        )
        y += line_height
    return y


def add_wrapped_paragraph(
    page: pymupdf.Page,
    text: str,
    y: float,
    font_size: float = 11,
    left_margin: float = 50,
    line_height: float = 15,
) -> float:
    return add_block(
        page=page,
        lines=chunk_lines(text),
        y=y,
        font_size=font_size,
        left_margin=left_margin,
        line_height=line_height,
    )


def add_header(page: pymupdf.Page, title: str, subtitle: str) -> float:
    y = 60
    y = add_block(page, [title], y=y, font_size=18, line_height=24)
    y = add_block(page, [subtitle], y=y + 6, font_size=10, line_height=14)
    return y + 14


def load_evaluation(evaluation_path: Path) -> dict:
    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found at {evaluation_path}. "
            "Run run_predefined_evaluation.py first."
        )
    with evaluation_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def generate_report_pdf(
    evaluation_path: Path = EVALUATION_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    evaluation = load_evaluation(evaluation_path)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    document = pymupdf.open()

    # Page 1: Overview + technical implementation
    page1 = document.new_page(width=595, height=842)  # A4
    y = add_header(
        page1,
        "Final Report: Multimodal RAG Assignment",
        f"Generated on {generated_at}",
    )

    y = add_block(page1, ["1. Overview"], y=y, font_size=13, line_height=18)
    y = add_wrapped_paragraph(
        page1,
        "This project implements a Retrieval-Augmented Generation (RAG) system over five "
        "research papers. The workflow covers PDF ingestion, chunking, embedding generation, "
        "FAISS-based semantic retrieval, open-source LLM response generation using Ollama, and "
        "short-term conversational memory over the latest four interactions.",
        y=y,
    )

    y += 8
    y = add_block(page1, ["2. Technical Implementation"], y=y, font_size=13, line_height=18)
    y = add_wrapped_paragraph(
        page1,
        "PDF ingestion uses PyMuPDF to extract text from each page and detect table content. "
        "Chunks are created via RecursiveCharacterTextSplitter (chunk size 900, overlap 150). "
        "Embeddings are generated with BAAI/bge-small-en-v1.5 and stored in FAISS IndexFlatIP "
        "with normalized vectors for cosine-similarity-style retrieval.",
        y=y,
    )
    y += 5
    y = add_wrapped_paragraph(
        page1,
        "The chatbot integrates qwen2.5:1.5b (text) and qwen3-vl:4b-instruct (vision) from "
        "Ollama. Responses are generated from retrieved context and conversation history. "
        "Conversation memory is implemented using a deque(maxlen=4), ensuring only the last "
        "four interactions are retained.",
        y=y,
    )

    y += 8
    y = add_block(page1, ["3. Assignment Coverage"], y=y, font_size=13, line_height=18)
    checklist = [
        "- PDF ingestion from all 5 assignment papers: implemented.",
        "- Text chunking and semantic embeddings: implemented.",
        "- Vector database creation (FAISS): implemented.",
        "- Open-source LLM integration: implemented through Ollama.",
        "- Conversational memory for last 4 interactions: implemented.",
        "- 10 predefined questions and evaluation script: implemented.",
        "- Automated final report PDF generation: implemented.",
    ]
    y = add_block(page1, checklist, y=y, font_size=11, line_height=15)

    # Page 2: Evaluation and framework
    page2 = document.new_page(width=595, height=842)
    y = add_header(
        page2,
        "4. Evaluation Methodology",
        "Self-devised metric with 10 predefined questions and weighted scoring",
    )

    y = add_wrapped_paragraph(
        page2,
        "Evaluation is executed through live model calls using run_predefined_evaluation.py. "
        "Each answer is scored using four criteria: keyword accuracy, relevance to question and "
        "retrieved sources, contextual awareness for follow-up queries, and response quality. "
        "The overall score is computed as: 0.35*keyword + 0.30*relevance + 0.20*context + "
        "0.15*quality.",
        y=y,
    )

    y += 10
    y = add_block(page2, ["Predefined 10 test questions:"], y=y, font_size=12, line_height=16)
    for question in evaluation.get("questions", []):
        line = f"Q{question['id']}. {question['question']}"
        if y > 790:
            page2 = document.new_page(width=595, height=842)
            y = 60
        y = add_wrapped_paragraph(page2, line, y=y)

    # Page 3: Quantitative results
    page3 = document.new_page(width=595, height=842)
    y = add_header(
        page3,
        "5. Results & Discussion",
        "Aggregate scores, per-question performance, and analysis",
    )

    aggregate = evaluation.get("aggregate", {})
    summary_lines = [
        f"Question count: {evaluation.get('question_count', 0)}",
        f"Average keyword accuracy: {aggregate.get('keyword_accuracy', 0):.4f}",
        f"Average relevance: {aggregate.get('relevance', 0):.4f}",
        f"Average contextual awareness: {aggregate.get('contextual_awareness', 0):.4f}",
        f"Average response quality: {aggregate.get('response_quality', 0):.4f}",
        f"Average overall score: {aggregate.get('overall', 0):.4f}",
        f"Average retrieval latency (s): {aggregate.get('avg_retrieval_time_sec', 0):.4f}",
        f"Average generation latency (s): {aggregate.get('avg_generation_time_sec', 0):.4f}",
    ]
    y = add_block(page3, summary_lines, y=y, font_size=11, line_height=15)

    y += 10
    y = add_block(page3, ["Per-question overview:"], y=y, font_size=12, line_height=16)

    for question in evaluation.get("questions", []):
        line = (
            f"Q{question['id']}: overall={question['overall']:.3f} | "
            f"keyword={question['keyword_accuracy']:.3f} | "
            f"relevance={question['relevance']:.3f} | "
            f"context={question['contextual_awareness']:.3f} | "
            f"quality={question['response_quality']:.3f}"
        )
        if y > 790:
            page3 = document.new_page(width=595, height=842)
            y = 60
        y = add_wrapped_paragraph(page3, line, y=y)

    y += 10
    y = add_wrapped_paragraph(
        page3,
        "Discussion: the system retrieves grounded sources and handles conversational memory "
        "effectively, but quality varies by question complexity and local model capacity. "
        "Long-form technical answers may trade speed for completeness on smaller local models.",
        y=y,
    )

    # Page 4: Conclusion
    page4 = document.new_page(width=595, height=842)
    y = add_header(
        page4,
        "6. Conclusion",
        "Summary, challenges, and improvement opportunities",
    )

    y = add_wrapped_paragraph(
        page4,
        "Results show that retrieval and grounding quality depend strongly on chunk granularity, "
        "source paper overlap, and the quality of semantic embeddings. Follow-up question handling "
        "benefits from bounded memory (last four turns), which preserves conversational focus while "
        "keeping prompts compact.",
        y=y,
    )
    y += 8
    y = add_wrapped_paragraph(
        page4,
        "Key practical challenge: local model serving availability (Ollama must be running and "
        "required models must be pulled). A second challenge is occasional retrieval ambiguity when "
        "multiple papers discuss related concepts. Future improvements include reranking, citation "
        "post-processing, and replacing heuristic evaluation with RAGAS metrics once API/model "
        "constraints are finalized.",
        y=y,
    )
    y += 8
    y = add_wrapped_paragraph(
        page4,
        "Conclusion: the project now satisfies the assignment end-to-end with PDF ingestion, vector "
        "database creation, open-source LLM-based QA, conversational memory, automated 10-question "
        "evaluation, and final report generation in PDF format.",
        y=y,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
    document.close()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate final assignment report PDF from evaluation results."
    )
    parser.add_argument(
        "--evaluation",
        default=str(EVALUATION_PATH),
        help="Path to evaluation JSON input.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Path to output PDF report.",
    )
    args = parser.parse_args()

    output = generate_report_pdf(
        evaluation_path=Path(args.evaluation),
        output_path=Path(args.output),
    )
    print(f"Generated final report: {output.resolve()}")
