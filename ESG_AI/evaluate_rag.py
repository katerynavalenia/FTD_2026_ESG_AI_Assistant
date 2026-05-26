"""
ESG RAG Evaluation Script
Runs a set of CSRD/ESG questions against the RAG engine and saves results to CSV.
Optionally computes Ragas quality metrics (faithfulness, answer relevancy,
context precision, context recall) using the Albert API as the judge LLM.

Usage:
    python evaluate_rag.py                               # all companies, built-in questions
    python evaluate_rag.py --company "Danone"            # filter to one company
    python evaluate_rag.py --questions-file questions.yaml
    python evaluate_rag.py --output data/evaluation/my_results.csv

    # Run with Ragas metrics (requires: pip install ragas langchain-openai datasets)
    python evaluate_rag.py --ragas
    python evaluate_rag.py --ragas --ground-truth ../sample_data/rag_evaluation_dataset.csv
    python evaluate_rag.py --ragas --company "Danone"   # faster — fewer questions
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from csrd_chat import (
    AlbertClient,
    DEFAULT_ALBERT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_TOP_K,
    RAGEngine,
    load_dotenv,
    resolve_api_key,
)

DEFAULT_GROUND_TRUTH = Path("rag_evaluation_dataset.csv")

DEFAULT_OUTPUT = Path("data/evaluation/rag_eval_results.csv")

BUILTIN_QUESTIONS: list[str] = [
    "What are the GHG emissions reduction targets (Scope 1, 2, and 3)?",
    "What net-zero or carbon neutrality commitments have been made?",
    "How does the company address biodiversity and ecosystem risks?",
    "What water management targets or achievements are reported?",
    "Describe the circular economy initiatives and waste reduction targets.",
    "What social commitments are made regarding the workforce?",
    "How does the company support workers in its value chain?",
    "What governance structures oversee sustainability and ESG topics?",
    "What CSRD double materiality assessment has been conducted?",
    "What climate-related risks and opportunities are disclosed?",
    "What renewable energy targets or achievements are reported?",
    "How does the company measure and disclose its supply chain emissions?",
]


def load_questions_yaml(path: Path) -> list[str]:
    """Load questions from a YAML file. Falls back to a simple line-by-line format."""
    try:
        import yaml  # type: ignore

        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(content, list):
            return [str(q) for q in content if q]
        if isinstance(content, dict):
            questions = content.get("questions", [])
            return [str(q) for q in questions if q]
        raise ValueError("YAML file must contain a list of questions or a dict with a 'questions' key.")
    except ImportError:
        logging.warning("PyYAML not installed. Reading questions as plain text (one per line).")
        lines = path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def compute_ragas_metrics(
    rows: list[dict],
    ground_truth_path: Path,
    api_key: str,
    base_url: str,
    chat_model: str,
) -> dict[str, float] | None:
    """
    Compute Ragas metrics (faithfulness, answer_relevancy, context_precision,
    context_recall) by joining the evaluation rows with a ground-truth CSV.

    Requires:
        pip install ragas langchain-openai datasets

    Returns a dict of metric name -> score, or None on failure.
    """
    try:
        import warnings
        import pandas as pd
        from datasets import Dataset
        from ragas import evaluate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import (
                faithfulness,
                context_precision,
                context_recall,
            )
    except ImportError as exc:
        print(
            f"\nRagas dependencies not installed: {exc}\n"
            "Install with: pip install ragas langchain-openai datasets\n",
            file=sys.stderr,
        )
        return None

    # Load ground-truth dataset
    if not ground_truth_path.exists():
        print(
            f"\nGround-truth file not found: {ground_truth_path}\n"
            "Pass --ground-truth <path> to specify a different location.",
            file=sys.stderr,
        )
        return None

    gt_df = pd.read_csv(ground_truth_path)
    # Normalise column names (the CSV may have extra whitespace)
    gt_df.columns = [c.strip() for c in gt_df.columns]

    # Map question text -> ground_truth answer
    gt_map: dict[str, str] = {}
    for _, gt_row in gt_df.iterrows():
        q = str(gt_row.get("question", "")).strip()
        g = str(gt_row.get("ground_truth", "")).strip()
        if q and g:
            gt_map[q] = g

    # Build Ragas dataset rows (only rows that have a matching ground truth)
    ragas_rows: list[dict] = []
    skipped = 0
    for row in rows:
        q = str(row.get("question", "")).strip()
        ground_truth = gt_map.get(q, "")
        if not ground_truth:
            skipped += 1
            continue
        # contexts is a list of strings (one per retrieved chunk)
        citations_raw = str(row.get("citations", ""))
        contexts = [c.strip() for c in citations_raw.split("|") if c.strip()] or [""]
        ragas_rows.append({
            "question": q,
            "answer": str(row.get("answer", "")),
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    if not ragas_rows:
        print(
            f"\nNo evaluation rows matched the ground-truth CSV "
            f"({skipped} questions had no matching ground truth). "
            "Make sure questions match exactly, or use --questions-file with the "
            "same questions as the ground-truth CSV.",
            file=sys.stderr,
        )
        return None

    if skipped:
        print(f"  Note: {skipped} question(s) skipped (no ground-truth match).")

    print(f"\nRunning Ragas on {len(ragas_rows)} matched question(s)…")

    # Configure Albert API as the judge LLM via llm_factory.
    # AnswerRelevancy is excluded — it calls /embeddings which returns HTTP 500
    # on the Albert API.  These three metrics use only chat-completion.
    from openai import OpenAI
    from ragas.llms import llm_factory
    judge_llm = llm_factory(
        chat_model,
        client=OpenAI(api_key=api_key, base_url=base_url),
        temperature=0.0,
    )
    metrics = [faithfulness, context_precision, context_recall]

    dataset = Dataset.from_list(ragas_rows)
    try:
        import inspect
        eval_kwargs: dict = {
            "dataset": dataset,
            "metrics": metrics,
            "llm": judge_llm,
            "allow_nest_asyncio": True,
        }
        if "raise_exceptions" in inspect.signature(evaluate).parameters:
            eval_kwargs["raise_exceptions"] = False
        result = evaluate(**eval_kwargs)
    except Exception as exc:
        print(f"\nRagas evaluation failed: {exc}", file=sys.stderr)
        return None

    # In Ragas 0.2+, result["metric"] returns a list of per-sample floats,
    # not a single aggregate.  Convert to a DataFrame and take the column mean,
    # dropping NaN (which arise from questions with no matching context).
    try:
        result_df = result.to_pandas()
    except Exception as exc:
        print(f"\nFailed to convert Ragas result to DataFrame: {exc}", file=sys.stderr)
        return None

    def _col_mean(name: str) -> float:
        if name not in result_df.columns:
            return 0.0
        col = result_df[name].dropna()
        return round(float(col.mean()), 4) if len(col) > 0 else 0.0

    scores: dict[str, float] = {
        "faithfulness":      _col_mean("faithfulness"),
        "context_precision": _col_mean("context_precision"),
        "context_recall":    _col_mean("context_recall"),
    }
    return scores


def run_evaluation(
    engine: RAGEngine,
    questions: list[str],
    company: str,
    chat_model: str,
    top_k: int,
    output_path: Path,
    run_ragas: bool = False,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH,
    api_key: str = "",
    base_url: str = DEFAULT_ALBERT_BASE_URL,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "company_filter",
        "question",
        "answer",
        "source_count",
        "citations",
        "top_score",
        "source_companies",
        # Ragas metric columns — populated only when --ragas is used
        # (answer_relevancy omitted: requires /embeddings which is unstable on Albert API)
        "faithfulness",
        "context_precision",
        "context_recall",
    ]

    rows: list[dict] = []
    total = len(questions)

    for idx, question in enumerate(questions, 1):
        print(f"[{idx}/{total}] {question[:80]}…")
        try:
            answer, sources = engine.answer(
                question,
                top_k=top_k,
                company=company,
                chat_model=chat_model,
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("Error on question %d: %s", idx, exc)
            answer = f"ERROR: {exc}"
            sources = []

        citations = " | ".join(c["citation"] for c in sources)
        top_score = sources[0]["_score"] if sources else 0.0
        source_companies = ", ".join(
            sorted({c["company"] for c in sources})
        )

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "company_filter": company or "all",
            "question": question,
            "answer": answer,
            "source_count": len(sources),
            "citations": citations,
            "top_score": round(top_score, 4),
            "source_companies": source_companies,
            # Ragas columns are empty until compute_ragas_metrics() fills them
            "faithfulness": "",
            "context_precision": "",
            "context_recall": "",
        }
        rows.append(row)

        # Print a short preview
        preview = answer.replace("\n", " ")[:120]
        print(f"    Answer: {preview}…")
        print(f"    Sources: {source_companies or 'none'} (top score: {top_score:.3f})")
        print()

    # Optionally compute Ragas quality metrics and back-fill into the rows
    if run_ragas:
        scores = compute_ragas_metrics(
            rows=rows,
            ground_truth_path=ground_truth_path,
            api_key=api_key,
            base_url=base_url,
            chat_model=chat_model,
        )
        if scores:
            # Apply the aggregate scores to every row that has a matching ground truth
            # (individual per-question scores require Ragas >=0.2 result.scores dict)
            for row in rows:
                row.update(scores)

            print("\n" + "=" * 50)
            print("RAGAS EVALUATION SUMMARY")
            print("=" * 50)
            targets = {
                "faithfulness":      0.85,
                "context_precision": 0.75,
                "context_recall":    0.70,
            }
            for metric, score in scores.items():
                target = targets.get(metric, 0.0)
                status = "PASS" if score >= target else "FAIL"
                print(f"  {metric:<22} {score:.4f}  (target >{target})  [{status}]")
            print("=" * 50)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nEvaluation complete. {len(rows)} questions evaluated.")
    print(f"Results saved to: {output_path}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate the ESG RAG pipeline on a set of CSRD questions."
    )
    parser.add_argument(
        "--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
        help="Directory containing the FAISS index files.",
    )
    parser.add_argument(
        "--company", default="",
        help="Filter questions to a specific company (e.g. 'Danone').",
    )
    parser.add_argument(
        "--questions-file", type=Path, default=None,
        help="Path to a YAML or plain-text file with one question per line.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output CSV file path.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--base-url",
        default=os.getenv("ALBERT_API_BASE_URL") or DEFAULT_ALBERT_BASE_URL,
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--ragas",
        action="store_true",
        help=(
            "Compute Ragas quality metrics after the run. "
            "Requires: pip install ragas langchain-openai datasets"
        ),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Path to the ground-truth CSV used for Ragas evaluation.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load questions
    if args.questions_file:
        if not args.questions_file.exists():
            print(f"Error: questions file not found: {args.questions_file}", file=sys.stderr)
            return 1
        questions = load_questions_yaml(args.questions_file)
        if not questions:
            print("Error: no questions found in the file.", file=sys.stderr)
            return 1
        print(f"Loaded {len(questions)} questions from {args.questions_file}")
    else:
        questions = BUILTIN_QUESTIONS
        print(f"Using {len(questions)} built-in CSRD evaluation questions.")

    api_key = resolve_api_key()
    client = AlbertClient(api_key=api_key, base_url=args.base_url)

    try:
        engine = RAGEngine(
            index_dir=args.index_dir,
            client=client,
            embedding_model=args.embedding_model,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    label = f"company={args.company}" if args.company else "all companies"
    print(f"\nRunning evaluation: {len(questions)} questions | {label} | model: {args.chat_model}")
    print(f"Output: {args.output}\n")

    run_evaluation(
        engine=engine,
        questions=questions,
        company=args.company,
        chat_model=args.chat_model,
        top_k=args.top_k,
        output_path=args.output,
        run_ragas=args.ragas,
        ground_truth_path=args.ground_truth,
        api_key=api_key,
        base_url=args.base_url,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
