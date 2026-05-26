"""
ESG CSRD RAG Chat — Q&A over indexed sustainability reports via Albert API.

Usage:
    python csrd_chat.py                              # interactive, all companies
    python csrd_chat.py --company "Danone"           # filter to one company
    python csrd_chat.py -q "What are TotalEnergies GHG targets?"
    python csrd_chat.py --top-k 8 --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import requests

_HERE = Path(__file__).parent.resolve()

DEFAULT_ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_CHAT_MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
DEFAULT_INDEX_DIR = _HERE / "data/vector_index_api"
DEFAULT_TOP_K = 6

SYSTEM_PROMPT = (
    "You are an expert ESG analyst specializing in CSRD (Corporate Sustainability "
    "Reporting Directive) compliance and sustainability reporting.\n\n"
    "Rules you must always follow:\n"
    "1. Answer ONLY from the document excerpts provided below. Do not use any external "
    "knowledge or make up information.\n"
    "2. Cite sources as [1], [2], etc. whenever you reference specific data.\n"
    "3. If the retrieved excerpts do not contain enough information to answer the question, "
    'say clearly: "The indexed reports do not contain sufficient information to answer '
    'this question."\n'
    "4. Keep answers concise, professional, and CSRD/ESG-focused.\n"
    "5. Never speculate, extrapolate, or fill gaps with general knowledge."
)


def load_dotenv(path: Path = _HERE / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_api_key() -> str:
    for name in ("ALBERT_API_KEY", "EMBEDDING_API_KEY", "OPENAI_API_KEY"):
        val = os.getenv(name, "").strip()
        if val:
            return val
    raise RuntimeError("No API key found. Set ALBERT_API_KEY in your .env file.")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype("float32", copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class AlbertClient:
    """Thin wrapper around the Albert OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_ALBERT_BASE_URL,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def embed(self, texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
        resp = self.session.post(
            f"{self.base_url}/embeddings",
            json={"model": model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        matrix = np.array([x["embedding"] for x in data], dtype="float32")
        return normalize_rows(matrix)

    def chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_CHAT_MODEL,
        temperature: float = 0.1,
    ) -> str:
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def stream_chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_CHAT_MODEL,
        temperature: float = 0.1,
    ):
        """Yield response tokens one at a time via SSE streaming."""
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json={"model": model, "messages": messages, "temperature": temperature, "stream": True},
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            # SSE lines are: b"data: {...}" or b"data: [DONE]"
            line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                token = json.loads(payload)["choices"][0]["delta"].get("content", "")
                if token:
                    yield token
            except Exception:
                continue

    def list_models(self) -> list[str]:
        resp = self.session.get(f"{self.base_url}/models", timeout=30)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]


def _require_index(index_dir: Path) -> None:
    """Raise a clear error if any required index file is missing."""
    index_dir = Path(index_dir)  # accept str as well
    required = ["faiss.index", "metadata.jsonl", "config.json"]
    missing = [f for f in required if not (index_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"\n\nIndex files not found in '{index_dir}':\n"
            + "".join(f"  - {f}\n" for f in missing)
            + "\nBuild the index first:\n"
            "  python vector_index.py build --index-dir data/vector_index_api --provider api\n"
        )


class RAGEngine:
    """Retrieval-Augmented Generation over indexed ESG reports."""

    def __init__(
        self,
        index_dir: Path,
        client: AlbertClient,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        _require_index(index_dir)
        self.client = client
        self.embedding_model = embedding_model

        config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.metadata = load_jsonl(index_dir / "metadata.jsonl")
        self.companies = sorted({row["company"] for row in self.metadata})
        self.index_config = config

        # Warn if the embedding model at query time differs from build time
        built_with = config.get("model", "")
        if built_with and embedding_model != built_with:
            logging.warning(
                "Embedding model mismatch: index built with '%s', querying with '%s'. "
                "Results may be inaccurate. Pass --embedding-model %s to align them.",
                built_with,
                embedding_model,
                built_with,
            )
            self.embedding_model = built_with  # silently use the correct model

        logging.info(
            "Index loaded: %d chunks | model=%s | companies=%s",
            config["chunk_count"],
            config["model"],
            self.companies,
        )

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        company: str = "",
    ) -> list[dict]:
        q_vec = self.client.embed([question], model=self.embedding_model)
        # When filtering by company, search the full index so no chunks are missed
        # due to low global ranking. FAISS handles 8k+ vectors in <1ms.
        fetch_k = len(self.metadata) if company else min(top_k, len(self.metadata))
        scores, indices = self.index.search(q_vec, fetch_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = dict(self.metadata[int(idx)])
            row["_score"] = float(score)
            if company and slugify(row["company"]) != slugify(company):
                continue
            results.append(row)
            if len(results) >= top_k:
                break
        return results

    def answer(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        company: str = "",
        chat_model: str = DEFAULT_CHAT_MODEL,
    ) -> tuple[str, list[dict]]:
        chunks = self.retrieve(question, top_k=top_k, company=company)
        if not chunks:
            return "No relevant passages found in the indexed reports for that question.", []
        messages = self._build_messages(question, chunks)
        answer_text = self.client.chat(messages, model=chat_model)
        return answer_text, chunks

    def _build_messages(self, question: str, chunks: list[dict]) -> list[dict]:
        """Shared helper: build the RAG prompt messages from retrieved chunks."""
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            context_blocks.append(
                f"[{i}] {chunk['citation']}\n"
                f"Section: {chunk.get('section_title', 'N/A')}\n\n"
                f"{chunk['text'].strip()}"
            )
        context = "\n\n---\n\n".join(context_blocks)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"},
        ]

    def stream_answer(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        company: str = "",
        chat_model: str = DEFAULT_CHAT_MODEL,
    ):
        """
        Returns (sources, token_generator).
        Retrieval is blocking (fast); LLM generation streams token by token.
        Usage in Streamlit:
            sources, stream = engine.stream_answer(question, ...)
            answer = st.write_stream(stream)
        """
        chunks = self.retrieve(question, top_k=top_k, company=company)
        if not chunks:
            return [], iter(["No relevant passages found in the indexed reports for that question."])
        messages = self._build_messages(question, chunks)
        return chunks, self.client.stream_chat(messages, model=chat_model)


def print_answer(answer: str, chunks: list[dict], show_sources: bool = True) -> None:
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)
    if show_sources and chunks:
        print("\n" + "-" * 40)
        print("SOURCES")
        print("-" * 40)
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}] {chunk['citation']}  (score: {chunk['_score']:.3f})")
    print()


def interactive_loop(
    engine: RAGEngine,
    company: str,
    chat_model: str,
    top_k: int,
    show_sources: bool = True,
) -> None:
    label = f"company={company}" if company else "all companies"
    print(f"\nESG CSRD Q&A  |  {label}  |  model: {chat_model}  |  top-k: {top_k}")
    print("Type your question and press Enter. Type 'quit' or Ctrl+C to exit.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        print("Searching and generating answer…")
        answer, chunks = engine.answer(
            question, top_k=top_k, company=company, chat_model=chat_model
        )
        print_answer(answer, chunks, show_sources=show_sources)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="RAG-based CSRD Q&A over indexed ESG sustainability reports."
    )
    parser.add_argument(
        "--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
        help="Directory containing faiss.index, metadata.jsonl, config.json",
    )
    parser.add_argument(
        "--company", default="",
        help="Filter results to a single company (e.g. 'Danone', 'ENGIE')",
    )
    parser.add_argument(
        "--question", "-q", default="",
        help="Ask a single question non-interactively and exit",
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
        "--no-sources", action="store_true",
        help="Hide source citations in the output.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    api_key = resolve_api_key()
    client = AlbertClient(api_key=api_key, base_url=args.base_url)
    engine = RAGEngine(
        index_dir=args.index_dir,
        client=client,
        embedding_model=args.embedding_model,
    )

    show_sources = not args.no_sources
    if args.question:
        answer, chunks = engine.answer(
            args.question,
            top_k=args.top_k,
            company=args.company,
            chat_model=args.chat_model,
        )
        print_answer(answer, chunks, show_sources=show_sources)
    else:
        interactive_loop(engine, args.company, args.chat_model, args.top_k, show_sources=show_sources)

    return 0


if __name__ == "__main__":
    sys.exit(main())
