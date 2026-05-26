
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import requests
from sklearn.feature_extraction.text import HashingVectorizer

_HERE = Path(__file__).parent.resolve()

DEFAULT_ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_API_MODEL = "BAAI/bge-m3"


def load_dotenv(path: Path = _HERE / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype("float32", copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_api_key() -> str | None:
    for env_name in (
        "EMBEDDING_API_KEY",
        "ALBERT_API_KEY",
        "OPENAI_API_KEY",
    ):
        value = os.getenv(env_name)
        if value:
            return value
    return None


class HashEmbeddingClient:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.vectorizer = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            analyzer="word",
            ngram_range=(1, 2),
            norm=None,
            lowercase=True,
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        dense = matrix.toarray().astype("float32", copy=False)
        return normalize_rows(dense)


class OpenAICompatibleEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self.session.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = sorted(payload["data"], key=lambda item: item["index"])
        matrix = np.array([item["embedding"] for item in data], dtype="float32")
        return normalize_rows(matrix)


def load_chunk_records(
    chunks_path: Path,
    companies: list[str],
    limit: int,
    min_tokens: int,
) -> list[dict]:
    records = load_jsonl(chunks_path)
    if companies:
        allowed = {slugify(company) for company in companies}
        records = [row for row in records if slugify(str(row["company"])) in allowed]
    if min_tokens > 0:
        records = [row for row in records if int(row.get("token_count_estimate", 0)) >= min_tokens]
    if limit > 0:
        records = records[:limit]
    return records


def slugify(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def get_client(args: argparse.Namespace, config: dict | None = None):
    provider = args.provider or (config or {}).get("provider") or "api"

    if provider == "hash":
        dim = args.hash_dim or (config or {}).get("embedding_dim") or 1024
        return provider, HashEmbeddingClient(dim=int(dim))

    base_url = (
        args.base_url
        or (config or {}).get("base_url")
        or os.getenv("EMBEDDING_API_BASE_URL")
        or os.getenv("ALBERT_API_BASE_URL")
        or DEFAULT_ALBERT_BASE_URL
    )
    model = args.model or (config or {}).get("model") or DEFAULT_API_MODEL
    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "No embedding API key found. Set EMBEDDING_API_KEY, ALBERT_API_KEY, or OPENAI_API_KEY."
        )
    return provider, OpenAICompatibleEmbeddingClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=args.timeout,
    )


def build_embeddings(
    records: list[dict],
    client,
    batch_size: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    total = len(records)
    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        texts = [str(row["text"]) for row in batch]
        batch_vectors = client.embed_texts(texts)
        vectors.append(batch_vectors)
        logging.info("Embedded %s/%s chunks", min(start + len(batch), total), total)
    return np.vstack(vectors).astype("float32", copy=False)


def save_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def command_build(args: argparse.Namespace) -> int:
    records = load_chunk_records(
        chunks_path=args.chunks_path,
        companies=args.companies,
        limit=args.limit,
        min_tokens=args.min_tokens,
    )
    if not records:
        logging.warning("No chunk records selected for indexing.")
        return 0

    provider, client = get_client(args)
    embeddings = build_embeddings(records, client, batch_size=args.batch_size)

    embedding_dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    index_dir = args.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))

    metadata_rows = []
    for position, row in enumerate(records):
        enriched = dict(row)
        enriched["vector_position"] = position
        metadata_rows.append(enriched)
    write_jsonl(index_dir / "metadata.jsonl", metadata_rows)

    if args.save_embeddings:
        np.save(index_dir / "embeddings.npy", embeddings)

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": args.model if provider == "api" else "hashing-vectorizer",
        "base_url": args.base_url if provider == "api" else "",
        "embedding_dim": embedding_dim,
        "similarity": "cosine_via_inner_product_on_normalized_vectors",
        "chunk_count": len(records),
        "chunks_path": str(args.chunks_path),
        "min_tokens": args.min_tokens,
        "companies": args.companies,
    }
    save_config(index_dir / "config.json", config)
    logging.info("Saved FAISS index to %s", index_dir / "faiss.index")
    logging.info("Saved metadata to %s", index_dir / "metadata.jsonl")
    logging.info("Indexed %s chunks with dim=%s", len(records), embedding_dim)
    return 0


def command_search(args: argparse.Namespace) -> int:
    index_dir = args.index_dir
    index = faiss.read_index(str(index_dir / "faiss.index"))
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    metadata = load_jsonl(index_dir / "metadata.jsonl")

    provider, client = get_client(args, config=config)
    question_embedding = client.embed_texts([args.question]).astype("float32", copy=False)
    scores, indices = index.search(question_embedding, args.top_k)

    logging.info("Provider: %s", provider)
    print(f"Question: {args.question}\n")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        row = metadata[int(idx)]
        excerpt = str(row["text"]).replace("\n", " ").strip()
        excerpt = excerpt[: args.preview_chars].rstrip()
        print(f"[{rank}] score={score:.4f}")
        print(f"chunk_id: {row['chunk_id']}")
        print(f"citation: {row['citation']}")
        print(f"section: {row.get('section_title', '')}")
        print(f"text: {excerpt}")
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query a vector index for ESG chunks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Embed chunks and build a FAISS index.")
    build.add_argument("--chunks-path", type=Path, default=_HERE / "data/chunks/all_chunks.jsonl")
    build.add_argument("--index-dir", type=Path, default=_HERE / "data/vector_index_api")
    build.add_argument("--provider", choices=("api", "hash"), default="api")
    build.add_argument("--model", default=os.getenv("EMBEDDING_MODEL") or DEFAULT_API_MODEL)
    build.add_argument("--base-url", default=os.getenv("EMBEDDING_API_BASE_URL") or os.getenv("ALBERT_API_BASE_URL") or DEFAULT_ALBERT_BASE_URL)
    build.add_argument("--timeout", type=int, default=60)
    build.add_argument("--batch-size", type=int, default=32)
    build.add_argument("--companies", nargs="*", default=[])
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--min-tokens", type=int, default=200)
    build.add_argument("--hash-dim", type=int, default=1024)
    build.add_argument("--save-embeddings", action="store_true")
    build.set_defaults(func=command_build)

    search = subparsers.add_parser("search", help="Query an existing FAISS index.")
    search.add_argument("question")
    search.add_argument("--index-dir", type=Path, default=_HERE / "data/vector_index_api")
    search.add_argument("--provider", choices=("api", "hash"), default="")
    search.add_argument("--model", default="")
    search.add_argument("--base-url", default="")
    search.add_argument("--timeout", type=int, default=60)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--preview-chars", type=int, default=500)
    search.add_argument("--hash-dim", type=int, default=1024)
    search.set_defaults(func=command_search)

    return parser


def main(argv: list[str]) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
