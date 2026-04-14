#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime, timezone

import requests
from openai import OpenAI


BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
PDF_PATH = "../sample_data/totalenergies_sustainability-climate-2024-progress-report_2024_en_pdf.pdf"


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Missing environment variable: {name}")
    return v


def create_collection(api_key: str) -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    payload = {
        "name": f"totalenergies-sustainability-climate-2024-{ts}",
        "visibility": "private",
        "description": "Temp collection for extracting Scope 1/2 emissions from the 2024 progress report PDF.",
    }
    resp = requests.post(
        f"{BASE_URL}/collections",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return int(resp.json()["id"])


def upload_document(api_key: str, collection_id: int, pdf_path: str) -> int:
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/documents",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
            data={
                "collection_id": str(collection_id),
                "chunk_size": "2048",
                "chunk_overlap": "200",
                "metadata": json.dumps(
                    {
                        "source": "totalenergies_sustainability-climate-2024-progress-report_2024_en_pdf.pdf",
                        "year": 2024,
                    }
                ),
            },
            timeout=600,
        )
    resp.raise_for_status()
    return int(resp.json()["id"])


def search_chunks(api_key: str, collection_id: int, query: str, *, method: str = "hybrid", limit: int = 8):
    resp = requests.post(
        f"{BASE_URL}/search",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "query": query,
            "collection_ids": [collection_id],
            "method": method,
            "limit": limit,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def build_context(results) -> str:
    chunks = []
    for r in results:
        ch = r.get("chunk", {}) or {}
        content = (ch.get("content") or "").strip()
        if content:
            chunks.append(content)
    return "\n\n---\n\n".join(chunks)


def answer_with_chat(api_key: str, model: str, question: str, context: str) -> str:
    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    prompt = (
        "Answer ONLY using the excerpts below. If the exact Scope 1 and Scope 2 numbers "
        "for the latest reporting year are not explicitly present, say you cannot find them "
        "in the provided excerpts.\n\n"
        f"Question: {question}\n\n"
        f"Excerpts:\n{context}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise analyst. Quote the relevant excerpt snippets when giving numbers."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return resp.choices[0].message.content or ""


def main():
    api_key = _require_env("ALBERT_API_KEY")
    model = os.environ.get("ALBERT_MODEL", DEFAULT_MODEL)

    if not os.path.exists(PDF_PATH):
        raise SystemExit(f"PDF not found at: {os.path.abspath(PDF_PATH)}")

    question = "What are the company's Scope 1 and Scope 2 GHG emissions for the latest reporting year?"
    query = "Scope 1 Scope 2 GHG emissions latest reporting year tCO2e"

    print(f"Using model: {model}")
    print("Creating collection...")
    collection_id = create_collection(api_key)
    print(f"Collection id: {collection_id}")

    print("Uploading document (this can take a bit)...")
    document_id = upload_document(api_key, collection_id, PDF_PATH)
    print(f"Document id: {document_id}")

    print("Searching for relevant chunks...")
    results = []
    for attempt in range(1, 11):
        results = search_chunks(api_key, collection_id, query, method="hybrid", limit=10)
        if results:
            break
        time.sleep(2)
        print(f"Search retry {attempt}/10 (waiting for indexing)...")

    if not results:
        raise SystemExit("No search results returned (indexing may still be running). Try again in a minute.")

    context = build_context(results)
    print("\n=== TOP EXCERPTS (for traceability) ===\n")
    print(context[:12000])
    if len(context) > 12000:
        print("\n[...truncated excerpts...]\n")

    print("\n=== ANSWER ===\n")
    answer = answer_with_chat(api_key, model, question, context)
    print(answer.strip())


if __name__ == "__main__":
    main()
