#!/usr/bin/env python3
"""
Batch pipeline:
- read company list from company_pdf_list.yaml
- download each PDF
- create a single Albert collection
- upload each PDF as a document (chunked)
- retrieve top chunks via /v1/search
- ask the same question for each document using chat completions
- output results to stdout + write JSONL artifact
"""

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from openai import OpenAI


BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_LIST_PATH = "../sample_data/company_pdf_list.yaml"
DEFAULT_OUT_PATH = "scope1_scope2_results.jsonl"
DEFAULT_DOWNLOAD_DIR = "downloads"

QUESTION = "What are the company's Scope 1 and Scope 2 GHG emissions for the latest reporting year?"


@dataclass(frozen=True)
class CompanyPDF:
    name: str
    pdf_url: str


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Missing environment variable: {name}")
    return v


def parse_company_list(yaml_text: str) -> list[CompanyPDF]:
    """
    Minimal parser for the known YAML shape:
    TARGET_COMPANIES:
      - name: Foo
        pdf_url: https://...
    """
    lines = yaml_text.splitlines()
    companies: list[CompanyPDF] = []
    cur_name: str | None = None
    cur_url: str | None = None

    def flush():
        nonlocal cur_name, cur_url
        if cur_name and cur_url:
            companies.append(CompanyPDF(name=cur_name, pdf_url=cur_url))
        cur_name, cur_url = None, None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("- "):
            # new item begins
            flush()
            line = line[2:].strip()

        if line.startswith("name:"):
            cur_name = line.split("name:", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("pdf_url:"):
            cur_url = line.split("pdf_url:", 1)[1].strip().strip('"').strip("'")

    flush()
    return companies


def safe_filename(company: str, url: str) -> str:
    parsed = urlparse(url)
    base = Path(parsed.path).name or "document.pdf"
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"
    company_slug = re.sub(r"[^a-zA-Z0-9]+", "_", company).strip("_")
    return f"{company_slug}__{base}"


def download_pdf(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return

    with requests.get(url, stream=True, timeout=240) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def create_collection(api_key: str) -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    payload = {
        "name": f"batch-scope1-scope2-{ts}",
        "visibility": "private",
        "description": "Batch extraction of Scope 1/2 emissions from sustainability reports.",
    }
    resp = requests.post(
        f"{BASE_URL}/collections",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return int(resp.json()["id"])


def upload_document(api_key: str, collection_id: int, pdf_path: Path, metadata: dict) -> int:
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/documents",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (pdf_path.name, f, "application/pdf")},
            data={
                "collection_id": str(collection_id),
                "chunk_size": "2048",
                "chunk_overlap": "200",
                "metadata": json.dumps(metadata),
            },
            timeout=900,
        )
    resp.raise_for_status()
    return int(resp.json()["id"])


def search_chunks(
    api_key: str,
    *,
    collection_id: int,
    document_id: int,
    query: str,
    method: str = "hybrid",
    limit: int = 10,
) -> list[dict]:
    resp = requests.post(
        f"{BASE_URL}/search",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "query": query,
            "collection_ids": [collection_id],
            "document_ids": [document_id],
            "method": method,
            "limit": limit,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def build_context(results: Iterable[dict], *, char_limit: int = 14000) -> str:
    parts: list[str] = []
    total = 0
    for r in results:
        ch = r.get("chunk", {}) or {}
        content = (ch.get("content") or "").strip()
        if not content:
            continue
        if total + len(content) + 10 > char_limit:
            break
        parts.append(content)
        total += len(content) + 10
    return "\n\n---\n\n".join(parts)


def answer_with_chat(api_key: str, model: str, *, company: str, context: str) -> str:
    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    prompt = (
        f"Company: {company}\n\n"
        "Answer ONLY using the excerpts below. Return:\n"
        "- latest_reporting_year\n"
        "- scope_1 (value + unit)\n"
        "- scope_2 (value + unit, specify location-based and/or market-based if present)\n"
        "- short quote(s) from excerpts that contain the numbers\n\n"
        f"Question: {QUESTION}\n\n"
        f"Excerpts:\n{context}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You extract exact numbers; if not explicitly present in excerpts, say 'not found in excerpts'.",
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip()


def main():
    api_key = _require_env("ALBERT_API_KEY")
    model = os.environ.get("ALBERT_MODEL", DEFAULT_MODEL)

    list_path = Path(os.environ.get("COMPANY_LIST", DEFAULT_LIST_PATH))
    out_path = Path(os.environ.get("OUT_PATH", DEFAULT_OUT_PATH))
    download_dir = Path(os.environ.get("DOWNLOAD_DIR", DEFAULT_DOWNLOAD_DIR))

    if not list_path.exists():
        raise SystemExit(f"Company list not found: {list_path.resolve()}")

    companies = parse_company_list(list_path.read_text(encoding="utf-8"))
    if not companies:
        raise SystemExit("No companies found in YAML (expected TARGET_COMPANIES list).")

    print(f"Using model: {model}")
    print(f"Companies: {len(companies)}")
    print("Creating Albert collection...")
    collection_id = create_collection(api_key)
    print(f"Collection id: {collection_id}")
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    for idx, c in enumerate(companies, start=1):
        print(f"[{idx}/{len(companies)}] {c.name}")
        pdf_file = download_dir / safe_filename(c.name, c.pdf_url)

        try:
            print(f"  Downloading: {c.pdf_url}")
            download_pdf(c.pdf_url, pdf_file)
            print(f"  Saved to: {pdf_file} ({pdf_file.stat().st_size} bytes)")

            print("  Uploading to Albert...")
            doc_id = upload_document(
                api_key,
                collection_id,
                pdf_file,
                metadata={"company": c.name, "source_url": c.pdf_url},
            )
            print(f"  Document id: {doc_id}")

            query = f"{c.name} Scope 1 Scope 2 GHG emissions latest reporting year tCO2e"
            print("  Searching for chunks...")
            results: list[dict] = []
            for attempt in range(1, 11):
                results = search_chunks(
                    api_key,
                    collection_id=collection_id,
                    document_id=doc_id,
                    query=query,
                    method="hybrid",
                    limit=12,
                )
                if results:
                    break
                time.sleep(2)
            context = build_context(results)

            if not context:
                answer = "not found in excerpts"
            else:
                print("  Asking question...")
                answer = answer_with_chat(api_key, model, company=c.name, context=context)

            record = {
                "company": c.name,
                "pdf_url": c.pdf_url,
                "local_pdf": str(pdf_file),
                "collection_id": collection_id,
                "document_id": doc_id,
                "question": QUESTION,
                "answer": answer,
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print("  Done.")
            print()
        except Exception as e:
            record = {
                "company": c.name,
                "pdf_url": c.pdf_url,
                "local_pdf": str(pdf_file),
                "collection_id": collection_id,
                "document_id": None,
                "question": QUESTION,
                "error": str(e),
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  ERROR: {e}")
            print()

    print(f"All done. Results written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

