"""
Albert API Connection Test — Step 1 of the ESG AI Assistant workflow.

Verifies that your ALBERT_API_KEY is set and the API is reachable.

Usage:
    python albert_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent.resolve()


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


def main() -> int:
    load_dotenv()

    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' package is not installed. Run: pip install requests")
        return 1

    api_key = os.getenv("ALBERT_API_KEY", "").strip()
    if not api_key:
        print("ERROR: ALBERT_API_KEY is not set.")
        print("  Copy .env.example to .env and add your key.")
        return 1

    base_url = os.getenv("ALBERT_API_BASE_URL", "https://albert.api.etalab.gouv.fr/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    print(f"Albert API Test")
    print(f"Base URL : {base_url}")
    print(f"API Key  : {api_key[:8]}{'*' * (len(api_key) - 8)}")
    print()

    # ── 1. List available models ──────────────────────────────────────────────
    print("1. Listing available models...")
    try:
        resp = requests.get(f"{base_url}/models", headers=headers, timeout=15)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", [])]
        print(f"   OK — {len(models)} models available:")
        for m in models:
            print(f"      - {m}")
    except Exception as exc:
        print(f"   FAILED: {exc}")
        return 1

    print()

    # ── 2. Test chat completion ───────────────────────────────────────────────
    chat_model = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    print(f"2. Testing chat completion ({chat_model})...")
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": "Reply with exactly: API connection successful."}],
                "temperature": 0.0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"   OK — Model reply: {reply}")
    except Exception as exc:
        print(f"   FAILED: {exc}")
        return 1

    print()

    # ── 3. Test embeddings ────────────────────────────────────────────────────
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    print(f"3. Testing embeddings ({embedding_model})...")
    try:
        resp = requests.post(
            f"{base_url}/embeddings",
            headers=headers,
            json={"model": embedding_model, "input": ["ESG sustainability report test"]},
            timeout=30,
        )
        resp.raise_for_status()
        dim = len(resp.json()["data"][0]["embedding"])
        print(f"   OK — Embedding dimension: {dim}")
    except Exception as exc:
        print(f"   FAILED: {exc}")
        return 1

    print()
    print("All tests passed. Albert API is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
