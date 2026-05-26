# ESG AI Assistant for CSRD Q&A

A Retrieval-Augmented Generation (RAG) assistant that answers CSRD and ESG questions grounded in real company sustainability reports. Built with the Albert API (French government LLM infrastructure), FAISS, and Streamlit.

---

## Project Objective

This project builds an end-to-end AI assistant that:

1. Downloads ESG / sustainability reports as PDFs from company websites
2. Parses and cleans the text from each page
3. Chunks the text into semantically coherent segments
4. Embeds the chunks with the Albert API (BAAI/bge-m3) and stores them in a local FAISS vector index
5. Answers natural-language CSRD/ESG questions by retrieving the most relevant chunks and generating grounded answers with an LLM

---

## Architecture Overview

```
Company websites
     │
     ▼
esg_report_scraper.py  ──▶  data/raw_pdfs/  +  data/pdf_metadata.csv
     │
     ▼
parse_esg_reports.py   ──▶  data/parsed_pages/  +  data/parsed_pages_manifest.csv
     │
     ▼
chunk_esg_reports.py   ──▶  data/chunks/all_chunks.jsonl  +  data/chunks_manifest.csv
     │
     ▼
vector_index.py build  ──▶  data/vector_index_api/  (faiss.index, metadata.jsonl, config.json)
     │
     ▼
csrd_chat.py           ──▶  CLI Q&A with source citations
app.py                 ──▶  Streamlit web interface
evaluate_rag.py        ──▶  Batch evaluation → data/evaluation/rag_eval_results.csv
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/RamiELM/FTD_2026_ESG_AI_Assistant
cd FTD_2026_ESG_AI_Assistant
```

### 2. Create and activate a Python environment

```bash
conda create -n esg-ai python=3.11
conda activate esg-ai
```

Or with venv:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your Albert API key:

```
ALBERT_API_KEY=your_api_key_here
ALBERT_API_BASE_URL=https://albert.api.etalab.gouv.fr/v1
EMBEDDING_MODEL=BAAI/bge-m3
```

---

## Pipeline Steps

### Step 1 — Scrape and download ESG reports

```bash
python esg_report_scraper.py
```

- Visits company investor-relations pages
- Scores and selects the best ESG/sustainability PDF for each company
- Downloads PDFs to `data/raw_pdfs/<company>/`
- Writes `data/pdf_metadata.csv`

To scrape only specific companies:

```bash
python esg_report_scraper.py --companies danone engie
```

### Step 2 — Parse PDFs to clean text

```bash
python parse_esg_reports.py
```

- Reads `data/pdf_metadata.csv`
- Extracts text page by page, removes headers/footers, detects section headings
- Writes JSONL files to `data/parsed_pages/<company>/`
- Writes `data/parsed_pages_manifest.csv`

### Step 3 — Chunk the text

```bash
python chunk_esg_reports.py
```

- Reads parsed page JSONL files
- Splits text into overlapping chunks of ~400 tokens
- Writes `data/chunks/all_chunks.jsonl` and `data/chunks_manifest.csv`

### Step 4 — Build the FAISS vector index

```bash
python vector_index.py build --index-dir data/vector_index_api --provider api
```

- Embeds all chunks using the Albert API (BAAI/bge-m3)
- Builds a FAISS inner-product index (cosine similarity on normalized vectors)
- Writes `data/vector_index_api/faiss.index`, `metadata.jsonl`, `config.json`

For a quick test with local hash embeddings (no API needed):

```bash
python vector_index.py build --provider hash --index-dir data/vector_index_hash
```

### Step 5 — Ask CSRD questions (CLI)

```bash
# Interactive session
python csrd_chat.py

# Single question
python csrd_chat.py -q "What are the GHG emissions reduction targets?"

# Filter to one company
python csrd_chat.py --company "Danone" -q "What are Danone's net-zero targets?"

# More source passages, hide citations
python csrd_chat.py --top-k 8 --no-sources
```

### Step 6 — Launch the Streamlit web UI

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

Features:
- Company filter in the sidebar
- Model selector (Mistral Small, GPT-OSS 120B, Ministral 3.8B)
- Adjustable top-k (number of retrieved passages)
- Expandable source citations with relevance scores and excerpts
- Example CSRD questions
- Chat history

### Step 7 — Run evaluation

```bash
# Run 12 built-in CSRD questions across all companies
python evaluate_rag.py

# Filter to one company
python evaluate_rag.py --company "Danone"

# Use a custom question file (YAML or plain text)
python evaluate_rag.py --questions-file my_questions.yaml
```

Results are saved to `data/evaluation/rag_eval_results.csv`.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ALBERT_API_KEY` | Yes | — | Your Albert API key |
| `ALBERT_API_BASE_URL` | No | `https://albert.api.etalab.gouv.fr/v1` | API endpoint |
| `EMBEDDING_MODEL` | No | `BAAI/bge-m3` | Embedding model name |

---

## Example Questions

```
What are the GHG emissions reduction targets (Scope 1, 2, 3)?
What net-zero or carbon neutrality commitments have been made?
How does the company address biodiversity risks?
What water management targets are disclosed?
What circular economy initiatives are reported?
What social commitments are made for employees?
How does governance oversight of ESG work?
What CSRD double materiality assessment has been done?
What climate-related risks and opportunities are disclosed?
```

---

## Expected Folder Structure

```
ESG AI/
├── .env                          # Your API key (not committed)
├── .env.example                  # Template
├── requirements.txt
├── README.md
├── DEMO_NOTES.md
├── esg_report_scraper.py
├── parse_esg_reports.py
├── chunk_esg_reports.py
├── vector_index.py
├── csrd_chat.py
├── app.py
├── evaluate_rag.py
└── data/
    ├── pdf_metadata.csv
    ├── parsed_pages_manifest.csv
    ├── chunks_manifest.csv
    ├── raw_pdfs/
    │   ├── airbus/
    │   ├── bnp-paribas/
    │   ├── danone/
    │   ├── engie/
    │   └── totalenergies/
    ├── parsed_pages/
    │   └── <company>/*.jsonl
    ├── chunks/
    │   ├── all_chunks.jsonl
    │   └── <company>/*.jsonl
    ├── vector_index_api/
    │   ├── config.json
    │   ├── faiss.index
    │   └── metadata.jsonl
    └── evaluation/
        └── rag_eval_results.csv
```

---

## Known Limitations

- The scraper may fail if a company changes its website structure. Re-run with `--companies <name>` after updating the target URL in the scraper.
- The Albert API has rate limits. If embedding fails mid-index, re-run with `--limit` to process in batches.
- PDF parsing quality depends on the PDF (scanned documents with no text layer are skipped).
- The LLM can only answer based on the indexed reports. Questions about companies not in the index will return no results.
- Embedding is done on Albert's servers; an internet connection is required.

---

## Future Improvements

- Add support for more companies (S&P 500 ESG dataset from Kaggle)
- Add a reranker step using `BAAI/bge-reranker-v2-m3` to improve retrieval precision
- Add streaming LLM responses in the Streamlit UI
- Add RAG evaluation metrics (faithfulness, answer relevance) with Ragas or DeepEval
- Support multilingual reports (French, German)
- Add document-level filtering by year or report type
