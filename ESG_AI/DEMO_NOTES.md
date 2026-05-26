# DEMO_NOTES.md — ESG AI Assistant

Notes for presenting or demonstrating the project.

---

## What Has Been Completed

| Component | Status | Notes |
|---|---|---|
| ESG report scraper | ✅ Done | 5 companies scraped |
| PDF parser | ✅ Done | Page-level text extraction |
| Text chunker | ✅ Done | ~400-token overlapping chunks |
| FAISS vector index | ✅ Done | 7,154 chunks, BAAI/bge-m3 |
| CLI chatbot (`csrd_chat.py`) | ✅ Done | Albert API LLM generation |
| Streamlit web UI (`app.py`) | ✅ Done | Company filter, chat history, sources |
| Evaluation script | ✅ Done | 12 CSRD questions, CSV output |
| Documentation | ✅ Done | README, DEMO_NOTES, .env.example |

---

## How the Assistant Works

1. **User asks a question** (via CLI or Streamlit UI).
2. **Embedding**: The question is embedded with the same model used to build the index (BAAI/bge-m3 via Albert API).
3. **Retrieval**: FAISS searches the local vector index for the top-k most semantically similar text chunks (cosine similarity).
4. **Augmented prompt**: The retrieved chunks are formatted as numbered context blocks and prepended to the user question.
5. **Generation**: The Albert API LLM (Mistral Small by default) receives the system prompt + context + question and generates a grounded answer.
6. **Citations**: The answer cites specific chunks as [1], [2], etc., and the UI shows the source report, section, relevance score, and excerpt.

---

## Data Used

| Company | Reports Indexed |
|---|---|
| Airbus | Annual Report 2024 |
| BNP Paribas | Integrated report 2023/2024 |
| Danone | Integrated Annual Report 2024, URD 2020 |
| ENGIE | Sustainability Report 2023/2024 |
| TotalEnergies | Sustainability & Climate Progress Report 2024 |

**Total:** 7,154 chunks, embeddings dimension 1024, similarity metric: cosine via inner product on normalized vectors.

---

## What Questions Can Be Asked

The assistant works best on structured CSRD topics:

- **Climate & GHG**: "What are the Scope 1, 2, and 3 emission targets?" / "What is the net-zero commitment?"
- **Biodiversity**: "How does Danone address biodiversity risks?"
- **Social / Workforce**: "What employee wellbeing programs are described?"
- **Governance**: "What board committees oversee ESG?"
- **Water & Circular Economy**: "What water targets has ENGIE set?"
- **CSRD compliance**: "Has a double materiality assessment been conducted?"
- **Energy**: "What renewable energy targets are reported?"
- **Supply chain**: "How are Scope 3 value chain emissions managed?"

---

## How Citations Are Generated

Each chunk stored in the index carries a `citation` field built at chunking time:

```
Danone | danone_danoneiar2024.pdf | pages 15-16
```

This is formatted from: `company | pdf_filename | page range`.

In the CLI output, sources appear as numbered list `[1] citation (score: 0.688)`.
In the Streamlit UI, each source is expandable and shows the relevance score and raw text excerpt.

---

## Known Limitations

1. **Index coverage**: Only 5 companies are indexed. Questions about other companies return no results.
2. **Temporal scope**: Reports from 2020–2024 only. The assistant cannot answer questions about more recent events.
3. **PDF quality**: Scanned-only pages have no extractable text and are skipped during parsing.
4. **Language**: All reports are in English. French-language sections may not parse cleanly.
5. **Hallucination guard**: The system prompt instructs the model not to hallucinate, but this is a soft constraint. Always verify critical data in the original reports.
6. **API dependency**: Both embedding and LLM generation require a live connection to the Albert API.

---

## Suggested Demo Flow (3–5 Minutes)

### Minute 1 — Introduction
- Explain the problem: ESG reports are long, dense, and hard to search manually.
- Show the `data/` folder: 5 company folders, ~7,000 chunks already indexed.

### Minute 2 — CLI demo
Run in terminal:
```bash
python csrd_chat.py -q "What are TotalEnergies' GHG reduction targets?" --company "TotalEnergies"
```
Point out the sourced answer with page citations.

### Minute 3 — Streamlit demo
```bash
streamlit run app.py
```
- Open `http://localhost:8501`
- Set company filter to "Danone"
- Click the example question: "What are the GHG emissions reduction targets?"
- Expand the Sources panel to show the excerpt and relevance score
- Change to "All companies" and ask a cross-company question:
  "Which companies have committed to net-zero by 2050?"

### Minute 4 — Architecture explanation
Walk through the pipeline diagram in README.md (scrape → parse → chunk → embed → retrieve → generate).

### Minute 5 — Evaluation
```bash
python evaluate_rag.py --company "Danone"
```
Show the output CSV in `data/evaluation/`. Discuss what the scores mean.

---

## Quick Start (for a professor or reviewer)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env: ALBERT_API_KEY=your_key

# 3. Launch the UI (index already built — no need to re-scrape)
streamlit run app.py

# 4. Or use the CLI
python csrd_chat.py -q "What are Danone's net-zero commitments?"
```
