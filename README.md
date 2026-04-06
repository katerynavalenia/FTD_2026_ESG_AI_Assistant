# FTD_2026_ESG_AI_Assistant

# Albert API & RAG System - Student Class Workflow

This guide outlines the steps students need to complete for the Albert API and RAG system class.

## Prerequisites

1. Python 3.8+
2. Required Python packages:
   ```bash
   pip install requests openai
   ```

3. Albert API Access:
   - Albert API key (set as `ALBERT_API_KEY` environment variable)
   - Albert API base URL (set as `ALBERT_BASE_URL` environment variable, defaults to `https://albert.api.etalab.gouv.fr/v1`)

## Step-by-Step Workflow

### 1. Test Albert API Connection

First, verify your connection to the Albert API:

```bash
python albert_test.py
```

This will:
- List available models
- Test chat completion functionality
- Confirm API connectivity

### 2. Test RAG System

Test the basic Retrieval-Augmented Generation system:

```bash
python rag_test.py
```

This will:
- Create a new collection called "tutorial"
- Upload a PDF document (`totalenergies_sustainability-climate-2024-progress-report_2024_en_pdf.pdf`)
- Allow you to ask questions about the document content

### 3. Download Company Reports

Run the company RAG system to download sustainability reports:

```bash
python company_rag_test.py
```

This will:
- Download PDF reports from various companies
- Create a collection named "company_reports"
- Upload all PDF documents to the collection
- Enable CSRD-related question answering

### 4. Test CSRD Questions

After downloading the PDFs, you can ask CSRD-related questions about each company:

1. Run `python company_rag_test.py`
2. When prompted, enter your CSRD-related question
3. Specify which company you're asking about
4. Get AI-powered answers based on the sustainability reports

## Key Files Explained

- `albert_test.py`: Basic API connectivity test
- `rag_test.py`: Simple RAG system with one document
- `company_rag_test.py`: Advanced RAG system with multiple company documents
- PDF files: Sustainability reports from various companies (need to be downloaded manually or via the script)

## Expected Outcomes

By completing this class:
1. Students will understand Albert API fundamentals
2. Students will learn RAG system implementation
3. Students will be able to ask CSRD-related questions on company sustainability reports
4. Students will gain experience with document indexing and semantic search