Albert API & RAG System - Student Class Workflow
This guide outlines the steps students need to complete for the Albert API and RAG system class.

Prerequisites
Python 3.8+

Required Python packages:

pip install requests openai
Albert API Access:

Albert API key (set as ALBERT_API_KEY environment variable)
Albert API base URL (set as ALBERT_BASE_URL environment variable, defaults to https://albert.api.etalab.gouv.fr/v1)

Useful Links
Class GitHub Repo: https://github.com/RamiELM/FTD_2026_ESG_AI_Assistant
IDE: Visual Studio Code
Coding assistance: Continue Dev
Vibe Coding: OpenCode.ai
Albert API:
API Documentation
Incubateur Documentation
RAG Frameworks:
LangChain
LlamaIndex
Sample Dataset: S&P 500 ESG Reports on Kaggle
WebUI Tools:
Streamlit
Gradio
Open WebUI
RAG Evaluation:
Ragas Documentation
DeepEval GitHub
Step-by-Step Workflow
1. Test Albert API Connection
First, verify your connection to the Albert API:

python albert_test.py
This will:

List available models
Test chat completion functionality
Confirm API connectivity
2. Test RAG System
Test the basic Retrieval-Augmented Generation system:

python rag_test.py
This will:

Create a new collection called "tutorial"
Upload a PDF document (totalenergies_sustainability-climate-2024-progress-report_2024_en_pdf.pdf)
Allow you to ask questions about the document content
3. Download Company Reports
Run the company RAG system to download sustainability reports:

python company_rag_test.py
This will:

Download PDF reports from various companies
Create a collection named "company_reports"
Upload all PDF documents to the collection
Enable CSRD-related question answering
4. Test CSRD Questions
After downloading the PDFs, you can ask CSRD-related questions about each company:

Run python company_rag_test.py
When prompted, enter your CSRD-related question
Specify which company you're asking about
Get AI-powered answers based on the sustainability reports
Key Files Explained
albert_test.py: Basic API connectivity test
rag_test.py: Simple RAG system with one document
company_rag_test.py: Advanced RAG system with multiple company documents
PDF files: Sustainability reports from various companies (need to be downloaded manually or via the script)
Expected Outcomes
By completing this class:

Students will understand Albert API fundamentals
Students will learn RAG system implementation
Students will be able to ask CSRD-related questions on company sustainability reports
Students will gain experience with document indexing and semantic search