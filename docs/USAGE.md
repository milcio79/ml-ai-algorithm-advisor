# Usage Guide

## Run The App

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd algorithm_advisor
streamlit run app.py
```

Open `http://localhost:8501`.

## Basic Workflow

1. Enter an optional project name, description, domain, and notes in the sidebar.
2. Select only the project characteristics you know.
3. Leave unknown characteristics unchecked.
4. Click **Suggest candidate algorithms**.
5. Review the deterministic candidates and expand any row for details.
6. Click **Rank with ChatGPT** only if you configured `OPENAI_API_KEY`.
7. Copy or download the Markdown export.

## Important Semantics

Unchecked boxes mean:

```text
unknown / not selected
```

Unchecked boxes do not mean:

```text
false / no / not applicable
```

Where a true negative matters, select an explicit negative checkbox:

- Explicitly no labels available
- Explicitly no need for explainability
- Explicitly no deployment required

## Example Profile

Click **Load example: Document Intelligence Compliance Workflow** in the sidebar.

This selects characteristics for scanned documents, OCR, layout-aware processing, sensitive information detection, compliance, auditability, clustering, retrieval, RAG, and human-in-the-loop review.

Then click:

1. **Suggest candidate algorithms**
2. **Rank with ChatGPT**

## Exporting Results

The app generates a Markdown export containing:

- project profile
- selected characteristics
- local warnings
- deterministic candidate algorithms
- ChatGPT ranking, if generated

Use **Download full Markdown export** to save it.

