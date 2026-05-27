# ML / AI Algorithm Advisor

A local Streamlit application that helps you explore potentially appropriate algorithms for machine learning, AI, data science, NLP, document intelligence, computer vision, time series, anomaly detection, optimization, GenAI, and RAG projects.

The app has two separate steps:

1. **Suggest candidate algorithms** uses local deterministic rules only.
2. **Rank with ChatGPT** sends the selected project profile and deterministic candidates to the OpenAI Responses API for a ranked Markdown recommendation.

Unchecked boxes mean **unknown / not selected**, never "No". If a negative answer matters, use the explicit negative options such as "Explicitly no labels available", "Explicitly no need for explainability", or "Explicitly no deployment required".

## Setup

From this folder:

```powershell
cd "D:\OneDrive\Proyectos DS\KpiChat\model_selection\algorithm_advisor"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer using the project-level virtual environment that already exists one folder above:

```powershell
cd "D:\OneDrive\Proyectos DS\KpiChat\model_selection\algorithm_advisor"
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.5
```

Or set the key for the current Windows PowerShell session:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_MODEL="gpt-5.5"
```

Security note: never commit `.env` or API keys. The app reads secrets from environment variables using `python-dotenv`; no key is hardcoded.

## Run

```powershell
streamlit run app.py
```

If Streamlit is not on your PATH:

```powershell
python -m streamlit run app.py
```

## How The Deterministic Recommender Works

`recommender.py` contains a catalog of algorithm families with matching triggers. The app compares your selected checkboxes against those triggers, computes a local confidence score from 0 to 100, and adds contextual cautions for risk, thresholds, deployment, explainability, data scale, and ambiguity.

This first step never calls OpenAI.

## How ChatGPT Ranking Works

`openai_client.py` calls the OpenAI Responses API with:

- project description
- selected checkbox options grouped by section
- a note that unchecked options are unknown
- deterministic candidate algorithms and their local reasons

The prompt asks ChatGPT to return Markdown with an executive recommendation, ranked table, baselines, advanced options, avoided options, data requirements, metrics, implementation steps, risks, first experiment, production path, and clarifying questions.

The OpenAI call happens only when you click **Rank with ChatGPT**.

If `OPENAI_API_KEY` is missing, deterministic recommendations still work and the ChatGPT ranking button shows a clear error.

## Example Profile

Click **Load example: Document Intelligence Compliance Workflow** in the sidebar. Then:

1. Click **Suggest candidate algorithms**.
2. Review the deterministic table and expand algorithm details.
3. Click **Rank with ChatGPT** if `OPENAI_API_KEY` is configured.
4. Use the Markdown text area or download button to export the full profile and recommendations.
