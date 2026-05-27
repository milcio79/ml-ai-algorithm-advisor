# Publishing To GitHub

## Recommended Repository Name

```text
ml-ai-algorithm-advisor
```

## Recommended Description

```text
Local Streamlit app that suggests and ranks ML, AI, NLP, computer vision, time series, anomaly detection, document intelligence, GenAI, and RAG algorithms for a project.
```

## Recommended Topics

Add these in GitHub under **About > Settings gear > Topics**:

```text
machine-learning
artificial-intelligence
streamlit
data-science
mlops
nlp
computer-vision
time-series
anomaly-detection
rag
genai
openai
algorithm-selection
document-intelligence
python
```

## Before Publishing

Check that secrets are not tracked:

```powershell
git status --short
git check-ignore -v .env algorithm_advisor/.env
```

If Git fails to initialize inside a OneDrive-synced directory because of `.git` lock or permission errors, copy the project to a normal development folder such as `C:\dev\ml-ai-algorithm-advisor` and initialize Git there.

Run validation:

```powershell
python -m compileall algorithm_advisor
```

Optional smoke test:

```powershell
cd algorithm_advisor
streamlit run app.py
```

## Suggested First Commit

```powershell
git init
git add .
git status --short
git commit -m "Initial public release of ML AI Algorithm Advisor"
```

Review `git status --short` before committing. `.env` files should not appear.

## GitHub Release Notes

Suggested `v0.1.0` release title:

```text
Initial local ML / AI Algorithm Advisor
```

Suggested release summary:

```text
First public release with deterministic algorithm suggestions, optional OpenAI ranking, grouped project characteristic checkboxes, example document-intelligence workflow, and Markdown export.
```
