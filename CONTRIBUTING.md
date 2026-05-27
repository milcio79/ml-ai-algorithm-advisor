# Contributing

Thanks for considering a contribution.

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd algorithm_advisor
streamlit run app.py
```

## Contribution Guidelines

- Keep the app local-first.
- Never hardcode API keys or secrets.
- Treat unchecked project characteristics as unknown, not negative.
- Keep deterministic recommender changes explainable and auditable.
- Add new algorithm rules with clear `why_applicable`, `best_when`, `cautions`, `required_data`, `typical_metrics`, and `implementation_notes`.
- Prefer readable Python with type hints.

## Validation

Run before opening a pull request:

```powershell
python -m compileall algorithm_advisor
```

If you change dependencies, update `algorithm_advisor/requirements.txt`.

## Pull Requests

Please include:

- What changed.
- Why it changed.
- How you tested it.
- Screenshots or notes for UI changes.
- Any known limitations.

