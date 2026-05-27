# Architecture

## Overview

The app is a local Streamlit application with two recommendation layers:

1. A deterministic rule-based recommender.
2. An optional OpenAI ranking layer.

The deterministic layer is always available. The OpenAI layer is only called when the user clicks **Rank with ChatGPT**.

## Modules

```text
algorithm_advisor/app.py
```

Streamlit UI, session state, checkbox rendering, actions, tables, Markdown export, and example profile loading.

```text
algorithm_advisor/models.py
```

Dataclasses for `ProjectProfile` and `AlgorithmRecommendation`.

```text
algorithm_advisor/recommender.py
```

Rule-based catalog and scoring logic. Each algorithm has triggers and explanatory metadata.

```text
algorithm_advisor/prompts.py
```

Builds the prompt sent to OpenAI for ranking.

```text
algorithm_advisor/openai_client.py
```

Loads environment variables and calls the OpenAI Responses API.

```text
algorithm_advisor/utils.py
```

Markdown export helpers and small utility functions.

## Deterministic Recommender

The recommender receives a `ProjectProfile`, extracts selected checkbox labels, and matches them against algorithm triggers.

Each candidate includes:

- name
- category
- why applicable
- best when
- cautions
- required data
- typical metrics
- implementation notes
- confidence score from 0 to 100

Scores are boosted or reduced based on selected project constraints such as explainability, production deployment, small data, high accuracy, low-code preference, and error cost.

## OpenAI Ranking

The ranking prompt includes:

- project metadata
- selected checkboxes grouped by section
- warning that unchecked boxes are unknown
- deterministic candidate algorithms
- instructions to avoid inventing missing information

The OpenAI client uses:

```python
client.responses.create(...)
```

The local environment may define placeholder proxy variables. The client uses `httpx.Client(trust_env=False)` so broken local proxy settings do not block OpenAI requests.

## State Management

Streamlit session state stores:

- `selected_options`
- `project_profile`
- `candidate_algorithms`
- `openai_ranking_markdown`

## Privacy Boundary

Local deterministic recommendations do not send data outside the machine.

OpenAI ranking sends the project profile and candidate algorithm list to OpenAI only after the user clicks **Rank with ChatGPT**.

