from __future__ import annotations

import json

from models import AlgorithmRecommendation, ProjectProfile


def build_ranking_prompt(
    project_profile: ProjectProfile,
    candidate_algorithms: list[AlgorithmRecommendation],
) -> str:
    profile_json = json.dumps(project_profile.to_dict(), indent=2, ensure_ascii=False)
    candidates_json = json.dumps(
        [candidate.to_dict() for candidate in candidate_algorithms],
        indent=2,
        ensure_ascii=False,
    )

    return f"""
You are a senior applied machine learning solution architect. Your job is to rank algorithmic approaches for a project based only on the project characteristics provided.

Important interpretation rules:
- Do not assume that unchecked items are false; treat unchecked items as unknown.
- If information is missing, state what is missing and how it affects the recommendation.
- Do not invent project facts, labels, data volumes, compliance rules, or deployment constraints.
- Explicitly state assumptions before making recommendations.
- Prefer practical, production-oriented solutions over research-only ideas.
- Consider explainability, data quality, label availability, operational risk, scalability, maintainability, and human review needs.

Project profile:
```json
{profile_json}
```

Unknown/unanswered note:
Any checkbox that is not present under selected_options should be treated as unknown / not selected, never as a negative answer.

Deterministic candidate algorithms with local reasons:
```json
{candidates_json}
```

Return Markdown with these sections:
1. Executive recommendation
2. Assumptions
3. Ranked table from most suitable to least suitable
4. Why each algorithm is ranked there
5. Baseline algorithms to run first
6. Advanced algorithms to consider later
7. Algorithms to avoid or defer and why
8. Data requirements
9. Evaluation metrics
10. Implementation steps
11. Risks and mitigations
12. Suggested first experiment
13. Suggested production path
14. Clarifying questions still needed
"""

