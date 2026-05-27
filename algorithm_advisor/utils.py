from __future__ import annotations

from datetime import datetime

from models import AlgorithmRecommendation, ProjectProfile


def selected_count(selected_options: dict[str, list[str]]) -> int:
    return sum(len(values) for values in selected_options.values())


def profile_to_markdown(
    profile: ProjectProfile | None,
    candidates: list[AlgorithmRecommendation] | None,
    ranking_markdown: str | None,
) -> str:
    parts: list[str] = ["# ML / AI Algorithm Advisor Export", ""]
    parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append("")

    if profile:
        parts.extend(
            [
                "## Project Profile",
                "",
                f"**Project name:** {profile.project_name or 'Not provided'}",
                f"**Domain:** {profile.domain or 'Not provided'}",
                "",
                "### Description",
                profile.description or "Not provided",
                "",
                "### Notes",
                profile.notes or "Not provided",
                "",
                "### Selected Characteristics",
            ]
        )
        for section, labels in profile.selected_options.items():
            parts.append(f"#### {section}")
            if labels:
                parts.extend([f"- {label}" for label in labels])
            else:
                parts.append("- None selected")
            parts.append("")
        if profile.warnings:
            parts.append("### Warnings")
            parts.extend([f"- {warning}" for warning in profile.warnings])
            parts.append("")

    parts.append("## Deterministic Candidate Algorithms")
    parts.append("")
    if candidates:
        for candidate in candidates:
            parts.extend(
                [
                    f"### {candidate.name}",
                    "",
                    f"- **Category:** {candidate.category}",
                    f"- **Local confidence:** {candidate.confidence_score}",
                    f"- **Why applicable:** {candidate.why_applicable}",
                    f"- **Best when:** {candidate.best_when}",
                    f"- **Cautions:** {candidate.cautions}",
                    f"- **Required data:** {candidate.required_data}",
                    f"- **Typical metrics:** {candidate.typical_metrics}",
                    f"- **Implementation notes:** {candidate.implementation_notes}",
                    "",
                ]
            )
    else:
        parts.append("No deterministic candidates generated yet.")
        parts.append("")

    parts.append("## ChatGPT Ranking")
    parts.append("")
    parts.append(ranking_markdown or "No ChatGPT ranking generated yet.")
    parts.append("")
    return "\n".join(parts)

