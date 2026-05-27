from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ProjectProfile:
    project_name: str = ""
    description: str = ""
    domain: str = ""
    notes: str = ""
    selected_options: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def selected_labels(self) -> set[str]:
        return {
            label
            for labels in self.selected_options.values()
            for label in labels
        }

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlgorithmRecommendation:
    name: str
    category: str
    why_applicable: str
    best_when: str
    cautions: str
    required_data: str
    typical_metrics: str
    implementation_notes: str
    confidence_score: int

    def to_dict(self) -> dict:
        return asdict(self)

