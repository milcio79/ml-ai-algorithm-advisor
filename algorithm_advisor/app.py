from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from openai_client import MissingOpenAIKeyError, get_openai_model, rank_with_chatgpt
from recommender import build_project_profile, recommend_algorithms
from utils import profile_to_markdown, selected_count


st.set_page_config(page_title="ML / AI Algorithm Advisor", page_icon="ML", layout="wide")


SECTIONS: dict[str, list[tuple[str, str]]] = {
    "Business Objective": [
        ("Predict a known category or class", "Use when the target is a known class, label, status, or decision category."),
        ("Predict a numeric value", "Use for continuous targets such as cost, time, demand, probability, or score."),
        ("Forecast future values over time", "Use when the prediction depends on a timestamped historical sequence."),
        ("Detect unusual or abnormal behavior", "Use for fraud, failures, defects, intrusions, or outlier detection."),
        ("Group similar records without labels", "Use when you need segments or clusters but do not have labels."),
        ("Find patterns or recurring structures", "Use for exploratory pattern discovery in events, text, transactions, or features."),
        ("Rank or score alternatives", "Use when the output is a priority list or risk score."),
        ("Recommend items or actions", "Use for next-best-action, product, content, or workflow recommendations."),
        ("Extract information from text", "Use for field extraction from notes, forms, documents, or messages."),
        ("Detect entities or sensitive information", "Use for names, IDs, PII, PHI, financial data, or custom entity spans."),
        ("Classify documents", "Use when whole documents need categories or routing labels."),
        ("Understand document layout", "Use when where text appears on a page matters."),
        ("Detect objects or regions in images", "Use when images need bounding boxes or localized regions."),
        ("Segment precise visual regions", "Use when pixel-level masks are needed."),
        ("Search semantically across documents", "Use when keyword search is not enough."),
        ("Generate natural language answers", "Use for answer generation, assistants, or conversational interfaces."),
        ("Automate workflow decisions", "Use when model outputs trigger operational actions."),
        ("Optimize a process or resource allocation", "Use for scheduling, allocation, routing, or constrained decisions."),
        ("Explain drivers behind an outcome", "Use when users need reasons, drivers, or feature influence."),
        ("Support executive decision-making", "Use for strategic summaries, scenario analysis, or leadership reporting."),
        ("Reduce manual review workload", "Use when the goal is triage, prioritization, or partial automation."),
        ("Improve operational reliability", "Use for monitoring, failure detection, or reliability improvement."),
    ],
    "Target / Label Availability": [
        ("Reliable labeled data is available", "Select only if labels are trustworthy enough for supervised learning."),
        ("Labels exist but may be noisy", "Use when labels are available but inconsistent, subjective, or error-prone."),
        ("Labels are scarce", "Use when labeled examples are limited."),
        ("Labels are highly imbalanced", "Use when important classes are rare."),
        ("No reliable labels are available", "Explicitly states that labels should not be assumed usable."),
        ("Explicitly no labels available", "Use only when you know labels do not exist."),
        ("Target variable is continuous", "Use for numeric target variables."),
        ("Target variable is categorical", "Use for class/category targets."),
        ("Target variable is binary", "Use for yes/no, pass/fail, fraud/not fraud, approved/rejected targets."),
        ("Target has multiple classes", "Use for multiclass classification."),
        ("Target changes over time", "Use when label definitions or behavior evolve."),
        ("Historical outcomes are available", "Use when past decisions or outcomes can train or evaluate a model."),
        ("Human review outcomes are available", "Use when reviewer decisions can become labels or evaluation data."),
        ("Only weak labels or proxy labels are available", "Use when labels are inferred indirectly."),
        ("Need semi-supervised approach", "Use when labeled and unlabeled data should be combined."),
        ("Need unsupervised exploration first", "Use when discovery must precede supervised modeling."),
    ],
    "Data Types Available": [
        ("Structured tabular data", "Rows and columns such as CRM, ERP, transactions, or operational tables."),
        ("Time series data", "Timestamped observations measured over time."),
        ("Text data", "Free text, messages, notes, descriptions, or transcripts."),
        ("Long documents", "Reports, contracts, manuals, policies, or multi-page text."),
        ("Scanned documents", "Image/PDF scans requiring OCR or visual processing."),
        ("OCR output", "Text extracted from scanned documents."),
        ("Images", "Photos, scans, frames, or visual data."),
        ("Video", "Video clips or streams."),
        ("Audio", "Speech, calls, sounds, or acoustic signals."),
        ("Sensor data", "IoT, machine, telemetry, or measurement streams."),
        ("Metadata", "Document, file, user, process, or system attributes."),
        ("Spatial coordinates or bounding boxes", "Coordinates, boxes, masks, or geometry annotations."),
        ("Layout information", "Token/page positions, zones, table locations, or reading order."),
        ("Logs or event streams", "System, application, clickstream, or process events."),
        ("Graph/network data", "Nodes and edges such as relationships, dependencies, or networks."),
        ("Transactional data", "Baskets, payments, orders, claims, or event transactions."),
        ("Customer/user behavior data", "Clicks, journeys, usage, preferences, or engagement data."),
        ("Geospatial data", "Locations, routes, regions, or spatial relationships."),
        ("Multimodal data", "Combinations of text, images, audio, metadata, and tables."),
    ],
    "Data Quality and Scale": [
        ("Small dataset", "Use when data volume is limited or labeling is expensive."),
        ("Medium dataset", "Use when there is enough data for standard ML validation."),
        ("Large dataset", "Use when scale supports complex models or distributed processing."),
        ("Very large dataset", "Use when data size affects storage, training, or serving architecture."),
        ("Noisy inputs", "Use when input values contain errors, artifacts, or measurement noise."),
        ("Missing values", "Use when fields are incomplete."),
        ("Outliers are expected", "Use when extreme values may be meaningful or problematic."),
        ("Duplicates are expected", "Use when records or documents may repeat."),
        ("Inconsistent formats", "Use when fields, templates, or text formats vary."),
        ("Legacy data", "Use when source systems or historical formats are old or inconsistent."),
        ("High-cardinality categorical variables", "Use when categorical fields have many distinct values."),
        ("Data drift is expected", "Use when data distributions may change after deployment."),
        ("Data arrives continuously", "Use for streaming, incremental, or frequent arrivals."),
        ("Batch processing is acceptable", "Use when offline scheduled processing is fine."),
        ("Near real-time processing is needed", "Use when freshness matters within seconds/minutes."),
        ("Low latency is required", "Use when predictions must be very fast."),
        ("Data privacy restrictions apply", "Use when data cannot freely leave local or approved systems."),
        ("Compliance requirements apply", "Use when legal, regulatory, or audit requirements constrain the solution."),
    ],
    "Interpretability / Risk / Compliance": [
        ("High explainability required", "Use when users must understand why a recommendation was made."),
        ("Explicitly no need for explainability", "Use only when explainability is known to be unnecessary."),
        ("Auditability required", "Use when decisions need to be reviewed later."),
        ("Traceability required", "Use when data, model, version, prompt, and output lineage matter."),
        ("Human-in-the-loop required", "Use when humans must review uncertain or high-risk cases."),
        ("High cost of false positives", "Use when incorrect positive flags are expensive or harmful."),
        ("High cost of false negatives", "Use when missed cases are expensive or harmful."),
        ("Automation must be conservative", "Use when only high-confidence cases should be automated."),
        ("Automation rate is more important than perfect accuracy", "Use when throughput is the main objective."),
        ("Regulatory or compliance risk exists", "Use when the model may affect regulated decisions."),
        ("Need confidence scores", "Use when downstream decisions depend on model certainty."),
        ("Need clear decision thresholds", "Use when actions require cutoffs or score bands."),
        ("Need fallback logic", "Use when low-confidence or error cases need a defined path."),
        ("Need model monitoring", "Use when model health must be monitored over time."),
        ("Need versioned decisions", "Use when you need to know exactly what produced each decision."),
        ("Need reproducible results", "Use when experiments and outputs must be repeatable."),
    ],
    "Deployment and Engineering": [
        ("Prototype only", "Use when the goal is exploration or proof of concept."),
        ("Explicitly no deployment required", "Use only when no deployment is needed."),
        ("Production deployment required", "Use when the solution will run for real users or workflows."),
        ("API endpoint required", "Use when other systems will call the model/service."),
        ("Batch scoring required", "Use when predictions are generated on schedules or files."),
        ("Dashboard output required", "Use when results need to be viewed in BI or dashboards."),
        ("Integration with existing workflow required", "Use when model outputs must fit current tools/processes."),
        ("MLOps required", "Use when lifecycle, registry, deployment, and monitoring practices are needed."),
        ("Docker deployment required", "Use when the app/model must be containerized."),
        ("CI/CD required", "Use when automated tests and deployment pipelines are needed."),
        ("Cloud deployment required", "Use when the solution will run in cloud infrastructure."),
        ("Local/offline deployment required", "Use when the solution must run without internet or cloud dependency."),
        ("Model monitoring required", "Use when production model performance and drift must be watched."),
        ("Feedback loop required", "Use when human or outcome feedback should improve the system."),
        ("Frequent retraining expected", "Use when data changes require regular model updates."),
        ("Edge deployment required", "Use when models run on devices or local hardware."),
        ("Need scalable pipeline", "Use when data/model workflows must scale reliably."),
        ("Need documentation and handoff", "Use when other teams will maintain or operate the solution."),
    ],
    "Computer Vision Specific": [
        ("Whole image classification", "Use when the whole image receives one or more labels."),
        ("Object detection required", "Use when bounding boxes are needed."),
        ("Region localization required", "Use when rough locations or regions are needed."),
        ("Pixel-level segmentation required", "Use when exact masks are needed."),
        ("Small objects", "Use when objects are visually small relative to the image."),
        ("Low-light conditions", "Use when lighting quality is poor."),
        ("Noisy images", "Use when images have blur, artifacts, compression, or sensor noise."),
        ("Scanned pages", "Use for scanned forms, letters, PDFs, pages, or records."),
        ("Signature detection", "Use when signatures must be detected or validated."),
        ("Form/table detection", "Use when tables or form regions must be found."),
        ("Sensitive region detection", "Use when visual redaction or sensitive area detection is needed."),
        ("Need OCR plus vision", "Use when text extraction and visual layout/object detection are both needed."),
        ("Need visual similarity search", "Use when visually similar images/pages must be found."),
        ("Need template matching", "Use when fixed visual patterns/templates are expected."),
        ("Need multimodal vision plus text", "Use when visual and textual context must be combined."),
    ],
    "NLP / Text / Document Intelligence Specific": [
        ("Text classification", "Use when text snippets or documents need labels."),
        ("Named entity recognition", "Use when spans such as people, locations, IDs, or domain entities are needed."),
        ("Sensitive information detection", "Use for PII/PHI/secrets/compliance-sensitive text."),
        ("Keyword/rule-based patterns", "Use when reliable explicit patterns exist."),
        ("Semantic similarity", "Use when meaning-based comparison matters."),
        ("Topic modeling", "Use when discovering themes in documents."),
        ("Document clustering", "Use when grouping documents without labels."),
        ("Document classification", "Use when documents need categories or routing."),
        ("Layout-aware document processing", "Use when layout and coordinates matter."),
        ("OCR correction needed", "Use when OCR errors must be cleaned or normalized."),
        ("Need retrieval over documents", "Use when relevant passages must be retrieved."),
        ("Need RAG", "Use when generation should be grounded in retrieved documents."),
        ("Need summarization", "Use when long content needs condensation."),
        ("Need question answering over documents", "Use when users ask questions against a document corpus."),
        ("Need structured extraction", "Use when outputs must be fields, JSON, tables, or records."),
        ("Need metadata validation", "Use when extracted metadata must be checked against rules."),
        ("Need deduplication or near-duplicate detection", "Use when duplicates or similar documents must be found."),
    ],
    "Time Series Specific": [
        ("Trend exists", "Use when the series has a long-term increase/decrease."),
        ("Seasonality exists", "Use when repeating calendar or periodic effects exist."),
        ("External variables available", "Use when covariates are available and known for forecasting."),
        ("Multiple related time series", "Use when many related entities or hierarchy levels exist."),
        ("Irregular time intervals", "Use when observations are not evenly spaced."),
        ("Forecast horizon is short", "Use when predicting near-term future values."),
        ("Forecast horizon is long", "Use when predicting far into the future."),
        ("Need prediction intervals", "Use when uncertainty bounds matter."),
        ("Need anomaly detection over time", "Use when temporal anomalies or alerts are needed."),
        ("Need rolling backtesting", "Use when time-aware validation is required."),
        ("Need causal interpretation", "Use when drivers and interventions matter."),
        ("Need scenario forecasting", "Use when what-if assumptions should be evaluated."),
    ],
    "Available Skills / Tools / Constraints": [
        ("Team is strong in Python", "Use when Python implementation is comfortable."),
        ("Team is strong in SQL", "Use when SQL-first approaches are preferred."),
        ("Team is strong in deep learning", "Use when complex neural systems can be supported."),
        ("Team needs low-code/simple solution", "Use when maintainability and simplicity dominate."),
        ("Team can support complex ML system", "Use when advanced pipelines and models are feasible."),
        ("Limited compute budget", "Use when training or serving cost must stay low."),
        ("Strong compute available", "Use when GPU/cloud compute is available."),
        ("Need open-source tools", "Use when proprietary services should be avoided."),
        ("Need cloud-managed services", "Use when managed platforms are preferred."),
        ("Need fast implementation", "Use when time-to-first-result matters."),
        ("Need high accuracy even if complex", "Use when performance can justify complexity."),
        ("Need easy maintenance", "Use when long-term support must be simple."),
        ("Need business-friendly explanation", "Use when nontechnical stakeholders need clear rationale."),
    ],
}


EXAMPLE_SELECTIONS = {
    "Business Objective": [
        "Extract information from text",
        "Detect entities or sensitive information",
        "Classify documents",
        "Understand document layout",
        "Search semantically across documents",
        "Reduce manual review workload",
    ],
    "Target / Label Availability": [
        "Labels are scarce",
        "Human review outcomes are available",
        "Need unsupervised exploration first",
    ],
    "Data Types Available": [
        "Text data",
        "Long documents",
        "Scanned documents",
        "OCR output",
        "Images",
        "Metadata",
        "Spatial coordinates or bounding boxes",
        "Layout information",
    ],
    "Data Quality and Scale": [
        "Noisy inputs",
        "Inconsistent formats",
        "Data privacy restrictions apply",
        "Compliance requirements apply",
    ],
    "Interpretability / Risk / Compliance": [
        "High explainability required",
        "Auditability required",
        "Traceability required",
        "Human-in-the-loop required",
        "Automation must be conservative",
        "Need confidence scores",
        "Need clear decision thresholds",
        "Need fallback logic",
        "Need versioned decisions",
    ],
    "Deployment and Engineering": [
        "Prototype only",
        "Dashboard output required",
        "Integration with existing workflow required",
        "Feedback loop required",
        "Need documentation and handoff",
    ],
    "Computer Vision Specific": [
        "Scanned pages",
        "Form/table detection",
        "Sensitive region detection",
        "Need OCR plus vision",
    ],
    "NLP / Text / Document Intelligence Specific": [
        "Named entity recognition",
        "Sensitive information detection",
        "Document clustering",
        "Document classification",
        "Layout-aware document processing",
        "Need retrieval over documents",
        "Need RAG",
        "Need structured extraction",
        "Need metadata validation",
        "Need deduplication or near-duplicate detection",
    ],
    "Time Series Specific": [],
    "Available Skills / Tools / Constraints": [
        "Team is strong in Python",
        "Need open-source tools",
        "Need fast implementation",
        "Need business-friendly explanation",
    ],
}


def option_key(section: str, label: str) -> str:
    digest = hashlib.sha1(f"{section}::{label}".encode("utf-8")).hexdigest()[:12]
    return f"opt_{digest}"


def reset_selections() -> None:
    for section, options in SECTIONS.items():
        for label, _ in options:
            st.session_state[option_key(section, label)] = False
    st.session_state.project_profile = None
    st.session_state.candidate_algorithms = []
    st.session_state.openai_ranking_markdown = ""


def load_example() -> None:
    reset_selections()
    st.session_state.project_name = "Document Intelligence Compliance Workflow"
    st.session_state.project_description = (
        "Classify scanned compliance documents, extract sensitive fields, validate metadata, "
        "cluster unknown document types, and route uncertain cases to human reviewers."
    )
    st.session_state.project_domain = "Compliance / Document Intelligence"
    st.session_state.project_notes = "Unchecked characteristics remain unknown, not negative."
    for section, labels in EXAMPLE_SELECTIONS.items():
        for label in labels:
            st.session_state[option_key(section, label)] = True


def collect_selected_options() -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for section, options in SECTIONS.items():
        selected[section] = [
            label
            for label, _help in options
            if st.session_state.get(option_key(section, label), False)
        ]
    return selected


def render_sidebar() -> tuple[str, str, str, str]:
    with st.sidebar:
        st.header("Project")
        project_name = st.text_input("Optional project name", key="project_name")
        description = st.text_area("Optional project description", key="project_description", height=140)
        domain = st.text_input("Optional domain", key="project_domain")
        notes = st.text_area("Optional notes", key="project_notes", height=100)
        st.divider()
        st.caption("OpenAI model")
        st.code(get_openai_model())
        st.button("Load example: Document Intelligence Compliance Workflow", on_click=load_example, width="stretch")
        st.button("Reset selections", on_click=reset_selections, width="stretch")
    return project_name, description, domain, notes


def render_checkboxes() -> None:
    for section, options in SECTIONS.items():
        with st.expander(section, expanded=section in {"Business Objective", "Target / Label Availability", "Data Types Available"}):
            cols = st.columns(2)
            for index, (label, help_text) in enumerate(options):
                with cols[index % 2]:
                    st.checkbox(label, key=option_key(section, label), help=help_text)


def render_candidates() -> None:
    candidates = st.session_state.get("candidate_algorithms", [])
    if not candidates:
        return

    df = pd.DataFrame(
        [
            {
                "name": item.name,
                "category": item.category,
                "confidence_score": item.confidence_score,
                "why_applicable": item.why_applicable,
                "cautions": item.cautions,
                "typical_metrics": item.typical_metrics,
            }
            for item in candidates
        ]
    )
    st.subheader("Deterministic candidate algorithms")
    st.dataframe(df, width="stretch", hide_index=True)

    st.subheader("Algorithm details")
    for item in candidates:
        with st.expander(f"{item.name} · {item.category} · {item.confidence_score}/100"):
            st.markdown(f"**Why applicable:** {item.why_applicable}")
            st.markdown(f"**Best when:** {item.best_when}")
            st.markdown(f"**Cautions:** {item.cautions}")
            st.markdown(f"**Required data:** {item.required_data}")
            st.markdown(f"**Typical metrics:** {item.typical_metrics}")
            st.markdown(f"**Implementation notes:** {item.implementation_notes}")


def main() -> None:
    st.title("ML / AI Algorithm Advisor")
    st.caption("Select what you know about your project. Leave unknown aspects unchecked.")

    st.session_state.setdefault("candidate_algorithms", [])
    st.session_state.setdefault("openai_ranking_markdown", "")
    st.session_state.setdefault("project_profile", None)

    project_name, description, domain, notes = render_sidebar()

    st.info("Unchecked boxes mean unknown / not selected. They are never interpreted as 'No'.")
    render_checkboxes()

    selected_options = collect_selected_options()
    count = selected_count(selected_options)
    st.caption(f"{count} known characteristic(s) selected.")

    col1, col2 = st.columns([1, 1])
    with col1:
        suggest = st.button("Suggest candidate algorithms", type="primary", width="stretch")
    with col2:
        rank = st.button("Rank with ChatGPT", width="stretch")

    if suggest:
        profile = build_project_profile(project_name, description, domain, notes, selected_options)
        st.session_state.project_profile = profile
        if not profile.selected_labels():
            st.warning("Please select at least one known project characteristic.")
            st.session_state.candidate_algorithms = []
        else:
            candidates = recommend_algorithms(profile)
            st.session_state.candidate_algorithms = candidates
            st.session_state.openai_ranking_markdown = ""
            if not candidates:
                st.warning("Not enough information to suggest algorithms. Please select more project details.")

    profile = st.session_state.get("project_profile")
    if profile and profile.warnings:
        for warning in profile.warnings:
            st.warning(warning)

    render_candidates()

    if rank:
        profile = build_project_profile(project_name, description, domain, notes, selected_options)
        candidates = st.session_state.get("candidate_algorithms", [])
        if not profile.selected_labels():
            st.warning("Please select at least one known project characteristic.")
        elif not candidates:
            st.warning("Please run 'Suggest candidate algorithms' before ranking with ChatGPT.")
        else:
            st.session_state.project_profile = profile
            with st.spinner("Ranking with ChatGPT..."):
                try:
                    st.session_state.openai_ranking_markdown = rank_with_chatgpt(profile, candidates)
                except MissingOpenAIKeyError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"ChatGPT ranking failed. Local recommendations are still available. Details: {exc}")

    ranking = st.session_state.get("openai_ranking_markdown", "")
    if ranking:
        st.subheader("ChatGPT ranking")
        st.markdown(ranking)
        st.text_area("Copy Markdown", ranking, height=320)

    export_md = profile_to_markdown(
        st.session_state.get("project_profile"),
        st.session_state.get("candidate_algorithms"),
        st.session_state.get("openai_ranking_markdown"),
    )
    st.download_button(
        "Download full Markdown export",
        data=export_md,
        file_name="algorithm_advisor_export.md",
        mime="text/markdown",
        width="stretch",
    )


if __name__ == "__main__":
    main()
