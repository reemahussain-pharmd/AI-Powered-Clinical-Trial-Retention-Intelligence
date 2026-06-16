"""
Clinical Evidence Retrieval Engine.

Implemented using lightweight RAG (Retrieval-Augmented Generation) principles.
Future integration point: replace with IQVIA.ai RAG pipeline or PubMed API retrieval.

Retrieves evidence-based citations from a curated knowledge base of clinical
trial retention guidance documents. Keyword matching maps intervention names
to relevant regulatory and scientific references.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List

PROJECT_ROOT = Path(__file__).parent.parent
KB_PATH = PROJECT_ROOT / "knowledge_base" / "retention_evidence.json"

# Keyword routing: maps intervention keywords to evidence topic IDs
INTERVENTION_TOPIC_MAP = {
    "transportation": "transportation_barrier",
    "transport": "transportation_barrier",
    "reimbursement": "transportation_barrier",
    "safety": "side_effect_management",
    "adverse": "side_effect_management",
    "side effect": "side_effect_management",
    "monitoring call": "side_effect_management",
    "protocol simplif": "protocol_complexity",
    "complexity": "protocol_complexity",
    "visit consolidat": "visit_burden",
    "visit frequency": "visit_burden",
    "decentrali": "visit_burden",
    "liaison": "patient_liaison",
    "patient advocacy": "patient_liaison",
    "medication reconcil": "polypharmacy",
    "pharmacist": "polypharmacy",
    "polypharmacy": "polypharmacy",
    "site support": "site_quality",
    "investigator site": "site_quality",
    "cra": "site_quality",
    "consent": "consent_complexity",
    "phase 1": "phase1_risk",
    "phase1": "phase1_risk",
    "investigator train": "investigator_training",
    "experience": "investigator_training",
    "elderly": "elderly_participants",
    "employment": "employment_burden",
    "digital": "digital_engagement",
}


def _load_knowledge_base() -> List[Dict]:
    """
    Load the retention evidence knowledge base from JSON.

    Returns:
        List of evidence entry dicts.
    """
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_evidence(intervention_name: str) -> Optional[Dict]:
    """
    Retrieve the most relevant evidence entry for a given intervention name.

    Uses keyword matching against a curated topic routing map.
    This is a lightweight RAG retrieval step — in production this would
    be replaced by semantic vector similarity search over a clinical literature
    corpus (e.g. IQVIA.ai knowledge retrieval or PubMed API).

    Args:
        intervention_name: Name of the recommended intervention.

    Returns:
        Matching evidence dict, or None if no match found.
    """
    kb = _load_knowledge_base()
    topic_index = {entry["topic"]: entry for entry in kb}

    intervention_lower = intervention_name.lower()

    for keyword, topic in INTERVENTION_TOPIC_MAP.items():
        if keyword in intervention_lower:
            return topic_index.get(topic)

    return None


def get_all_evidence_for_interventions(interventions: List[Dict]) -> List[Dict]:
    """
    Retrieve evidence entries for a list of interventions.

    Args:
        interventions: List of intervention dicts from intervention_engine.

    Returns:
        List of (intervention_name, evidence_dict or None) tuples as dicts.
    """
    results = []
    for iv in interventions:
        ev = get_evidence(iv["name"])
        results.append({
            "intervention": iv["name"],
            "evidence": ev,
        })
    return results


def format_evidence_panel(interventions: List[Dict]) -> str:
    """
    Format evidence citations as a readable panel for the Streamlit UI.

    Args:
        interventions: List of intervention dicts from intervention_engine.

    Returns:
        Formatted markdown string of citations.
    """
    evidence_items = get_all_evidence_for_interventions(interventions)
    lines = []
    for item in evidence_items:
        ev = item["evidence"]
        if ev:
            lines.append(f"**{item['intervention']}**")
            lines.append(f"- *Source:* {ev['source']}")
            lines.append(f"- *Evidence:* {ev['evidence']}")
            lines.append(f"- *Recommendation:* {ev['recommendation']}")
            lines.append("")
    return "\n".join(lines) if lines else "No evidence entries found."


if __name__ == "__main__":
    test_interventions = [
        "Transportation Reimbursement Program",
        "Dedicated Safety Monitoring Call (weekly)",
        "Protocol Simplification Review",
        "Assign Dedicated Patient Liaison",
        "Pharmacist-Led Medication Reconciliation",
        "Investigator Site Support Visit",
    ]
    for name in test_interventions:
        ev = get_evidence(name)
        print(f"\n{name}")
        print(f"  → {ev['source'] if ev else 'No match'}")
