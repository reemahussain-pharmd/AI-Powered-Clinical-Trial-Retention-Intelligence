"""
Clinical Intervention Recommendation Engine.

Maps patient-specific risk factors to evidence-based retention interventions.
Each intervention is grounded in FDA guidance, ICH E6(R2), or peer-reviewed
clinical operations literature — matching the PharmD expectation of
evidence-justified clinical decision support.
"""

from typing import List, Dict
import pandas as pd


def get_interventions(
    patient_features: pd.Series,
    top_risk_factors: list,
) -> List[Dict]:
    """
    Generate a personalised list of retention interventions for a patient.

    Intervention selection is rule-based, driven by the patient's raw feature
    values. Rules reflect established causes of clinical trial dropout and the
    modifiable interventions that address them.

    Args:
        patient_features: Series of raw patient feature values (single row).
        top_risk_factors: List of (feature, shap_value, label) tuples from explainer.

    Returns:
        List of intervention dicts, each with name, owner, estimated potential
        risk reduction, cost, and PharmD rationale.
    """
    interventions = []

    distance = float(patient_features.get("distance_from_site_km", 0))
    transport = str(patient_features.get("transportation_access", "yes")).lower()
    side_effects = float(patient_features.get("side_effect_severity_at_week2", 0))
    protocol_complexity = float(patient_features.get("protocol_complexity_score", 0))
    prior_ae = str(patient_features.get("prior_adverse_event_history", "no")).lower()
    visit_freq = float(patient_features.get("visit_frequency_per_month", 0))
    concomitant_meds = float(patient_features.get("concomitant_medications", 0))
    investigator_exp = float(patient_features.get("investigator_experience_years", 10))

    # Geographic barrier — modifiable without protocol changes
    if distance > 50 and transport == "no":
        interventions.append({
            "name": "Transportation Reimbursement Program",
            "owner": "Site Coordinator",
            "estimated_potential_risk_reduction": "Estimated Potential Risk Reduction: Moderate",
            "cost": 300.0,
            "pharmd_rationale": (
                "Geographic barrier is a primary non-clinical dropout cause. "
                "Reimbursing transport removes practical friction without protocol changes. "
                "(Reference: FDA Guidance for Industry: Patient Retention in Clinical Trials, 2012)"
            ),
        })

    # Early AE severity — week-2 pharmacovigilance intervention
    if side_effects >= 3:
        interventions.append({
            "name": "Dedicated Safety Monitoring Call (weekly)",
            "owner": "Principal Investigator + Study Nurse",
            "estimated_potential_risk_reduction": "Estimated Potential Risk Reduction: High",
            "cost": 150.0,
            "pharmd_rationale": (
                "Early AE severity at week 2 is associated with discontinuation. "
                "Proactive pharmacovigilance contact increases patient confidence and allows "
                "early management of adverse events before they become dropout triggers."
            ),
        })

    # High protocol complexity — ICH E6(R2) proportionality principle
    if protocol_complexity >= 7:
        interventions.append({
            "name": "Protocol Simplification Review",
            "owner": "Sponsor Medical Monitor",
            "estimated_potential_risk_reduction": "Estimated Potential Risk Reduction: Moderate",
            "cost": 1200.0,
            "pharmd_rationale": (
                "ICH E6(R2) guidance emphasises proportionate monitoring. "
                "Excessive assessments cause visit fatigue. Combining or eliminating "
                "non-critical endpoints may reduce patient burden."
            ),
        })

    # Prior AE history — heightened safety anxiety, dedicated liaison
    if prior_ae == "yes":
        interventions.append({
            "name": "Assign Dedicated Patient Liaison",
            "owner": "Patient Advocacy Team",
            "estimated_potential_risk_reduction": "Estimated Potential Risk Reduction: Moderate-High",
            "cost": 500.0,
            "pharmd_rationale": (
                "Patients with prior AE history have heightened safety anxiety. "
                "A named liaison provides consistent reassurance and early intervention "
                "at the first sign of concern."
            ),
        })

    # High visit frequency — scheduling fatigue, decentralised trial approach
    if visit_freq >= 8:
        interventions.append({
            "name": "Visit Consolidation Review",
            "owner": "Clinical Operations",
            "estimated_potential_risk_reduction": "Estimated Potential Risk Reduction: Moderate",
            "cost": 0.0,
            "pharmd_rationale": (
                "High visit frequency creates scheduling burden, particularly "
                "for employed patients. Consolidating assessments per decentralised trial "
                "guidance may reduce dropout without compromising data integrity."
            ),
        })

    # Polypharmacy — pharmacist-led reconciliation
    if concomitant_meds >= 8:
        interventions.append({
            "name": "Pharmacist-Led Medication Reconciliation",
            "owner": "Clinical Pharmacist",
            "estimated_potential_risk_reduction": "Estimated Potential Risk Reduction: Low-Moderate",
            "cost": 200.0,
            "pharmd_rationale": (
                "Polypharmacy patients face compounded adverse event risk. "
                "Pharmacist-led reconciliation identifies potential drug-drug interactions "
                "early, supporting patient safety and trial retention."
            ),
        })

    # Inexperienced investigator — site quality improvement
    if investigator_exp <= 3:
        interventions.append({
            "name": "Investigator Site Support Visit",
            "owner": "CRA / Sponsor",
            "estimated_potential_risk_reduction": "Estimated Potential Risk Reduction: Moderate",
            "cost": 800.0,
            "pharmd_rationale": (
                "Less experienced sites may have weaker patient retention protocols. "
                "A sponsor support visit can improve investigator confidence and standardise "
                "patient follow-up procedures."
            ),
        })

    # Fallback: ensure at least one recommendation is always present
    if not interventions:
        interventions.append({
            "name": "Standard Enhanced Retention Monitoring",
            "owner": "Site Coordinator",
            "estimated_potential_risk_reduction": "Estimated Potential Risk Reduction: Low",
            "cost": 0.0,
            "pharmd_rationale": (
                "Regular check-in contact with the participant to monitor for "
                "emerging barriers to attendance. No specific high-risk flags identified; "
                "standard retention monitoring is recommended per site SOPs."
            ),
        })

    return interventions


def format_interventions_table(interventions: List[Dict]) -> pd.DataFrame:
    """
    Convert the interventions list to a formatted DataFrame for display.

    Args:
        interventions: List of intervention dicts from get_interventions.

    Returns:
        DataFrame suitable for Streamlit table or PDF rendering.
    """
    rows = []
    for iv in interventions:
        rows.append({
            "Intervention": iv["name"],
            "Owner": iv["owner"],
            "Est. Potential Risk Reduction": iv["estimated_potential_risk_reduction"].replace(
                "Estimated Potential Risk Reduction: ", ""
            ),
            "Cost (USD)": f"${iv['cost']:,.0f}",
            "PharmD Rationale": iv["pharmd_rationale"],
        })
    return pd.DataFrame(rows)
