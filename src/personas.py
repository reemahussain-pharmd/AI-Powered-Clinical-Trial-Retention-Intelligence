"""
Participant Persona Classifier for Clinical Trial Retention.

Classifies each participant into one of four archetypes based on their
clinical and demographic profile. Personas enable tailored communication
strategies and intervention prioritisation, reflecting a participant-centric
approach to trial design (FDA, 2012; CTTI, 2014).
"""

import pandas as pd
from typing import Tuple


PERSONA_DESCRIPTIONS = {
    "High AE Vulnerability Participant": (
        "Older participant (>65) with multiple comorbidities and complex medication regimen. "
        "Key risks: adverse event sensitivity, visit fatigue, and caregiver dependency. "
        "Recommended approach: simplified scheduling, caregiver involvement, home visit options."
    ),
    "Young Mobile Professional": (
        "Employed participant (25–45) with long commute and high visit frequency. "
        "Key risks: scheduling conflicts and work absenteeism from frequent trial visits. "
        "Recommended approach: flexible scheduling, remote options, and visit consolidation."
    ),
    "High Burden Polypharmacy Patient": (
        "Participant on ≥8 medications with multiple comorbidities and prior adverse event history. "
        "Key risks: drug-drug interactions, AE cascade, and clinical complexity. "
        "Recommended approach: pharmacist reconciliation, dedicated monitoring, close PI follow-up."
    ),
    "Transportation-Limited Participant": (
        "Participant residing >80 km from site without transportation access. "
        "Key risk: logistical dropout — no clinical issue, but practically unable to attend. "
        "Recommended approach: transportation reimbursement, home nursing, remote visits."
    ),
}


def classify_persona(patient_features: pd.Series) -> Tuple[str, str]:
    """
    Classify a participant into a retention persona based on their feature profile.

    Persona assignment is hierarchical: the most clinically specific persona
    takes precedence. Long-distance rural participants are identified first since
    logistics alone can drive dropout regardless of clinical status.

    Args:
        patient_features: Series of participant feature values (single row).

    Returns:
        Tuple of (persona_name, persona_description).
    """
    age          = float(patient_features.get("age", 40))
    comorbidities = float(patient_features.get("number_of_comorbidities", 0))
    medications  = float(patient_features.get("concomitant_medications", 0))
    employment   = str(patient_features.get("employment_status", "")).lower()
    distance     = float(patient_features.get("distance_from_site_km", 0))
    transport    = str(patient_features.get("transportation_access", "yes")).lower()
    prior_ae     = str(patient_features.get("prior_adverse_event_history", "no")).lower()
    visit_freq   = float(patient_features.get("visit_frequency_per_month", 0))

    # Transportation-Limited — logistical dropout risk without clinical cause
    if distance > 80 and transport == "no":
        name = "Transportation-Limited Participant"
        return name, PERSONA_DESCRIPTIONS[name]

    # High Burden Polypharmacy — clinical complexity with AE cascade potential
    if medications >= 8 and comorbidities >= 4 and prior_ae == "yes":
        name = "High Burden Polypharmacy Patient"
        return name, PERSONA_DESCRIPTIONS[name]

    # High AE Vulnerability — comorbid older participant with visit fatigue risk
    if age > 65 and comorbidities >= 4 and medications >= 6:
        name = "High AE Vulnerability Participant"
        return name, PERSONA_DESCRIPTIONS[name]

    # Young Mobile Professional — scheduling and work absenteeism risk
    if 25 <= age <= 45 and employment == "employed" and (distance > 40 or visit_freq >= 6):
        name = "Young Mobile Professional"
        return name, PERSONA_DESCRIPTIONS[name]

    # Default — use closest match heuristic
    scores = {
        "High AE Vulnerability Participant":  (age / 80) * 0.4 + (comorbidities / 8) * 0.3 + (medications / 12) * 0.3,
        "Young Mobile Professional":          (1 if 25 <= age <= 45 else 0) * 0.4 + (visit_freq / 12) * 0.3 + (distance / 200) * 0.3,
        "High Burden Polypharmacy Patient":   (medications / 12) * 0.4 + (comorbidities / 8) * 0.3 + (1 if prior_ae == "yes" else 0) * 0.3,
        "Transportation-Limited Participant": (distance / 200) * 0.6 + (0 if transport == "yes" else 0.4),
    }
    best_persona = max(scores, key=scores.get)
    return best_persona, PERSONA_DESCRIPTIONS[best_persona]
