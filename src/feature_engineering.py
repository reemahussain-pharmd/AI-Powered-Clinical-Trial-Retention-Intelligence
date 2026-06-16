"""
PharmD-Informed Feature Engineering for Clinical Trial Retention.

Composite features are designed to capture the multi-dimensional burden
experienced by trial participants. Each feature encodes a clinically
meaningful interaction that raw variables alone cannot express.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PharmD-informed composite features to the patient DataFrame.

    All composite features are derived from established clinical trial attrition
    literature. They encode burden constructs that predict dropout beyond
    individual raw variables.

    Args:
        df: Patient DataFrame with raw features from data_generator.

    Returns:
        DataFrame with five additional composite feature columns.
    """
    df = df.copy()

    # Visit Burden Index: captures scheduling and cognitive load from frequent visits
    # combined with a complex protocol. A patient attending 10 visits/month under
    # a high-complexity protocol faces compounded fatigue beyond either factor alone.
    df["visit_burden_index"] = (
        df["visit_frequency_per_month"] * df["protocol_complexity_score"]
    )

    # Polypharmacy Risk Score: concomitant medications plus comorbidities
    # combines direct drug-drug interaction risk with systemic disease burden.
    # Both increase AE probability, and AEs are the leading driver of dropout.
    df["polypharmacy_risk_score"] = (
        df["concomitant_medications"] + df["number_of_comorbidities"]
    )

    # Patient Burden Score: integrates logistical distance (normalised to 20 km
    # increments), baseline pain, and week-2 side effect severity.
    # Represents cumulative daily-life impact of trial participation.
    df["patient_burden_score"] = (
        (df["distance_from_site_km"] / 20.0)
        + df["baseline_pain_score"]
        + df["side_effect_severity_at_week2"]
    )

    # Logistic Friction Score: distance becomes a hard barrier only when no
    # transportation is available. For patients with transport access, distance
    # is manageable; without it, each kilometre adds proportional friction.
    df["logistic_friction_score"] = np.where(
        df["transportation_access"] == "yes",
        0.0,
        df["distance_from_site_km"] / 10.0,
    )

    # Phase-Complexity Interaction: early-phase trials (Phase 1) with high
    # protocol complexity represent the highest participant burden scenario —
    # unknown safety profile layered onto intensive monitoring requirements.
    df["phase_complexity_interaction"] = (
        df["trial_phase"] * df["protocol_complexity_score"]
    )

    return df


def get_feature_columns() -> dict:
    """
    Return categorised lists of feature columns used in modelling.

    Returns:
        Dict with keys 'numerical', 'categorical', and 'composite'.
    """
    return {
        "numerical": [
            "age",
            "bmi",
            "distance_from_site_km",
            "disease_severity_score",
            "number_of_comorbidities",
            "baseline_pain_score",
            "prior_trial_participation",
            "concomitant_medications",
            "side_effect_severity_at_week2",
            "visit_frequency_per_month",
            "protocol_complexity_score",
            "trial_duration_months",
            "consent_complexity_score",
            "investigator_experience_years",
            "trial_phase",
        ],
        "categorical": [
            "gender",
            "employment_status",
            "insurance_status",
            "transportation_access",
            "education_level",
            "prior_adverse_event_history",
            "site_id",
        ],
        "composite": [
            "visit_burden_index",
            "polypharmacy_risk_score",
            "patient_burden_score",
            "logistic_friction_score",
            "phase_complexity_interaction",
        ],
    }


if __name__ == "__main__":
    from data_generator import generate_dataset

    df = generate_dataset()
    df = add_composite_features(df)
    print(df[get_feature_columns()["composite"]].describe())
