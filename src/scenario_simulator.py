"""
What-If Protocol Change Scenario Simulator.

Enables sponsors and clinical operations teams to model the impact of
protocol modifications on individual patient dropout risk. Uses the
trained XGBoost model to re-score a patient after applying hypothetical
feature changes, quantifying potential risk reduction.
"""

import numpy as np
import pandas as pd
from typing import Dict

PRESET_SCENARIOS = [
    {
        "label": "Reduce visit frequency to 4/month",
        "changes": {"visit_frequency_per_month": 4},
        "description": "Consolidating visits to 4 per month reduces scheduling burden for employed and distance-constrained participants.",
    },
    {
        "label": "Add transportation support",
        "changes": {"transportation_access": "yes"},
        "description": "Providing transportation access or reimbursement removes the primary logistical barrier for rural participants.",
    },
    {
        "label": "Switch to Phase 2 protocol",
        "changes": {"trial_phase": 2},
        "description": "Moving from Phase 1 to a Phase 2 protocol (with established safety data) may reduce participant safety anxiety.",
    },
    {
        "label": "Reduce protocol complexity by 30%",
        "changes": {"protocol_complexity_multiplier": 0.7},
        "description": "A 30% reduction in protocol complexity (e.g. removing non-critical assessments) may reduce visit fatigue.",
    },
    {
        "label": "Add safety monitoring",
        "changes": {"side_effect_reduction": 1},
        "description": "Proactive safety monitoring intervention aims to reduce perceived side effect burden by one severity level.",
    },
]


def simulate_scenario(
    patient_features: pd.DataFrame,
    model,
    preprocessor,
    changes_dict: Dict,
) -> Dict:
    """
    Simulate a protocol change and re-score patient dropout risk.

    Applies the specified feature changes to the patient record, re-runs
    feature engineering, and obtains a new predicted dropout probability
    from the trained model.

    Args:
        patient_features: Single-row DataFrame with raw patient features.
        model: Trained XGBoost classifier.
        preprocessor: Fitted ColumnTransformer preprocessor.
        changes_dict: Dict of feature_name → new_value (or special keys
                      'protocol_complexity_multiplier' and 'side_effect_reduction').

    Returns:
        Dict with original_risk, new_risk, risk_reduction_pct, interpretation.
    """
    from feature_engineering import add_composite_features, get_feature_columns

    original_fe = add_composite_features(patient_features.copy())
    cols = get_feature_columns()
    feature_cols = cols["numerical"] + cols["categorical"] + cols["composite"]
    available_cols = [c for c in feature_cols if c in original_fe.columns]

    original_proc = preprocessor.transform(original_fe[available_cols])
    original_risk = float(model.predict_proba(original_proc)[0, 1])

    modified = patient_features.copy()

    for key, value in changes_dict.items():
        if key == "protocol_complexity_multiplier":
            current = float(modified["protocol_complexity_score"].iloc[0])
            modified["protocol_complexity_score"] = max(1, round(current * value))
        elif key == "side_effect_reduction":
            current = float(modified["side_effect_severity_at_week2"].iloc[0])
            modified["side_effect_severity_at_week2"] = max(0, current - value)
        else:
            modified[key] = value

    modified_fe = add_composite_features(modified)
    modified_proc = preprocessor.transform(modified_fe[available_cols])
    new_risk = float(model.predict_proba(modified_proc)[0, 1])

    risk_reduction_pct = round((original_risk - new_risk) * 100, 1)

    if risk_reduction_pct > 0:
        direction = "reduce"
        interpretation = (
            f"This protocol change may {direction} dropout risk from "
            f"{original_risk:.0%} to approximately {new_risk:.0%} "
            f"(estimated reduction: {risk_reduction_pct:.1f} percentage points)."
        )
    elif risk_reduction_pct < 0:
        interpretation = (
            f"This change may increase dropout risk from "
            f"{original_risk:.0%} to approximately {new_risk:.0%}. "
            "Consider combining with other retention strategies."
        )
    else:
        interpretation = (
            f"This change has minimal estimated impact on dropout risk "
            f"(current: {original_risk:.0%}). Other risk factors may be dominant."
        )

    return {
        "original_risk": original_risk,
        "new_risk": new_risk,
        "risk_reduction_pct": risk_reduction_pct,
        "interpretation": interpretation,
    }


def run_top_scenarios(
    patient_features: pd.DataFrame,
    model,
    preprocessor,
    n: int = 2,
) -> list:
    """
    Run all preset scenarios and return the top-n by risk reduction.

    Args:
        patient_features: Single-row patient DataFrame.
        model: Trained XGBoost classifier.
        preprocessor: Fitted preprocessor.
        n: Number of top scenarios to return.

    Returns:
        List of scenario result dicts, sorted by risk_reduction_pct descending.
    """
    results = []
    for scenario in PRESET_SCENARIOS:
        result = simulate_scenario(
            patient_features, model, preprocessor, scenario["changes"]
        )
        results.append({
            "label": scenario["label"],
            "description": scenario["description"],
            **result,
        })
    results.sort(key=lambda x: x["risk_reduction_pct"], reverse=True)
    return results[:n]
