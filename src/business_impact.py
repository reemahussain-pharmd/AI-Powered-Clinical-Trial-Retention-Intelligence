"""
Business Impact and ROI Calculator for Clinical Trial Retention Interventions.

Quantifies the financial value of AI-driven retention in terms of avoided
patient replacement costs. Replacement of a clinical trial participant involves
rescreening, re-enrolment, additional site visits, and regulatory re-submission
costs — industry estimates range from $15,000–$25,000 per patient
(Getz KA et al., Ther Innov Regul Sci, 2016).
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config() -> Dict:
    """Load project configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def calculate_patient_impact(
    risk_score: float,
    interventions: List[Dict],
    config: Dict = None,
) -> Dict:
    """
    Calculate the estimated financial impact of intervening on one high-risk patient.

    The model estimates that intervening before dropout occurs avoids the full
    patient replacement cost. Intervention costs are summed across all recommended
    strategies for the patient.

    Args:
        risk_score: Predicted dropout probability for the patient.
        interventions: List of intervention dicts from intervention_engine.
        config: Configuration dict (loaded from config.yaml if None).

    Returns:
        Dict with replacement_cost_avoided, intervention_total_cost,
        net_savings, roi_ratio, and formatted_summary.
    """
    if config is None:
        config = _load_config()

    replacement_cost = config["costs"]["patient_replacement_cost"]
    intervention_total = sum(iv.get("cost", 0) for iv in interventions)

    # Only count replacement cost as avoided if patient is high or medium risk
    # (intervention is only deployed for at-risk patients)
    if risk_score >= 0.30:
        replacement_cost_avoided = replacement_cost
    else:
        replacement_cost_avoided = 0.0

    net_savings = replacement_cost_avoided - intervention_total
    roi_ratio = (net_savings / intervention_total) if intervention_total > 0 else float("inf")

    formatted_summary = (
        f"Estimated replacement cost avoided: ${replacement_cost_avoided:,.0f}\n"
        f"Total intervention cost: ${intervention_total:,.0f}\n"
        f"Estimated net savings: ${net_savings:,.0f}\n"
        f"Return on investment: {roi_ratio:.1f}× (modelled estimate)"
    )

    return {
        "replacement_cost_avoided": replacement_cost_avoided,
        "intervention_total_cost": intervention_total,
        "net_savings": net_savings,
        "roi_ratio": round(roi_ratio, 2),
        "formatted_summary": formatted_summary,
    }


def calculate_population_impact(
    scored_df: pd.DataFrame,
    model_recall: float,
    config: Dict = None,
) -> Dict:
    """
    Estimate the aggregate financial impact of the retention system across all patients.

    Uses model recall to estimate the fraction of true dropouts identified and
    potentially prevented through intervention. All estimates are explicitly
    modelled and should not be presented as guaranteed outcomes.

    Args:
        scored_df: DataFrame with a 'risk_score' column (predicted probabilities)
                   and a 'dropout' column (ground truth, if available).
        model_recall: Model recall on the test set.
        config: Configuration dict (loaded from config.yaml if None).

    Returns:
        Dict with total_patients, high_risk_identified, dropouts_prevented,
        total_savings, total_intervention_cost, net_benefit.
    """
    if config is None:
        config = _load_config()

    replacement_cost = config["costs"]["patient_replacement_cost"]
    avg_intervention_cost = 650.0  # Approximate blended cost across all intervention types

    total_patients = len(scored_df)
    low_t = config["thresholds"]["low_risk"]
    med_t = config["thresholds"]["medium_risk"]

    high_risk = scored_df[scored_df["risk_score"] >= med_t]
    medium_risk = scored_df[
        (scored_df["risk_score"] >= low_t) & (scored_df["risk_score"] < med_t)
    ]
    high_risk_identified = len(high_risk)

    # Estimated true dropouts identified = at-risk patients × recall
    # Modelled assumption: intervention prevents dropout in ~60% of identified cases
    intervention_success_rate = 0.60
    dropouts_prevented = int(high_risk_identified * model_recall * intervention_success_rate)

    total_savings = dropouts_prevented * replacement_cost
    total_intervention_cost = high_risk_identified * avg_intervention_cost
    net_benefit = total_savings - total_intervention_cost

    return {
        "total_patients": total_patients,
        "high_risk_identified": high_risk_identified,
        "medium_risk_identified": len(medium_risk),
        "dropouts_prevented": dropouts_prevented,
        "total_savings": total_savings,
        "total_intervention_cost": total_intervention_cost,
        "net_benefit": net_benefit,
        "model_recall_used": model_recall,
        "intervention_success_rate": intervention_success_rate,
    }
