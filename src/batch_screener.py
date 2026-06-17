"""
Multi-Participant Batch Screening — Version 3.0

CSV upload -> batch risk scoring -> ranked output with site-level summary.
No LLMs. Uses the existing trained model pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple

from feature_engineering import add_composite_features

REQUIRED_COLS = [
    "age", "gender", "bmi", "disease_severity_score", "number_of_comorbidities",
    "concomitant_medications", "distance_from_site_km", "visit_frequency_per_month",
    "side_effect_severity_at_week2", "insurance_status", "transportation_access",
    "prior_trial_participation", "trial_phase", "consent_complexity_score",
    "visit_burden_index", "logistic_friction_score",
]

_DEFAULTS = {
    "age": 45, "bmi": 24.0, "disease_severity_score": 5.0,
    "number_of_comorbidities": 2, "concomitant_medications": 4,
    "distance_from_site_km": 30, "visit_frequency_per_month": 4,
    "side_effect_severity_at_week2": 2.0, "prior_trial_participation": 0,
    "trial_phase": 2, "consent_complexity_score": 5,
    "visit_burden_index": 4, "logistic_friction_score": 3,
    "gender": "M", "insurance_status": "insured", "transportation_access": "yes",
}

COST_PER_AT_RISK    = 1_800   # avg intervention cost per high/critical participant
REPLACEMENT_COST    = 18_000  # avg dropout replacement cost


def _risk_cat(pct: int) -> str:
    if pct >= 81:   return "Critical"
    if pct >= 61:   return "High"
    if pct >= 31:   return "Moderate"
    return "Low"


def prepare_dataframe(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Normalise column names, fill missing cols with defaults. Returns (df, warnings)."""
    warnings = []
    df = raw_df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        warnings.append(f"Missing columns filled with defaults: {', '.join(missing)}")
        for c in missing:
            df[c] = _DEFAULTS.get(c, 0)

    if "patient_id" not in df.columns:
        df["patient_id"] = [f"PT-{i+1:04d}" for i in range(len(df))]
    if "site_id" not in df.columns:
        df["site_id"] = "SITE_01"
    if "baseline_pain_score" not in df.columns:
        df["baseline_pain_score"] = 5.0
    if "protocol_complexity_score" not in df.columns:
        df["protocol_complexity_score"] = df.get("consent_complexity_score", 5)
    if "trial_duration_months" not in df.columns:
        df["trial_duration_months"] = 12
    if "investigator_experience_years" not in df.columns:
        df["investigator_experience_years"] = 5
    if "employment_status" not in df.columns:
        df["employment_status"] = "employed"
    if "education_level" not in df.columns:
        df["education_level"] = "secondary"
    if "prior_adverse_event_history" not in df.columns:
        df["prior_adverse_event_history"] = "no"

    return df, warnings


def batch_screen(df: pd.DataFrame, model: Any, preprocessor: Any) -> Dict[str, Any]:
    """
    Score every row in df and return structured results dict.
    """
    rows = []
    for _, row in df.iterrows():
        try:
            pdata = pd.DataFrame([row.to_dict()])
            pdata = add_composite_features(pdata)
            X     = preprocessor.transform(pdata)
            prob  = float(model.predict_proba(X)[0, 1])
            pct   = round(prob * 100)
            rows.append({
                "Patient ID":     str(row.get("patient_id", "—")),
                "Site":           str(row.get("site_id", "SITE_01")),
                "Risk Score (%)": pct,
                "Risk Category":  _risk_cat(pct),
                "Age":            int(float(row.get("age", 45))),
                "Distance (km)":  int(float(row.get("distance_from_site_km", 30))),
                "Comorbidities":  int(float(row.get("number_of_comorbidities", 2))),
                "Medications":    int(float(row.get("concomitant_medications", 4))),
            })
        except Exception:
            continue

    if not rows:
        return {"error": "No participants could be scored. Check column names."}

    results_df = (
        pd.DataFrame(rows)
        .sort_values("Risk Score (%)", ascending=False)
        .reset_index(drop=True)
    )

    critical_n = int((results_df["Risk Category"] == "Critical").sum())
    high_n     = int((results_df["Risk Category"] == "High").sum())
    moderate_n = int((results_df["Risk Category"] == "Moderate").sum())
    low_n      = int((results_df["Risk Category"] == "Low").sum())
    at_risk_n  = critical_n + high_n

    est_budget          = at_risk_n * COST_PER_AT_RISK
    est_prevented       = round(at_risk_n * 0.45)
    est_savings         = est_prevented * REPLACEMENT_COST
    net_benefit         = est_savings - est_budget

    # Site-level summary
    site_df = (
        results_df.groupby("Site")
        .agg(
            Participants   = ("Patient ID",      "count"),
            Mean_Risk      = ("Risk Score (%)",  "mean"),
            High_Critical  = ("Risk Score (%)",  lambda x: (x >= 61).sum()),
        )
        .reset_index()
        .rename(columns={
            "Participants":  "Participants",
            "Mean_Risk":     "Mean Risk (%)",
            "High_Critical": "High/Critical Count",
        })
    )
    site_df["Mean Risk (%)"] = site_df["Mean Risk (%)"].round(1)
    site_df = site_df.sort_values("Mean Risk (%)", ascending=False).reset_index(drop=True)

    return {
        "results_df":     results_df,
        "total":          len(results_df),
        "critical_n":     critical_n,
        "high_n":         high_n,
        "moderate_n":     moderate_n,
        "low_n":          low_n,
        "at_risk_n":      at_risk_n,
        "est_budget":     est_budget,
        "est_prevented":  est_prevented,
        "est_savings":    est_savings,
        "net_benefit":    net_benefit,
        "site_summary":   site_df,
    }
