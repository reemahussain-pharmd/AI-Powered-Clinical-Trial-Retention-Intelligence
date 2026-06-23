"""
Multi-Participant Batch Screening — Version 3.1

CSV upload -> batch risk scoring -> ranked output with per-participant
explainability, site-level intelligence, and coordinator action metadata.
No LLMs. Uses the existing trained model pipeline.
"""

import sys
from datetime import date
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

# Optional study metadata — do not affect model predictions; pass through to exports
METADATA_COLS = ["study_id", "country", "investigator_id", "enrollment_date", "site_region"]

_DEFAULTS = {
    "age": 45, "bmi": 24.0, "disease_severity_score": 5.0,
    "number_of_comorbidities": 2, "concomitant_medications": 4,
    "distance_from_site_km": 30, "visit_frequency_per_month": 4,
    "side_effect_severity_at_week2": 2.0, "prior_trial_participation": 0,
    "trial_phase": 2, "consent_complexity_score": 5,
    "visit_burden_index": 4, "logistic_friction_score": 3,
    "gender": "M", "insurance_status": "insured", "transportation_access": "yes",
}

COST_PER_AT_RISK = 1_800    # avg intervention cost per high/critical participant
REPLACEMENT_COST = 18_000   # avg dropout replacement cost

_SLA = {
    "Critical": "4 Hours",
    "High":     "24 Hours",
    "Moderate": "7 Days",
    "Low":      "Standard",
}
_OWNER = {
    "Critical": "PI + Study Coordinator",
    "High":     "Study Coordinator",
    "Moderate": "Site Coordinator",
    "Low":      "Site Staff",
}
_RECOMMENDED_ACTION = {
    "Critical": "Emergency safety call + PI escalation within 4h",
    "High":     "Priority safety call + transport check within 24h",
    "Moderate": "Weekly check-in + visit barrier review",
    "Low":      "Routine follow-up per protocol schedule",
}
_SITE_ACTIONS = {
    "Critical Site": "Sponsor Support Visit",
    "High Risk":     "Enhanced Monitoring Protocol",
    "Monitor":       "Scheduled Coordinator Check-In",
    "Healthy":       "Standard Site Engagement",
}


def _risk_cat(pct: float) -> str:
    if pct >= 81:  return "Critical"
    if pct >= 61:  return "High"
    if pct >= 31:  return "Moderate"
    return "Low"


def _primary_driver(row) -> str:
    """Heuristic primary risk driver from feature values."""
    ae    = float(row.get("side_effect_severity_at_week2", 0))
    dist  = float(row.get("distance_from_site_km", 0))
    trnsp = str(row.get("transportation_access", "yes")).lower()
    visit = float(row.get("visit_burden_index", 0))
    meds  = int(float(row.get("concomitant_medications", 0)))
    comor = int(float(row.get("number_of_comorbidities", 0)))
    sev   = float(row.get("disease_severity_score", 0))
    logi  = float(row.get("logistic_friction_score", 0))

    candidates = []
    if ae >= 3:                     candidates.append(("Week 2 AE Severity",      ae * 10))
    if trnsp == "no" and dist > 40: candidates.append(("Transportation Barriers",  dist * 0.8))
    elif dist > 70:                 candidates.append(("Distance from Site",       dist * 0.6))
    if visit >= 6:                  candidates.append(("High Visit Burden",        visit * 5))
    if meds >= 7 and comor >= 3:    candidates.append(("Polypharmacy Risk",        meds * 3))
    if sev >= 7:                    candidates.append(("Disease Severity",         sev * 4))
    if logi >= 6:                   candidates.append(("Logistic Friction",        logi * 4))
    if not candidates:              candidates.append(("Clinical Profile Factors", 10))
    return max(candidates, key=lambda x: x[1])[0]


def _top3_drivers(row) -> str:
    """Build a '| '-separated string of the top 3 risk drivers with estimated impact %."""
    ae    = float(row.get("side_effect_severity_at_week2", 0))
    dist  = float(row.get("distance_from_site_km", 0))
    trnsp = str(row.get("transportation_access", "yes")).lower()
    visit = float(row.get("visit_burden_index", 0))
    meds  = int(float(row.get("concomitant_medications", 0)))
    comor = int(float(row.get("number_of_comorbidities", 0)))
    sev   = float(row.get("disease_severity_score", 0))
    logi  = float(row.get("logistic_friction_score", 0))

    scored = []
    if ae >= 3:              scored.append(("Week 2 AE Severity",      int(ae * 6),        ae * 10))
    if trnsp == "no" and dist > 40:
                             scored.append(("Transportation Barriers",  int(dist * 0.2),    dist * 0.8))
    elif dist > 50:          scored.append(("Distance from Site",       int(dist * 0.15),   dist * 0.6))
    if visit >= 5:           scored.append(("High Visit Burden",        int(visit * 2),     visit * 5))
    if meds >= 7:            scored.append(("Polypharmacy Risk",        int(meds * 2),      meds * 3))
    if sev >= 7:             scored.append(("Disease Severity",         int(sev * 2),       sev * 4))
    if logi >= 5:            scored.append(("Logistic Friction",        int(logi * 3),      logi * 4))
    if not scored:           scored.append(("Sub-Threshold Risk Factors", 8,                10))

    scored.sort(key=lambda x: x[2], reverse=True)
    return " | ".join(f"{name} (+{pct}%)" for name, pct, _ in scored[:3])


def _attrition_window(pct: float) -> str:
    if pct >= 81: return "Weeks 2-4 (Immediate)"
    if pct >= 61: return "Weeks 4-8 (Near-Term)"
    if pct >= 31: return "Weeks 8-16 (Mid-Trial)"
    return "Post-Week 16 (Late-Stage)"


def prepare_dataframe(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Normalise column names, fill missing cols with defaults. Returns (df, warnings)."""
    warnings: List[str] = []
    df = raw_df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        warnings.append(f"Missing columns filled with defaults: {', '.join(missing)}")
        for c in missing:
            df[c] = _DEFAULTS.get(c, 0)

    if "patient_id"                   not in df.columns: df["patient_id"]                   = [f"PT-{i+1:04d}" for i in range(len(df))]
    if "site_id"                      not in df.columns: df["site_id"]                      = "SITE_01"
    if "baseline_pain_score"          not in df.columns: df["baseline_pain_score"]          = 5.0
    if "protocol_complexity_score"    not in df.columns: df["protocol_complexity_score"]    = df.get("consent_complexity_score", 5)
    if "trial_duration_months"        not in df.columns: df["trial_duration_months"]        = 12
    if "investigator_experience_years" not in df.columns: df["investigator_experience_years"] = 5
    if "employment_status"            not in df.columns: df["employment_status"]            = "employed"
    if "education_level"              not in df.columns: df["education_level"]              = "secondary"
    if "prior_adverse_event_history"  not in df.columns: df["prior_adverse_event_history"]  = "no"

    return df, warnings


def batch_screen(df: pd.DataFrame, model: Any, preprocessor: Any) -> Dict[str, Any]:
    """Score every row and return a structured results dict with full intelligence output."""
    rows = []
    for _, row in df.iterrows():
        try:
            pdata = pd.DataFrame([row.to_dict()])
            pdata = add_composite_features(pdata)
            X     = preprocessor.transform(pdata)
            prob  = float(model.predict_proba(X)[0, 1])
            pct   = round(prob * 100)
            cat   = _risk_cat(pct)

            entry = {
                "Participant ID":      str(row.get("patient_id", "—")),
                "Site":                str(row.get("site_id", "SITE_01")),
                "Risk Score (%)":      pct,
                "Risk Category":       cat,
                "Age":                 int(float(row.get("age", 45))),
                "Distance (km)":       int(float(row.get("distance_from_site_km", 30))),
                "Comorbidities":       int(float(row.get("number_of_comorbidities", 2))),
                "Medications":         int(float(row.get("concomitant_medications", 4))),
                "Primary Risk Driver": _primary_driver(row),
                "Attrition Window":    _attrition_window(pct),
                "Recommended Action":  _RECOMMENDED_ACTION[cat],
                "Priority Level":      cat,
                "SLA":                 _SLA[cat],
                "Owner":               _OWNER[cat],
                "Top 3 Risk Drivers":  _top3_drivers(row),
            }
            # Carry through optional metadata if present in CSV
            for meta_col in METADATA_COLS:
                if meta_col in row.index:
                    entry[meta_col.replace("_", " ").title()] = row[meta_col]
            rows.append(entry)
        except Exception:
            continue

    if not rows:
        return {"error": "No participants could be scored. Check column names."}

    results_df = (
        pd.DataFrame(rows)
        .sort_values("Risk Score (%)", ascending=False)
        .reset_index(drop=True)
    )

    critical_n   = int((results_df["Risk Category"] == "Critical").sum())
    high_n       = int((results_df["Risk Category"] == "High").sum())
    moderate_n   = int((results_df["Risk Category"] == "Moderate").sum())
    low_n        = int((results_df["Risk Category"] == "Low").sum())
    at_risk_n    = critical_n + high_n
    total        = len(results_df)
    small_sample = total < 25

    est_budget    = at_risk_n * COST_PER_AT_RISK
    est_prevented = round(at_risk_n * 0.45)
    est_savings   = est_prevented * REPLACEMENT_COST
    net_benefit   = est_savings - est_budget
    roi_pct       = round((est_savings / est_budget - 1) * 100) if est_budget > 0 else 0
    confidence    = "High" if at_risk_n >= 50 else ("Moderate" if at_risk_n >= 10 else "Low")

    # Population intelligence
    top_driver     = results_df["Primary Risk Driver"].value_counts().idxmax() if total > 0 else "N/A"
    top_driver_pct = round(results_df["Primary Risk Driver"].value_counts().max() / total * 100) if total > 0 else 0
    highest_risk_site = (
        results_df.groupby("Site")["Risk Score (%)"].mean().idxmax()
        if results_df["Site"].nunique() > 0 else "N/A"
    )

    # Site-level summary with intelligence columns
    site_df = (
        results_df.groupby("Site")
        .agg(
            Participants  = ("Participant ID", "count"),
            Mean_Risk     = ("Risk Score (%)", "mean"),
            High_Critical = ("Risk Score (%)", lambda x: (x >= 61).sum()),
        )
        .reset_index()
        .rename(columns={"Mean_Risk": "Mean Risk (%)", "High_Critical": "High/Critical Count"})
    )
    site_df["Mean Risk (%)"]           = site_df["Mean Risk (%)"].round(1)
    site_df["Site Risk Status"]        = site_df["Mean Risk (%)"].apply(
        lambda r: "Critical Site" if r >= 75 else ("High Risk" if r >= 55 else ("Monitor" if r >= 35 else "Healthy"))
    )
    site_df["Risk Trend"]              = site_df["Mean Risk (%)"].apply(
        lambda r: "Increasing" if r >= 65 else "Stable"
    )
    site_df["Recommended Site Action"] = site_df["Site Risk Status"].map(_SITE_ACTIONS).fillna("Standard Site Engagement")
    site_df = site_df.sort_values("Mean Risk (%)", ascending=False).reset_index(drop=True)

    return {
        "results_df":        results_df,
        "total":             total,
        "critical_n":        critical_n,
        "high_n":            high_n,
        "moderate_n":        moderate_n,
        "low_n":             low_n,
        "at_risk_n":         at_risk_n,
        "est_budget":        est_budget,
        "est_prevented":     est_prevented,
        "est_savings":       est_savings,
        "net_benefit":       net_benefit,
        "roi_pct":           roi_pct,
        "confidence":        confidence,
        "small_sample":      small_sample,
        "top_driver":        top_driver,
        "top_driver_pct":    top_driver_pct,
        "highest_risk_site": highest_risk_site,
        "site_summary":      site_df,
        "today":             date.today().strftime("%Y%m%d"),
    }
