"""
SHAP Explainability Module for Clinical Trial Dropout Prediction.

SHAP (SHapley Additive exPlanations) values translate black-box model predictions
into clinically interpretable feature attributions. This is essential for sponsor
acceptance and regulatory defensibility of AI-assisted retention decisions.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Human-readable labels for all model features
FEATURE_LABEL_MAP = {
    "age": "Participant Age",
    "bmi": "Body Mass Index",
    "distance_from_site_km": "Distance from Trial Site",
    "disease_severity_score": "Disease Severity Score",
    "number_of_comorbidities": "Number of Comorbidities",
    "baseline_pain_score": "Baseline Pain Score",
    "prior_trial_participation": "Prior Trial Participation",
    "concomitant_medications": "Concomitant Medications",
    "side_effect_severity_at_week2": "Week 2 Side Effect Severity",
    "visit_frequency_per_month": "Visit Frequency per Month",
    "protocol_complexity_score": "Protocol Complexity Score",
    "trial_duration_months": "Trial Duration (Months)",
    "consent_complexity_score": "Consent Complexity Score",
    "investigator_experience_years": "Investigator Experience (Years)",
    "trial_phase": "Trial Phase",
    "visit_burden_index": "Visit Burden Index",
    "polypharmacy_risk_score": "Polypharmacy Risk Score",
    "patient_burden_score": "Participant Burden Score",
    "logistic_friction_score": "Logistic Friction Score",
    "phase_complexity_interaction": "Phase-Complexity Interaction",
    "gender_F": "Gender: Female",
    "gender_M": "Gender: Male",
    "gender_Other": "Gender: Other",
    "employment_status_employed": "Employment: Employed",
    "employment_status_retired": "Employment: Retired",
    "employment_status_student": "Employment: Student",
    "employment_status_unemployed": "Employment: Unemployed",
    "insurance_status_insured": "Insurance: Insured",
    "insurance_status_partial": "Insurance: Partial",
    "insurance_status_uninsured": "Insurance: Uninsured",
    "transportation_access_no": "No Transportation Access",
    "transportation_access_yes": "Has Transportation Access",
    "education_level_graduate": "Education: Graduate",
    "education_level_primary": "Education: Primary",
    "education_level_secondary": "Education: Secondary",
    "prior_adverse_event_history_no": "No Prior Adverse Events",
    "prior_adverse_event_history_yes": "Prior Adverse Event History",
}


def to_label(feature_name: str) -> str:
    """Convert a raw feature name to a human-readable clinical label."""
    if feature_name in FEATURE_LABEL_MAP:
        return FEATURE_LABEL_MAP[feature_name]
    if feature_name.startswith("site_id_"):
        return f"Trial Site: {feature_name[len('site_id_'):]}"
    for key, label in FEATURE_LABEL_MAP.items():
        if feature_name.startswith(key):
            return label
    return feature_name.replace("_", " ").title()


def _contextual_label(feature_name: str, feature_val: float) -> str:
    """
    Return a directionally correct label for one-hot encoded _no features.

    When a _no OHE column has value 0, the participant actually has the positive
    condition (e.g., transportation_access_no=0 means they DO have transport).
    In that case, swap to the _yes label so the displayed factor makes clinical sense.
    """
    if feature_name.endswith("_no") and feature_val == 0.0:
        yes_feature = feature_name[:-3] + "_yes"
        if yes_feature in FEATURE_LABEL_MAP:
            return FEATURE_LABEL_MAP[yes_feature]
    return to_label(feature_name)


def load_model_and_preprocessor():
    """Load saved XGBoost model and preprocessor from disk."""
    model = joblib.load(MODELS_DIR / "model_v1.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor_v1.pkl")
    return model, preprocessor


def compute_shap_values(model, X_processed: np.ndarray, feature_names: list):
    """
    Compute SHAP values using TreeExplainer for XGBoost.

    TreeExplainer is exact (not approximated) for tree-based models and orders
    of magnitude faster than KernelExplainer, making it suitable for real-time
    per-patient explanations in the Streamlit UI.

    Args:
        model: Trained XGBoost classifier.
        X_processed: Preprocessed feature matrix.
        feature_names: List of feature names corresponding to columns.

    Returns:
        Tuple of (shap_values, explainer).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    return shap_values, explainer


def generate_global_plots(model, X_processed: np.ndarray,
                          feature_names: list) -> None:
    """
    Generate and save global SHAP plots: beeswarm, bar, and dependence.

    Global SHAP plots reveal which features drive dropout across the population,
    informing sponsor-level protocol improvement decisions.

    Args:
        model: Trained XGBoost classifier.
        X_processed: Preprocessed feature matrix (test or full set).
        feature_names: Feature names corresponding to columns.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    shap_values, explainer = compute_shap_values(model, X_processed, feature_names)
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_processed,
        feature_names=feature_names,
    )

    # --- Beeswarm plot ---
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Global Feature Impact on Dropout Risk",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "shap_summary_beeswarm.png", dpi=150,
                bbox_inches="tight")
    plt.close("all")
    print("[Explainer] Saved shap_summary_beeswarm.png")

    # --- Bar plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.title("SHAP Bar Plot — Mean Absolute Feature Importance",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "shap_bar_plot.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("[Explainer] Saved shap_bar_plot.png")

    # --- Dependence: side_effect_severity ---
    _save_dependence_plot(
        shap_values, X_processed, feature_names,
        "side_effect_severity_at_week2",
        "shap_dependence_side_effects.png",
        "SHAP Dependence — Week 2 Side Effect Severity vs Dropout Risk",
    )

    # --- Dependence: distance ---
    _save_dependence_plot(
        shap_values, X_processed, feature_names,
        "distance_from_site_km",
        "shap_dependence_distance.png",
        "SHAP Dependence — Distance from Site vs Dropout Risk",
    )


def _save_dependence_plot(shap_values, X_processed, feature_names,
                          feature_name, filename, title) -> None:
    """Save a SHAP dependence plot for a single feature."""
    if feature_name not in feature_names:
        return
    idx = feature_names.index(feature_name)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        idx, shap_values, X_processed,
        feature_names=feature_names,
        ax=ax, show=False,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(to_label(feature_name), fontsize=11)
    ax.set_ylabel("SHAP Value (Impact on Dropout Risk)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Explainer] Saved {filename}")


def explain_patient(
    patient_df: pd.DataFrame,
    model=None,
    preprocessor=None,
) -> Dict:
    """
    Generate a complete SHAP explanation for a single patient.

    Returns risk score, category, dropout window, and ranked risk/protective
    factors with plain-English labels — ready for the PDF report and Streamlit UI.

    Args:
        patient_df: Single-row DataFrame with raw patient features.
        model: Trained XGBoost model (loads from disk if None).
        preprocessor: Fitted ColumnTransformer (loads from disk if None).

    Returns:
        Dict with keys: risk_score, risk_category, risk_pct, top3_risk_factors,
        top3_protective_factors, dropout_window.
    """
    from feature_engineering import add_composite_features, get_feature_columns
    from model import get_dropout_window

    if model is None or preprocessor is None:
        model, preprocessor = load_model_and_preprocessor()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    patient_fe = add_composite_features(patient_df.copy())
    cols = get_feature_columns()
    feature_cols = cols["numerical"] + cols["categorical"] + cols["composite"]

    available_cols = [c for c in feature_cols if c in patient_fe.columns]
    X_raw = patient_fe[available_cols]
    X_proc = preprocessor.transform(X_raw)

    risk_score = float(model.predict_proba(X_proc)[0, 1])

    low_t = config["thresholds"]["low_risk"]
    med_t = config["thresholds"]["medium_risk"]
    if risk_score < low_t:
        risk_category = "LOW"
    elif risk_score < med_t:
        risk_category = "MEDIUM"
    else:
        risk_category = "HIGH"

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_proc)[0]

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(cols["categorical"]).tolist()
    feature_names = cols["numerical"] + cols["composite"] + cat_names

    feature_vals = X_proc[0]
    shap_triples = list(zip(feature_names, shap_vals, feature_vals))
    shap_triples_sorted = sorted(shap_triples, key=lambda x: x[1], reverse=True)

    top3_risk = [
        (f, sv, _contextual_label(f, fv))
        for f, sv, fv in shap_triples_sorted[:3]
        if sv > 0
    ]
    top3_protective = [
        (f, sv, _contextual_label(f, fv))
        for f, sv, fv in shap_triples_sorted[-3:]
        if sv < 0
    ]

    return {
        "risk_score": risk_score,
        "risk_category": risk_category,
        "risk_pct": int(risk_score * 100),
        "top3_risk_factors": top3_risk,
        "top3_protective_factors": top3_protective,
        "dropout_window": get_dropout_window(risk_score),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_generator import generate_dataset
    from feature_engineering import add_composite_features, get_feature_columns
    from model import train_all_models

    df = generate_dataset()
    model, preprocessor, metrics, feature_names = train_all_models(df)

    df_fe = add_composite_features(df.copy())
    cols = get_feature_columns()
    feature_cols = cols["numerical"] + cols["categorical"] + cols["composite"]
    X_proc = preprocessor.transform(df_fe[feature_cols])

    generate_global_plots(model, X_proc, feature_names)
