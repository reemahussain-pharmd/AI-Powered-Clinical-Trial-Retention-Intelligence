"""
TrialGuard Intelligence Agent.

Orchestrates all analytical modules in a sequential 9-step pipeline
to produce a complete retention analysis for a single patient.
This agent pattern mirrors the architecture of IQVIA.ai intelligent agents:
predict → explain → contextualise → intervene → evidence → quantify → simulate → compile → report.
"""

import pandas as pd
from typing import Dict
from pathlib import Path


class RetentionAgent:
    """
    Multi-step AI agent for clinical trial retention analysis.

    Integrates prediction, explainability, persona classification, intervention
    recommendation, evidence retrieval, business impact, scenario simulation,
    and PDF report generation into a single orchestrated pipeline.
    """

    def __init__(self, model=None, preprocessor=None, config: Dict = None):
        """
        Initialise the agent with model artefacts and configuration.

        Args:
            model: Trained XGBoost classifier. Loaded from disk if None.
            preprocessor: Fitted ColumnTransformer. Loaded from disk if None.
            config: Configuration dict. Loaded from config.yaml if None.
        """
        import yaml
        import joblib

        PROJECT_ROOT = Path(__file__).parent.parent
        CONFIG_PATH = PROJECT_ROOT / "config.yaml"
        MODELS_DIR = PROJECT_ROOT / "models"

        if model is None or preprocessor is None:
            self.model = joblib.load(MODELS_DIR / "model_v1.pkl")
            self.preprocessor = joblib.load(MODELS_DIR / "preprocessor_v1.pkl")
        else:
            self.model = model
            self.preprocessor = preprocessor

        if config is None:
            with open(CONFIG_PATH) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

    def run(self, patient_features: pd.DataFrame) -> Dict:
        """
        Execute the full 9-step retention intelligence pipeline for one patient.

        Args:
            patient_features: Single-row DataFrame with raw patient features.

        Returns:
            Complete analysis dict containing results from all pipeline steps.
        """
        result = {}
        patient_series = patient_features.iloc[0]

        # ── Step 1: PREDICT ──────────────────────────────────────────────────
        print("[TrialGuard Intelligence Agent] Step 1: Predicting dropout risk...")
        from explainer import explain_patient
        explanation = explain_patient(patient_features, self.model, self.preprocessor)
        result["risk_score"] = explanation["risk_score"]
        result["risk_category"] = explanation["risk_category"]
        result["risk_pct"] = explanation["risk_pct"]

        # ── Step 2: EXPLAIN ──────────────────────────────────────────────────
        print("[TrialGuard Intelligence Agent] Step 2: Generating SHAP explanation...")
        result["top3_risk_factors"] = explanation["top3_risk_factors"]
        result["top3_protective_factors"] = explanation["top3_protective_factors"]
        result["dropout_window"] = explanation["dropout_window"]

        # ── Step 3: PERSONA ──────────────────────────────────────────────────
        print("[TrialGuard Intelligence Agent] Step 3: Classifying patient persona...")
        from personas import classify_persona
        persona_name, persona_desc = classify_persona(patient_series)
        result["persona"] = persona_name
        result["persona_description"] = persona_desc

        # ── Step 4: INTERVENE ────────────────────────────────────────────────
        print("[TrialGuard Intelligence Agent] Step 4: Generating intervention recommendations...")
        from intervention_engine import get_interventions
        interventions = get_interventions(patient_series, explanation["top3_risk_factors"])
        result["interventions"] = interventions

        # ── Step 5: EVIDENCE ─────────────────────────────────────────────────
        print("[TrialGuard Intelligence Agent] Step 5: Retrieving clinical evidence...")
        from evidence_retrieval import get_all_evidence_for_interventions
        evidence = get_all_evidence_for_interventions(interventions)
        result["evidence"] = evidence

        # ── Step 6: IMPACT ───────────────────────────────────────────────────
        print("[TrialGuard Intelligence Agent] Step 6: Calculating business impact...")
        from business_impact import calculate_patient_impact
        impact = calculate_patient_impact(
            result["risk_score"], interventions, self.config
        )
        result["business_impact"] = impact

        # ── Step 7: SIMULATE ─────────────────────────────────────────────────
        print("[TrialGuard Intelligence Agent] Step 7: Running what-if scenarios...")
        from scenario_simulator import run_top_scenarios
        scenarios = run_top_scenarios(patient_features, self.model, self.preprocessor, n=2)
        result["top_scenarios"] = scenarios

        # ── Step 8: COMPILE ──────────────────────────────────────────────────
        print("[TrialGuard Intelligence Agent] Step 8: Compiling full analysis...")
        result["patient_features"] = patient_series.to_dict()
        result["config"] = self.config

        # ── Step 9: REPORT ───────────────────────────────────────────────────
        print("[TrialGuard Intelligence Agent] Step 9: Generating PDF report...")
        try:
            from report_generator import generate_report
            patient_id = str(patient_series.get("patient_id", "DEMO"))
            report_path = generate_report(result, patient_id)
            result["report_path"] = str(report_path)
            print(f"[TrialGuard Intelligence Agent] Report saved → {report_path}")
        except Exception as e:
            print(f"[TrialGuard Intelligence Agent] PDF generation skipped: {e}")
            result["report_path"] = None

        print("[TrialGuard Intelligence Agent] Pipeline complete.")
        return result
