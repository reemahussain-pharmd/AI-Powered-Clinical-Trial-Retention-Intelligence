# AI-Powered Clinical Trial Retention Intelligence System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

> ⚠️ **Disclaimer:** This project is intended for educational and portfolio demonstration purposes only.
> It does not constitute clinical advice and should not be used for patient care decisions.

---

## The Clinical Problem

Clinical trial attrition remains a major industry challenge, with studies reporting dropout rates of **20–30%**, leading to significant delays, cost overruns, and compromised trial integrity *(FDA, 2012; Getz KA et al., Ther Innov Regul Sci, 2016)*.

Patient replacement costs an estimated **$15,000–$25,000 per participant**, including rescreening, re-enrolment, additional monitoring, and regulatory burden. Across the industry, this represents billions of dollars in annual avoidable costs.

Existing retention strategies are largely reactive — triggered after dropout occurs. AI-driven early prediction allows sponsors to deploy targeted, evidence-based interventions *before* a participant disengages.

---

## System Architecture

![Architecture](outputs/architecture.png)

---

## Core Capabilities

| Capability | Description |
|-----------|-------------|
| **PREDICT** | XGBoost + LightGBM + CatBoost ensemble identifies dropout risk per patient |
| **EXPLAIN** | SHAP TreeExplainer surfaces why each patient is at risk, in plain English |
| **INTERVENE** | 7 evidence-based intervention strategies, matched to patient risk profile |
| **JUSTIFY** | Clinical Evidence Retrieval Engine cites FDA, ICH E6(R2), and peer-reviewed literature |
| **SIMULATE** | What-If Scenario Simulator models the impact of protocol changes on risk |
| **CALCULATE** | Business Impact Calculator quantifies ROI of each intervention |
| **REPORT** | Downloadable 2-page PDF report for trial sponsors |

---

## PharmD Clinical Insight

As a PharmD-trained clinical data scientist, I recognise that clinical trial dropout is not a single-cause phenomenon. This system encodes the multi-dimensional burden of trial participation — pharmacological, logistical, psychosocial, and protocol-driven — as clinically informed features.

Three of the most actionable insights from this system:

1. **Week-2 side effect severity** is the dominant dropout predictor. Proactive pharmacovigilance contact at this window is both low-cost and high-impact *(ICH E6(R2), 2016)*.
2. **Distance without transportation** creates a hard logistical barrier independent of clinical profile. Transportation reimbursement is estimated to provide moderate risk reduction at low cost *(FDA, 2012)*.
3. **Protocol complexity** amplifies visit burden. ICH E6(R2)'s principle of proportionate monitoring supports eliminating non-critical assessments to reduce patient fatigue *(Getz KA et al., 2016)*.

**Key References:**
- FDA (2012). *Guidance for Industry: Patient Retention in Clinical Trials.* U.S. Department of Health and Human Services.
- ICH E6(R2). *Good Clinical Practice: Integrated Addendum to ICH E6(R1).* (2016).
- Getz KA et al. (2016). Measuring the incidence, causes and repercussions of protocol amendments. *Ther Innov Regul Sci*, 50(4), 435–441.

---

## Model Results

| Model | AUC | F1 | Recall | Brier Score |
|-------|-----|----|--------|-------------|
| **Logistic Regression ⭐** | **0.694** | **0.531** | **0.779** | **0.216** |
| Random Forest | 0.668 | 0.435 | 0.442 | 0.200 |
| XGBoost (Optuna-tuned) | 0.640 | 0.429 | 0.411 | 0.243 |
| LightGBM | 0.660 | 0.387 | 0.316 | 0.219 |
| CatBoost | 0.663 | 0.443 | 0.432 | 0.205 |

*Logistic Regression achieved highest recall (0.78) on this synthetic dataset — prioritised for clinical retention use. SHAP explanations use the XGBoost model (saved as model_v1.pkl) for tree-based attribution.*

*Recall is prioritised: in clinical retention, missing a future dropout (false negative) is more costly than an unnecessary intervention (false positive).*

---

## Key Finding

Week-2 side effect severity is the single strongest predictor of clinical trial dropout in this modelled population, with a SHAP contribution approximately **3× larger** than the next strongest factor (logistic friction score). This aligns with the pharmacovigilance principle that early adverse events, if unmanaged, become primary discontinuation triggers.

---

## Business Impact (Modelled Estimates)

> All figures below are modelled estimates using synthetic data. They are not guaranteed outcomes.

| Scenario | Estimated Value |
|----------|----------------|
| Patient replacement cost | ~$18,000 per participant |
| High-risk patients identified (600 pts, 35% high-risk) | ~210 patients |
| Estimated dropouts preventable (model recall × 60% success rate) | ~95 patients |
| Potential total savings | ~$1.7M |
| Total intervention cost | ~$137K |
| Estimated net benefit | ~$1.6M |

*"1,000-patient trial × 30% dropout × $18,000 replacement = $5.4M at risk. A model with strong recall may identify the majority of at-risk patients. Intervention costs per at-risk patient: approximately $500. Modelled net benefit: significant potential savings in a single trial."*

---

## System Components

| Module | File | Purpose |
|--------|------|---------|
| Data Generator | `src/data_generator.py` | Generates 2,000 synthetic patient records |
| Feature Engineering | `src/feature_engineering.py` | PharmD-informed composite features |
| Validator | `src/validator.py` | Pre-model data quality checks |
| Model Pipeline | `src/model.py` | 5 classifiers, Optuna, MLflow, survival analysis |
| SHAP Explainer | `src/explainer.py` | Global & per-patient SHAP explanations |
| Intervention Engine | `src/intervention_engine.py` | Evidence-based intervention recommendations |
| Business Impact | `src/business_impact.py` | ROI and savings calculations |
| Clinical Evidence Retrieval Engine | `src/evidence_retrieval.py` | Lightweight RAG citation retrieval |
| Persona Classifier | `src/personas.py` | 4 clinical patient archetypes |
| Scenario Simulator | `src/scenario_simulator.py` | What-if protocol change modelling |
| Retention Intelligence Agent | `src/agent.py` | 9-step orchestration pipeline |
| Report Generator | `src/report_generator.py` | 2-page professional PDF output |

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data and train models
```bash
cd project1-retention-intelligence
python src/data_generator.py
python src/model.py
python src/explainer.py        # generates SHAP plots
python src/generate_architecture.py  # generates architecture diagram
python src/report_generator.py # generates sample PDF
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

### 4. Run the Jupyter notebook
```bash
jupyter notebook notebooks/01_retention_intelligence_analysis.ipynb
```

### 5. Run unit tests
```bash
pytest tests/ -v
```

---

## Screenshots

*(Add after running the app)*

---

## IQVIA Alignment

| This System Component | IQVIA Equivalent |
|----------------------|-----------------|
| XGBoost + LightGBM + CatBoost | Predictive Analytics Engine |
| SHAP Explainability | Explainable AI Module |
| Retention Intelligence Agent | IQVIA.ai Intelligent Agent |
| Intervention Engine | Retention Optimization |
| Clinical Evidence Retrieval Engine | IQVIA.ai Knowledge Retrieval (RAG) |
| What-If Scenario Simulator | Trial Design Optimization |
| MLflow Tracking | MLOps |
| PDF Clinical Reports | Sponsor Intelligence Output |

---

## Limitations

- Synthetic data for demonstration — not derived from real sponsor datasets
- Intervention estimates are modelled, not clinically validated
- Not tested on real trial datasets
- Proof of concept only — intended to demonstrate analytical and PharmD-informed AI capability

---

## Author

**Dr. Reema Mohamed Sulthan, PharmD**
Clinical Data Scientist | AI Expert (IABAC 2025)
DataMites Batch 21-OCT-24-CDS | June 2026

📧 reemahussain2097@gmail.com
🔗 github.com/reemahussain-pharmd
