"""
Clinical Trial Retention Intelligence Platform
Dr. Reema Mohamed Sulthan | PharmD | Clinical Data Scientist | Certified AI Expert

Disclaimer: For educational, research, and portfolio demonstration purposes only.
Not for clinical use or patient care decisions.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import joblib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH  = PROJECT_ROOT / "config.yaml"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
DATA_PATH    = PROJECT_ROOT / "data" / "clinical_trial_data.csv"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Powered Clinical Trial Retention Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
.stDeployButton{display:none!important}
#MainMenu{visibility:hidden!important}
header[data-testid="stHeader"]{display:none!important}
footer{visibility:hidden!important}

/* Sidebar */
[data-testid="stSidebar"]{background-color:#0D1B2A}
[data-testid="stSidebar"] *{color:#FFFFFF!important}
[data-testid="stSidebar"] .stButton button{background-color:#1a2f45!important;border:1px solid #1D9E75!important;color:#FFFFFF!important;font-weight:600!important;font-size:13px!important}
[data-testid="stSidebar"] .stButton button p{color:#FFFFFF!important}
[data-testid="stSidebar"] .stButton button:hover{background-color:#1D9E75!important;border-color:#1D9E75!important}
[data-testid="stSidebar"] .stExpander{background-color:#1a2f45!important;border:1px solid #2a4a6a!important;border-radius:6px!important}
[data-testid="stSidebar"] .stExpander details{background-color:#1a2f45!important}
[data-testid="stSidebar"] [data-testid="stExpanderToggleIcon"]{color:#1D9E75!important}
[data-testid="stSidebar"] details summary{background-color:#1a2f45!important;color:#AADDCC!important;font-weight:600!important;padding:8px 10px!important;border-radius:6px!important}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label{color:#AADDCC!important}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]>div{
    background-color:#1a2f45!important;border-color:#1D9E75!important}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input{color:#FFFFFF!important}
[data-testid="stSidebar"] .stSelectbox svg{fill:#AADDCC!important}
[data-testid="stSidebar"] .stExpander{margin-bottom:6px!important}

/* Main */
.main{background-color:#F8FAFC}
.block-container{padding-top:0.75rem}

/* Capability cards */
.cap-card{
    background:#FFFFFF;border-radius:10px;padding:18px 16px;
    box-shadow:0 2px 8px rgba(0,0,0,0.07);text-align:center;
    border-top:3px solid #1D9E75;transition:box-shadow .15s;height:100%
}
.cap-icon{font-size:28px;line-height:1;margin-bottom:8px}
.cap-title{font-size:13px;font-weight:700;color:#0D1B2A;margin-bottom:4px}
.cap-desc{font-size:11px;color:#6B7280;line-height:1.4}

/* KPI cards */
.kpi-card{
    background:#FFFFFF;border-radius:10px;padding:18px 20px;
    box-shadow:0 2px 6px rgba(0,0,0,0.07);border-left:4px solid #1D9E75
}
.kpi-label{font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.6px;font-weight:600}
.kpi-value{font-size:36px;font-weight:800;color:#0D1B2A;line-height:1.15;margin:4px 0 2px}
.kpi-sub{font-size:11px;color:#9CA3AF}

/* Result metric cards */
.metric-card{
    background:#FFFFFF;border-radius:10px;border-left:4px solid #1D9E75;
    padding:16px 18px;box-shadow:0 2px 6px rgba(0,0,0,0.07)
}
.risk-high{background:#FFF5F5;border-left-color:#D9534F}
.risk-medium{background:#FFFBF0;border-left-color:#F4B942}
.risk-low{background:#F0FAF6;border-left-color:#2E8B57}

/* Risk driver cards */
.driver-card{
    background:#FFF5F5;border-radius:8px;padding:14px 16px;
    border-left:4px solid #D9534F;margin-bottom:8px;
    display:flex;align-items:center;gap:12px
}
.driver-icon{font-size:22px;flex-shrink:0}
.driver-label{font-size:13px;font-weight:600;color:#1F2937;flex:1}
.driver-pct{font-size:18px;font-weight:800;color:#D9534F;white-space:nowrap}
.protect-card{
    background:#F0FAF6;border-radius:8px;padding:14px 16px;
    border-left:4px solid #2E8B57;margin-bottom:8px;
    display:flex;align-items:center;gap:12px
}
.protect-pct{font-size:18px;font-weight:800;color:#2E8B57;white-space:nowrap}

/* Intervention cards */
.iv-card{
    background:#FFFFFF;border-radius:10px;padding:18px 20px;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:12px;
    border-top:3px solid #1D9E75
}
.iv-title{font-size:14px;font-weight:700;color:#0D1B2A;margin-bottom:6px}
.iv-rationale{font-size:12px;color:#4B5563;margin-bottom:10px;line-height:1.5}
.iv-row{display:flex;gap:20px;flex-wrap:wrap;margin-top:6px}
.iv-stat{font-size:11px;color:#6B7280}
.iv-stat strong{color:#0D1B2A;font-size:13px}
.badge-high{background:#FEE2E2;color:#B91C1C;border-radius:20px;padding:2px 10px;font-size:11px;font-weight:700}
.badge-medium{background:#FEF3C7;color:#92400E;border-radius:20px;padding:2px 10px;font-size:11px;font-weight:700}
.badge-low{background:#D1FAE5;color:#065F46;border-radius:20px;padding:2px 10px;font-size:11px;font-weight:700}

/* What-if comparison */
.wif-before{
    background:#FFF5F5;border-radius:10px;padding:20px;text-align:center;
    border:2px solid #D9534F
}
.wif-after{
    background:#F0FAF6;border-radius:10px;padding:20px;text-align:center;
    border:2px solid #2E8B57
}
.wif-label{font-size:11px;color:#6B7280;text-transform:uppercase;font-weight:600;letter-spacing:0.5px}
.wif-pct{font-size:48px;font-weight:800;line-height:1.1}
.wif-arrow{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%}
.delta-badge{
    background:#0D1B2A;color:#FFFFFF;border-radius:8px;
    padding:8px 16px;font-size:14px;font-weight:700;margin-top:8px
}

/* Section headers */
.section-header{
    background:#0D1B2A;color:white;padding:9px 16px;
    border-radius:6px;font-weight:600;margin:20px 0 12px;
    font-size:13px;letter-spacing:0.4px
}

/* Chart captions */
.chart-caption{
    font-size:11.5px;color:#6B7280;font-style:italic;
    margin-top:4px;padding:7px 12px;
    background:#F9FAFB;border-radius:4px;border-left:3px solid #1D9E75
}

/* Operational alert boxes */
.alert-critical{
    background:#FFF5F5;border:1px solid #D9534F;
    border-radius:8px;padding:12px 16px;margin:6px 0;
    display:flex;align-items:flex-start;gap:10px
}
.alert-monitor{
    background:#FFFBF0;border:1px solid #F4B942;
    border-radius:8px;padding:12px 16px;margin:6px 0;
    display:flex;align-items:flex-start;gap:10px
}
.alert-ok{
    background:#F0FAF6;border:1px solid #2E8B57;
    border-radius:8px;padding:12px 16px;margin:6px 0
}

/* About tab cards */
.about-card{
    background:#FFFFFF;border-radius:10px;padding:16px 18px;
    box-shadow:0 1px 5px rgba(0,0,0,0.07);margin-bottom:10px;
    border-left:4px solid #1D9E75
}
.challenge-stat{
    background:#0D1B2A;color:white;border-radius:8px;
    padding:14px 18px;text-align:center
}
.challenge-num{font-size:28px;font-weight:800;color:#1D9E75}
.challenge-txt{font-size:11px;color:#9CA3AF;margin-top:2px}

/* Architecture diagram */
.arch-flow{display:flex;align-items:center;flex-wrap:wrap;gap:6px;justify-content:center;padding:16px 0}
.arch-box{
    background:#0D1B2A;color:white;border-radius:8px;
    padding:10px 14px;font-size:12px;font-weight:600;text-align:center;min-width:120px
}
.arch-box-teal{background:#1D9E75}
.arch-box-amber{background:#F4B942;color:#0D1B2A}
.arch-arrow{color:#6B7280;font-size:18px;font-weight:300}

/* Demo profile buttons */
.demo-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px}
</style>
""", unsafe_allow_html=True)


# ── Demo profiles ─────────────────────────────────────────────────────────────
DEMO_PROFILES = {
    "high_risk": dict(
        sb_age=58, sb_gender="M", sb_bmi=29.5,
        sb_employment_status="unemployed", sb_insurance_status="uninsured",
        sb_distance_from_site_km=145, sb_transportation_access="no",
        sb_education_level="secondary", sb_disease_severity_score=8.5,
        sb_number_of_comorbidities=5, sb_baseline_pain_score=7.5,
        sb_prior_trial_participation=0, sb_prior_adverse_event_history="yes",
        sb_concomitant_medications=8, sb_side_effect_severity_at_week2=4.8,
        sb_trial_phase=1, sb_visit_frequency_per_month=10,
        sb_protocol_complexity_score=9, sb_trial_duration_months=24,
        sb_consent_complexity_score=8, sb_investigator_experience_years=2,
        sb_site_id="SITE_03",
    ),
    "rural": dict(
        sb_age=45, sb_gender="F", sb_bmi=24.0,
        sb_employment_status="unemployed", sb_insurance_status="partial",
        sb_distance_from_site_km=185, sb_transportation_access="no",
        sb_education_level="primary", sb_disease_severity_score=5.5,
        sb_number_of_comorbidities=2, sb_baseline_pain_score=4.0,
        sb_prior_trial_participation=0, sb_prior_adverse_event_history="no",
        sb_concomitant_medications=3, sb_side_effect_severity_at_week2=2.5,
        sb_trial_phase=3, sb_visit_frequency_per_month=8,
        sb_protocol_complexity_score=6, sb_trial_duration_months=18,
        sb_consent_complexity_score=7, sb_investigator_experience_years=5,
        sb_site_id="SITE_07",
    ),
    "polypharmacy": dict(
        sb_age=68, sb_gender="M", sb_bmi=32.5,
        sb_employment_status="retired", sb_insurance_status="insured",
        sb_distance_from_site_km=40, sb_transportation_access="yes",
        sb_education_level="secondary", sb_disease_severity_score=7.0,
        sb_number_of_comorbidities=7, sb_baseline_pain_score=6.5,
        sb_prior_trial_participation=1, sb_prior_adverse_event_history="yes",
        sb_concomitant_medications=11, sb_side_effect_severity_at_week2=3.8,
        sb_trial_phase=2, sb_visit_frequency_per_month=6,
        sb_protocol_complexity_score=7, sb_trial_duration_months=20,
        sb_consent_complexity_score=6, sb_investigator_experience_years=8,
        sb_site_id="SITE_02",
    ),
    "low_risk": dict(
        sb_age=35, sb_gender="F", sb_bmi=23.0,
        sb_employment_status="employed", sb_insurance_status="insured",
        sb_distance_from_site_km=12, sb_transportation_access="yes",
        sb_education_level="graduate", sb_disease_severity_score=3.0,
        sb_number_of_comorbidities=0, sb_baseline_pain_score=1.5,
        sb_prior_trial_participation=3, sb_prior_adverse_event_history="no",
        sb_concomitant_medications=2, sb_side_effect_severity_at_week2=0.5,
        sb_trial_phase=3, sb_visit_frequency_per_month=2,
        sb_protocol_complexity_score=3, sb_trial_duration_months=12,
        sb_consent_complexity_score=3, sb_investigator_experience_years=18,
        sb_site_id="SITE_01",
    ),
}

_FACTOR_ICONS = {
    "Week 2 Side Effect Severity": "⚠️",
    "Distance from Trial Site": "🚗",
    "Protocol Complexity Score": "📋",
    "Visit Frequency per Month": "📅",
    "Visit Burden Index": "⏱️",
    "Polypharmacy Risk Score": "💊",
    "Patient Burden Score": "⚖️",
    "Logistic Friction Score": "🗺️",
    "Disease Severity Score": "🏥",
    "Prior Adverse Event History": "🔔",
    "Concomitant Medications": "💊",
    "Consent Complexity Score": "📄",
    "Phase-Complexity Interaction": "🔬",
    "Trial Duration (Months)": "📆",
    "Investigator Experience (Years)": "👨‍⚕️",
    "Number of Comorbidities": "🩺",
    "No Transportation Access": "🚫",
    "Baseline Pain Score": "😣",
    "Body Mass Index": "📏",
    "Patient Age": "👤",
}


def _icon(label: str) -> str:
    for k, v in _FACTOR_ICONS.items():
        if k.lower() in label.lower() or label.lower() in k.lower():
            return v
    return "📌"


def _init_state():
    defaults = DEMO_PROFILES["high_risk"]
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_demo(profile_key: str):
    for k, v in DEMO_PROFILES[profile_key].items():
        st.session_state[k] = v
    st.rerun()


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model_artefacts():
    model = joblib.load(MODELS_DIR / "model_v1.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor_v1.pkl")
    return model, preprocessor


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


# ── Utility helpers ───────────────────────────────────────────────────────────
def risk_colour(cat: str) -> str:
    return {"HIGH": "#D9534F", "MEDIUM": "#F4B942", "LOW": "#2E8B57"}.get(cat, "#888")


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def chart_caption(text: str):
    st.markdown(f'<div class="chart-caption">{text}</div>', unsafe_allow_html=True)


def gauge_chart(value: float, title: str, color: str, height: int = 240) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"size": 38, "color": color, "family": "Arial Black"}},
        title={"text": title, "font": {"size": 13, "color": "#374151"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 30],  "color": "#D1FAE5"},
                {"range": [30, 60], "color": "#FEF3C7"},
                {"range": [60, 100], "color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": "#0D1B2A", "width": 3},
                "thickness": 0.75,
                "value": round(value * 100, 1),
            },
        },
    ))
    fig.update_layout(
        height=height, margin=dict(t=55, b=5, l=5, r=5),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Landing KPI strip ─────────────────────────────────────────────────────────
def render_landing_kpis():
    if not DATA_PATH.exists():
        return
    try:
        df = load_dataset()
        config = load_config()
        model, preprocessor = load_model_artefacts()
        from feature_engineering import add_composite_features, get_feature_columns
        df_fe = add_composite_features(df.copy())
        cols = get_feature_columns()
        feat = [c for c in cols["numerical"] + cols["categorical"] + cols["composite"] if c in df_fe.columns]
        probs = model.predict_proba(preprocessor.transform(df_fe[feat]))[:, 1]
        high_n = int((probs >= config["thresholds"]["medium_risk"]).sum())
        attr_pct = df["dropout"].mean() * 100
        cost_exp = int(df["dropout"].sum()) * config["costs"]["patient_replacement_cost"]
    except Exception:
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">Participants Assessed</div>'
        f'<div class="kpi-value">{len(df):,}</div>'
        f'<div class="kpi-sub">Synthetic demonstration cohort</div></div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        f'<div class="kpi-card" style="border-left-color:#D9534F">'
        f'<div class="kpi-label">High-Risk Participants</div>'
        f'<div class="kpi-value" style="color:#D9534F">{high_n:,}</div>'
        f'<div class="kpi-sub">AI-estimated risk score &ge; 0.60</div></div>',
        unsafe_allow_html=True,
    )
    k3.markdown(
        f'<div class="kpi-card" style="border-left-color:#F4B942">'
        f'<div class="kpi-label">Observed Attrition Rate</div>'
        f'<div class="kpi-value" style="color:#F4B942">{attr_pct:.1f}%</div>'
        f'<div class="kpi-sub">Cohort dropout rate</div></div>',
        unsafe_allow_html=True,
    )
    k4.markdown(
        f'<div class="kpi-card" style="border-left-color:#0D1B2A">'
        f'<div class="kpi-label">Potential Cost Exposure</div>'
        f'<div class="kpi-value" style="color:#0D1B2A">${cost_exp/1_000_000:.1f}M</div>'
        f'<div class="kpi-sub">Unaddressed attrition cost</div></div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> pd.DataFrame:
    _init_state()

    st.sidebar.markdown(
        "<div style='padding:12px 0 4px'>"
        "<div style='font-size:18px;font-weight:700;color:#1D9E75'>🧬 Participant Input</div>"
        "<div style='font-size:11px;color:#9CA3AF;margin-top:2px'>Select a demo profile or configure manually</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Demo profile buttons
    st.sidebar.markdown(
        "<div style='font-size:11px;font-weight:600;color:#AADDCC;text-transform:uppercase;"
        "letter-spacing:0.5px;margin:8px 0 6px'>Quick Demo Profiles</div>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🔴 High-Risk Participant",    key="demo_hr", use_container_width=True): _load_demo("high_risk")
    if st.sidebar.button("🌾 Rural Participant",        key="demo_ru", use_container_width=True): _load_demo("rural")
    if st.sidebar.button("💊 Polypharmacy Participant", key="demo_pp", use_container_width=True): _load_demo("polypharmacy")
    if st.sidebar.button("🟢 Low-Risk Participant",     key="demo_lr", use_container_width=True): _load_demo("low_risk")
    st.sidebar.divider()

    # Demographics
    with st.sidebar.expander("👤 Demographics", expanded=True):
        age        = st.slider("Age", 18, 80, key="sb_age")
        gender     = st.selectbox("Gender", ["M", "F", "Other"], key="sb_gender")
        bmi        = st.slider("BMI", 16.0, 45.0, step=0.5, key="sb_bmi")
        employment = st.selectbox("Employment Status",
                                  ["employed", "unemployed", "retired", "student"],
                                  key="sb_employment_status")
        insurance  = st.selectbox("Insurance Status",
                                  ["insured", "uninsured", "partial"],
                                  key="sb_insurance_status")
        education  = st.selectbox("Education Level",
                                  ["graduate", "secondary", "primary"],
                                  key="sb_education_level")
        distance   = st.slider("Distance to Site (km)", 1, 200, key="sb_distance_from_site_km",
                               help="Travel distance from participant's home to trial site.")
        transport  = st.selectbox("Transportation Access", ["yes", "no"],
                                  key="sb_transportation_access",
                                  help="Access to reliable transportation for site visits.")

    # Clinical Profile
    with st.sidebar.expander("🏥 Clinical Profile", expanded=False):
        severity   = st.slider("Disease Severity (0–10)", 0.0, 10.0, step=0.1,
                               key="sb_disease_severity_score")
        comorbid   = st.slider("Comorbidities", 0, 8, key="sb_number_of_comorbidities")
        pain       = st.slider("Baseline Pain Score (0–10)", 0.0, 10.0, step=0.1,
                               key="sb_baseline_pain_score")
        prior_trial = st.slider("Prior Trial Participation", 0, 4,
                                key="sb_prior_trial_participation",
                                help="Number of prior clinical trials participated in.")
        prior_ae   = st.selectbox("Prior Adverse Event History", ["no", "yes"],
                                  key="sb_prior_adverse_event_history")
        medications = st.slider("Concomitant Medications", 0, 12,
                                key="sb_concomitant_medications",
                                help="Number of concurrent medications being taken.")
        side_effects = st.slider("Side Effects at Week 2 (0–5)", 0.0, 5.0, step=0.1,
                                 key="sb_side_effect_severity_at_week2",
                                 help="The single strongest predictor of dropout in this model.")

    # Trial Characteristics
    with st.sidebar.expander("⚗️ Trial Characteristics", expanded=False):
        phase      = st.selectbox("Trial Phase", [1, 2, 3, 4], key="sb_trial_phase")
        visit_freq = st.slider("Visits per Month", 1, 12,
                               key="sb_visit_frequency_per_month")
        complexity = st.slider("Protocol Complexity (1–10)", 1, 10,
                               key="sb_protocol_complexity_score")
        duration   = st.slider("Trial Duration (months)", 3, 36,
                               key="sb_trial_duration_months")
        consent    = st.slider("Consent Complexity (1–10)", 1, 10,
                               key="sb_consent_complexity_score")
        invest_exp = st.slider("Investigator Experience (years)", 1, 30,
                               key="sb_investigator_experience_years")
        site_id    = st.selectbox("Trial Site",
                                  [f"SITE_{str(i).zfill(2)}" for i in range(1, 9)],
                                  key="sb_site_id")

    return pd.DataFrame([{
        "patient_id": "DEMO_001",
        "age": st.session_state["sb_age"],
        "gender": st.session_state["sb_gender"],
        "bmi": st.session_state["sb_bmi"],
        "employment_status": st.session_state["sb_employment_status"],
        "insurance_status": st.session_state["sb_insurance_status"],
        "distance_from_site_km": st.session_state["sb_distance_from_site_km"],
        "transportation_access": st.session_state["sb_transportation_access"],
        "education_level": st.session_state["sb_education_level"],
        "disease_severity_score": st.session_state["sb_disease_severity_score"],
        "number_of_comorbidities": st.session_state["sb_number_of_comorbidities"],
        "baseline_pain_score": st.session_state["sb_baseline_pain_score"],
        "prior_trial_participation": st.session_state["sb_prior_trial_participation"],
        "prior_adverse_event_history": st.session_state["sb_prior_adverse_event_history"],
        "concomitant_medications": st.session_state["sb_concomitant_medications"],
        "side_effect_severity_at_week2": st.session_state["sb_side_effect_severity_at_week2"],
        "trial_phase": st.session_state["sb_trial_phase"],
        "visit_frequency_per_month": st.session_state["sb_visit_frequency_per_month"],
        "protocol_complexity_score": st.session_state["sb_protocol_complexity_score"],
        "trial_duration_months": st.session_state["sb_trial_duration_months"],
        "consent_complexity_score": st.session_state["sb_consent_complexity_score"],
        "investigator_experience_years": st.session_state["sb_investigator_experience_years"],
        "site_id": st.session_state["sb_site_id"],
    }])


# ── V3 helpers ────────────────────────────────────────────────────────────────

def render_ner_section(ner_result):
    """Display clinical entities extracted from document (Module 2)."""
    if ner_result.total_entities == 0:
        return

    section_header("Clinical Entity Recognition")
    st.markdown(
        "Entities automatically identified from the uploaded document. "
        "These inform extraction confidence and provide an audit trail of clinical content."
    )

    def chips(items, color):
        html = "".join(
            f'<span style="display:inline-block;background:{color};color:#fff;'
            f'border-radius:14px;padding:3px 10px;margin:3px 4px 3px 0;font-size:11.5px;'
            f'font-weight:600">{label}</span>'
            for _, label in items
        )
        return html

    col_a, col_b = st.columns(2)
    with col_a:
        if ner_result.diseases:
            st.markdown("**Conditions / Diagnoses**")
            st.markdown(chips(ner_result.diseases, "#1D9E75"), unsafe_allow_html=True)
        if ner_result.adverse_events:
            st.markdown("**Adverse Events / Safety Signals**")
            st.markdown(chips(ner_result.adverse_events, "#D9534F"), unsafe_allow_html=True)
        if ner_result.symptoms:
            st.markdown("**Symptoms Reported**")
            st.markdown(chips(ner_result.symptoms, "#F4B942"), unsafe_allow_html=True)
    with col_b:
        if ner_result.burden_flags:
            st.markdown("**Trial Burden Indicators**")
            st.markdown(chips(ner_result.burden_flags, "#E05C25"), unsafe_allow_html=True)
        if ner_result.engagement_flags:
            st.markdown("**Engagement / Protective Signals**")
            st.markdown(chips(ner_result.engagement_flags, "#3B82F6"), unsafe_allow_html=True)
        if ner_result.medications:
            st.markdown("**Medications Identified**")
            med_html = "".join(
                f'<span style="display:inline-block;background:#6B7280;color:#fff;'
                f'border-radius:14px;padding:3px 10px;margin:3px 4px 3px 0;font-size:11px">'
                f'{m.capitalize()}</span>'
                for m in ner_result.medications
            )
            st.markdown(med_html, unsafe_allow_html=True)

    n_disease = len(ner_result.diseases)
    if n_disease > 0:
        chart_caption(
            f"NER identified {n_disease} distinct condition(s) — inferred comorbidity count: {n_disease}. "
            "Dictionary-based matching only. Clinical review required."
        )
    st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)


def render_coordinator_copilot(analysis: dict, risk_cat: str):
    """Retention Coordinator Copilot — Module 5 + 6 + 7 + 8."""
    from coordinator_copilot import CoordinatorCopilot

    risk_pct    = analysis.get("risk_pct", 0)
    top3_risk   = analysis.get("top3_risk_factors", [])
    top3_prot   = analysis.get("top3_protective_factors", [])
    interventions = analysis.get("interventions", [])

    copilot  = CoordinatorCopilot()
    summary  = copilot.generate(
        risk_pct=risk_pct,
        risk_cat=risk_cat,
        top3_risk_factors=top3_risk,
        top3_protective=top3_prot,
        interventions=interventions,
        participant_data={},
    )

    # Store for PDF
    st.session_state["_copilot_summary"] = summary

    section_header("Retention Coordinator Copilot")
    st.markdown(
        '_Powered by V3.0 Clinical Reasoning Engine — template-based, deterministic. '
        'Clinical review required before action._'
    )

    # Risk narrative
    risk_color = {"Critical": "#D9534F", "High": "#E05C25", "Moderate": "#F4B942", "Low": "#1D9E75"}
    rc = risk_color.get(risk_cat, "#F4B942")
    st.markdown(
        f'<div style="background:#F8FAFC;border-left:4px solid {rc};padding:14px 18px;'
        f'border-radius:6px;margin-bottom:12px">'
        f'<div style="font-size:13px;font-weight:600;color:{rc};margin-bottom:6px">AI Coordinator Assessment</div>'
        f'<div style="font-size:13px;color:#374151;line-height:1.6">{summary.risk_narrative}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Clinical reasoning expander
    with st.expander("📖 Clinical Reasoning — Why is this participant at risk?"):
        st.markdown(summary.reasoning_text)

    if summary.expected_improvement_high > 0:
        st.success(
            f"✅ Expected retention improvement with full action plan: "
            f"**{summary.expected_improvement_low}–{summary.expected_improvement_high} percentage points**"
        )

    # Action plan with timeline
    if summary.action_items:
        section_header("Prioritised Coordinator Action Plan")
        priority_colors = {
            "Critical": ("#D9534F", "#fff"),
            "High":     ("#F4B942", "#000"),
            "Medium":   ("#1D9E75", "#fff"),
            "Low":      ("#9CA3AF", "#fff"),
        }
        timeline_icons = {
            "Within 24 hours":       "🚨",
            "Within 72 hours":       "⚡",
            "Before Next Visit":     "📅",
            "Protocol Review Cycle": "📋",
        }
        for i, act in enumerate(summary.action_items, 1):
            bg, fg = priority_colors.get(act.priority, ("#9CA3AF", "#fff"))
            icon   = timeline_icons.get(act.timeline, "•")
            st.markdown(
                f'<div style="border:1px solid #E5E7EB;border-radius:8px;padding:12px 16px;margin-bottom:8px">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
                f'<span style="background:{bg};color:{fg};border-radius:12px;padding:2px 10px;'
                f'font-size:11px;font-weight:700">{act.priority}</span>'
                f'<span style="font-weight:700;font-size:13px;color:#0D1B2A">{i}. {act.title}</span>'
                f'<span style="margin-left:auto;font-size:12px;color:#6B7280">'
                f'{icon} {act.timeline}</span>'
                f'</div>'
                f'<div style="font-size:12px;color:#374151;line-height:1.6">{act.description}</div>'
                f'<div style="font-size:11px;color:#1D9E75;margin-top:6px;font-weight:600">'
                f'Est. reduction: ~{act.expected_reduction:.0f} percentage points</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Scenario optimisation
    if summary.combo_scenarios:
        section_header("Scenario Optimisation — Intervention Combinations")
        st.markdown("Ranked intervention strategies by estimated risk reduction and ROI.")
        for sc in summary.combo_scenarios:
            badge_html = ""
            if sc.get("badge"):
                badge_col = "#1D9E75" if "ROI" in sc["badge"] else "#3B82F6"
                badge_html = (
                    f'<span style="background:{badge_col};color:#fff;border-radius:10px;'
                    f'padding:2px 8px;font-size:10px;font-weight:700;margin-left:8px">'
                    f'{sc["badge"]}</span>'
                )
            delta_pp = sc["risk_reduction_pp"]
            old_r    = sc["current_risk_pct"]
            new_r    = sc["projected_risk_pct"]
            st.markdown(
                f'<div style="background:#F8FAFC;border:1px solid #E5E7EB;border-radius:8px;'
                f'padding:10px 14px;margin-bottom:6px">'
                f'<div style="font-weight:600;font-size:12px;color:#0D1B2A">'
                f'{sc["label"]}{badge_html}</div>'
                f'<div style="display:flex;gap:20px;margin-top:6px;font-size:12px">'
                f'<span style="color:#D9534F">Current: <b>{old_r}%</b></span>'
                f'<span>→</span>'
                f'<span style="color:#1D9E75">Projected: <b>{new_r}%</b></span>'
                f'<span style="color:#374151">Reduction: <b>{delta_pp:.1f}pp</b></span>'
                f'<span style="color:#6B7280">Est. cost: <b>${sc["est_cost"]:,}</b></span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        chart_caption(
            "Diminishing returns model applied — each additional intervention is ~80% as effective as its "
            "isolated estimate. Costs are modelled approximations. Clinical review required."
        )


def render_tab_batch():
    """Multi-Participant Batch Screening — Module 9."""
    section_header("Multi-Participant Screening")
    st.markdown(
        "Upload a CSV file containing participant data for batch risk scoring. "
        "The system scores all participants and returns a ranked attrition risk table."
    )

    # Template download
    import io as _io
    template_cols = (
        "patient_id,site_id,age,gender,bmi,disease_severity_score,"
        "number_of_comorbidities,concomitant_medications,distance_from_site_km,"
        "visit_frequency_per_month,side_effect_severity_at_week2,"
        "insurance_status,transportation_access,prior_trial_participation,"
        "trial_phase,consent_complexity_score,visit_burden_index,logistic_friction_score\n"
        "PT-0001,SITE_01,58,M,26.5,7.0,3,8,75,5,3.5,insured,no,1,2,6,6,5\n"
        "PT-0002,SITE_01,42,F,22.1,4.0,1,2,12,3,0.5,insured,yes,0,3,4,2,1\n"
        "PT-0003,SITE_02,67,F,30.2,8.5,5,11,90,6,4.5,uninsured,no,0,2,8,8,7\n"
    )
    st.download_button(
        "📥 Download Sample CSV Template",
        data=template_cols,
        file_name="batch_screening_template.csv",
        mime="text/csv",
    )

    uploaded_csv = st.file_uploader(
        "Upload Participant CSV",
        type=["csv"],
        key="batch_uploader",
        help="Required columns: age, gender, bmi, disease_severity_score, number_of_comorbidities, "
             "concomitant_medications, distance_from_site_km, visit_frequency_per_month, "
             "side_effect_severity_at_week2, insurance_status, transportation_access, "
             "prior_trial_participation, trial_phase, consent_complexity_score, "
             "visit_burden_index, logistic_friction_score",
    )

    if uploaded_csv is None:
        st.info("Upload a CSV file to begin batch screening. Use the template above as a guide.")
        return

    try:
        raw_df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    from batch_screener import prepare_dataframe, batch_screen
    df, prep_warnings = prepare_dataframe(raw_df)
    for w in prep_warnings:
        st.warning(f"⚠️ {w}")

    try:
        model, preprocessor = load_model_artefacts()
    except Exception:
        st.error("Model artefacts not found. Run `python src/model.py` first.")
        return

    with st.spinner(f"Scoring {len(df)} participants…"):
        result = batch_screen(df, model, preprocessor)

    if "error" in result:
        st.error(result["error"])
        return

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Participants", result["total"])
    k2.metric("Critical Risk",  result["critical_n"],  delta=None)
    k3.metric("High Risk",      result["high_n"])
    k4.metric("Moderate Risk",  result["moderate_n"])
    k5.metric("Low Risk",       result["low_n"])
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # Budget summary
    at_risk = result["at_risk_n"]
    if at_risk > 0:
        section_header("Intervention Budget Estimate")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("High/Critical Participants", at_risk,
                  help="Participants recommended for active retention intervention.")
        b2.metric("Est. Intervention Budget", f"${result['est_budget']:,}",
                  help="Estimated total intervention cost at $1,800 per at-risk participant.")
        b3.metric("Est. Dropouts Preventable", result["est_prevented"],
                  help="Based on model recall × estimated 45% intervention success rate.")
        b4.metric("Est. Net Benefit", f"${result['net_benefit']:,}",
                  help="Estimated savings minus intervention cost. Modelled estimate only.")
        st.caption("_Modelled estimates. $18,000 average replacement cost (Getz KA et al., 2016)._")

    # Ranked risk table
    section_header("Participant Risk Ranking")
    results_df = result["results_df"].copy()

    def risk_color_row(val):
        colors = {"Critical": "background-color:#FEE2E2", "High": "background-color:#FEF3C7",
                  "Moderate": "background-color:#FFF7ED", "Low": "background-color:#D1FAE5"}
        return [colors.get(v, "") for v in val]

    styled = results_df.style.apply(
        lambda row: risk_color_row(row[["Risk Category"]].values.flatten()),
        subset=["Risk Category"], axis=1
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # CSV download
    st.download_button(
        "📥 Download Risk Rankings CSV",
        data=results_df.to_csv(index=False),
        file_name="batch_risk_ranking.csv",
        mime="text/csv",
    )

    # Site summary
    if not result["site_summary"].empty:
        section_header("Site-Level Retention Summary")
        st.dataframe(result["site_summary"], use_container_width=True, hide_index=True)
        chart_caption(
            "Mean risk and high/critical count per site. Sites with elevated mean risk "
            "may require site-level retention intervention or investigator support."
        )


# ── TAB 0: Clinical Document Intake ──────────────────────────────────────────
def render_tab_intake():
    from document_intake import (
        extract_text_from_pdf, ClinicalDocumentParser,
        FIELD_DEFAULTS, FIELD_LABELS, EXTRACTION_ORDER,
        build_audit_log, generate_sample_crf,
    )

    section_header("Clinical Document Intake & Auto-Population")
    st.markdown(
        "Upload a clinical trial screening form, CRF, or participant summary PDF. "
        "The system extracts participant data automatically for your review before analysis."
    )
    st.warning(
        "⚕️ **Human-in-the-loop required.** Extracted values must be reviewed and confirmed "
        "by a qualified user before retention analysis is run. "
        "This module is for educational and portfolio demonstration purposes only."
    )

    # Sample CRF download
    sample_bytes = generate_sample_crf()
    if sample_bytes:
        st.download_button(
            "📥 Download Sample CRF (use this to test the module)",
            data=sample_bytes,
            file_name="sample_clinical_crf.pdf",
            mime="application/pdf",
        )

    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Participant Clinical Document",
        type=["pdf"],
        help="Supported: text-based PDF — clinical screening forms, CRFs, participant summaries.",
        key="intake_uploader",
    )

    if uploaded is None:
        st.info(
            "**How to use:** Download the sample CRF above, then upload it here "
            "to see automatic field extraction with confidence scores."
        )
        return

    # ── Text extraction ───────────────────────────────────────────────────────
    file_bytes = uploaded.read()
    with st.spinner("Extracting document text…"):
        doc_text, extraction_method = extract_text_from_pdf(file_bytes)

    if not doc_text.strip():
        st.error(
            "No text could be extracted. The PDF may be image-only (scanned). "
            "Please upload a text-based PDF."
        )
        return

    with st.expander("📄 Document Preview", expanded=False):
        preview = doc_text[:3000] + ("…" if len(doc_text) > 3000 else "")
        st.text_area("Extracted text", preview, height=240, disabled=True, label_visibility="collapsed")
        st.caption(f"Extraction engine: **{extraction_method}** · {len(doc_text):,} characters extracted")

    # ── Field parsing ─────────────────────────────────────────────────────────
    parser = ClinicalDocumentParser()
    with st.spinner("Parsing clinical fields…"):
        results = parser.parse(doc_text)

    # ── Extraction summary strip ──────────────────────────────────────────────
    section_header("Extraction Summary")
    high_n = sum(1 for r in results.values() if r.confidence == "High"   and not r.is_fallback)
    med_n  = sum(1 for r in results.values() if r.confidence == "Medium" and not r.is_fallback)
    low_n  = sum(1 for r in results.values() if r.is_fallback)
    total  = len(results)

    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Total Fields</div>'
        f'<div class="kpi-value" style="font-size:28px">{total}</div>'
        f'<div class="kpi-sub">Parsed from document</div></div>',
        unsafe_allow_html=True,
    )
    ec2.markdown(
        f'<div class="kpi-card" style="border-left-color:#2E8B57">'
        f'<div class="kpi-label">High Confidence</div>'
        f'<div class="kpi-value" style="font-size:28px;color:#2E8B57">{high_n}</div>'
        f'<div class="kpi-sub">Explicit pattern match</div></div>',
        unsafe_allow_html=True,
    )
    ec3.markdown(
        f'<div class="kpi-card" style="border-left-color:#F4B942">'
        f'<div class="kpi-label">Medium Confidence</div>'
        f'<div class="kpi-value" style="font-size:28px;color:#F4B942">{med_n}</div>'
        f'<div class="kpi-sub">Inferred from context</div></div>',
        unsafe_allow_html=True,
    )
    ec4.markdown(
        f'<div class="kpi-card" style="border-left-color:#D9534F">'
        f'<div class="kpi-label">Missing / Fallback</div>'
        f'<div class="kpi-value" style="font-size:28px;color:#D9534F">{low_n}</div>'
        f'<div class="kpi-sub">Default applied — review required</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    missing_labels = [FIELD_LABELS[k] for k, r in results.items() if r.is_fallback]
    if missing_labels:
        st.warning(
            f"⚠️ **{len(missing_labels)} field(s) not found** — default values applied. "
            f"Please complete: {', '.join(missing_labels)}."
        )

    # ── Participant snapshot ───────────────────────────────────────────────────
    section_header("Participant Snapshot")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;font-weight:600">Age / Gender</div>'
        f'<div style="font-size:22px;font-weight:800;color:#0D1B2A">'
        f'{results["age"].value} / {results["gender"].value}</div>'
        f'<div style="font-size:11px;color:#9CA3AF">Demographics</div></div>',
        unsafe_allow_html=True,
    )
    s2.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;font-weight:600">Comorbidities / Meds</div>'
        f'<div style="font-size:22px;font-weight:800;color:#0D1B2A">'
        f'{results["number_of_comorbidities"].value} / {results["concomitant_medications"].value}</div>'
        f'<div style="font-size:11px;color:#9CA3AF">Clinical complexity</div></div>',
        unsafe_allow_html=True,
    )
    s3.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;font-weight:600">Distance / Transport</div>'
        f'<div style="font-size:22px;font-weight:800;color:#0D1B2A">'
        f'{results["distance_from_site_km"].value} km / {results["transportation_access"].value.upper()}</div>'
        f'<div style="font-size:11px;color:#9CA3AF">Logistical profile</div></div>',
        unsafe_allow_html=True,
    )
    s4.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;font-weight:600">Phase / Week 2 AE</div>'
        f'<div style="font-size:22px;font-weight:800;color:#0D1B2A">'
        f'Ph.{results["trial_phase"].value} / {results["side_effect_severity_at_week2"].value}</div>'
        f'<div style="font-size:11px;color:#9CA3AF">Trial characteristics</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # ── Clinical Entity Recognition (V3) ──────────────────────────────────────
    try:
        from clinical_ner import ClinicalNERExtractor
        ner_extractor = ClinicalNERExtractor()
        ner_result    = ner_extractor.extract(doc_text)
        render_ner_section(ner_result)
        # Use NER-inferred comorbidity count to improve extraction if fallback
        if (ner_result.inferred_comorbidity_count > 0
                and results.get("number_of_comorbidities", None) is not None
                and results["number_of_comorbidities"].is_fallback):
            from document_intake import FieldResult
            results["number_of_comorbidities"] = FieldResult(
                value=ner_result.inferred_comorbidity_count,
                confidence="Medium",
                raw_match=f"{ner_result.inferred_comorbidity_count} conditions identified by NER",
                method="ner_inferred",
                is_fallback=False,
            )
    except Exception:
        pass

    # ── Validation form ───────────────────────────────────────────────────────
    section_header("Review & Edit Extracted Values")
    st.markdown(
        "Each field shows the extracted value and its confidence level. "
        "Edit any incorrect values, then click **Confirm** below."
    )

    def conf_badge(r) -> str:
        pct = r.confidence_pct
        if r.is_fallback:
            return "🔴 Missing (0%)"
        if r.confidence == "High":
            return f"🟢 High ({pct}%)"
        if r.confidence == "Medium":
            return f"🟡 Medium ({pct}%)"
        return f"🔴 Low ({pct}%)"

    fv = {}   # final (possibly edited) form values
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("##### Demographics & Logistics")

        r = results["age"]
        fv["age"] = st.number_input(
            f"Participant Age  [{conf_badge(r)}]",
            18, 100, int(r.value), 1, key="iv_age",
        )
        r = results["gender"]
        opts_g = ["M", "F", "Other"]
        fv["gender"] = st.selectbox(
            f"Gender  [{conf_badge(r)}]", opts_g,
            index=opts_g.index(r.value) if r.value in opts_g else 0, key="iv_gender",
        )
        r = results["bmi"]
        fv["bmi"] = st.number_input(
            f"BMI (kg/m²)  [{conf_badge(r)}]",
            10.0, 60.0, float(r.value), 0.1, key="iv_bmi",
        )
        r = results["distance_from_site_km"]
        fv["distance_from_site_km"] = st.number_input(
            f"Distance from Trial Site (km)  [{conf_badge(r)}]",
            0, 500, int(r.value), 1, key="iv_dist",
        )
        r = results["transportation_access"]
        fv["transportation_access"] = st.selectbox(
            f"Transportation Access  [{conf_badge(r)}]",
            ["yes", "no"], index=0 if r.value == "yes" else 1, key="iv_transport",
        )
        r = results["insurance_status"]
        opts_ins = ["insured", "uninsured", "partial"]
        fv["insurance_status"] = st.selectbox(
            f"Insurance Status  [{conf_badge(r)}]",
            opts_ins,
            index=opts_ins.index(r.value) if r.value in opts_ins else 0,
            key="iv_insurance",
        )
        r = results["prior_trial_participation"]
        fv["prior_trial_participation"] = st.number_input(
            f"Prior Trial Participation  [{conf_badge(r)}]",
            0, 10, int(r.value), 1, key="iv_prior",
        )
        r = results["visit_frequency_per_month"]
        fv["visit_frequency_per_month"] = st.number_input(
            f"Visit Frequency per Month  [{conf_badge(r)}]",
            1, 20, int(r.value), 1, key="iv_visits",
        )

    with col_r:
        st.markdown("##### Clinical & Trial Characteristics")

        r = results["disease_severity_score"]
        fv["disease_severity_score"] = st.number_input(
            f"Disease Severity Score (0–10)  [{conf_badge(r)}]",
            0.0, 10.0, float(r.value), 0.1, key="iv_severity",
        )
        r = results["number_of_comorbidities"]
        fv["number_of_comorbidities"] = st.number_input(
            f"Number of Comorbidities  [{conf_badge(r)}]",
            0, 15, int(r.value), 1, key="iv_comorbid",
        )
        r = results["concomitant_medications"]
        fv["concomitant_medications"] = st.number_input(
            f"Concomitant Medications  [{conf_badge(r)}]",
            0, 25, int(r.value), 1, key="iv_meds",
        )
        r = results["side_effect_severity_at_week2"]
        fv["side_effect_severity_at_week2"] = st.number_input(
            f"Side Effect Severity at Week 2 (0–5)  [{conf_badge(r)}]",
            0.0, 5.0, float(r.value), 0.1, key="iv_ae",
        )
        r = results["trial_phase"]
        opts_ph = [1, 2, 3, 4]
        fv["trial_phase"] = st.selectbox(
            f"Trial Phase  [{conf_badge(r)}]",
            opts_ph,
            index=opts_ph.index(int(r.value)) if int(r.value) in opts_ph else 1,
            key="iv_phase",
        )
        r = results["consent_complexity_score"]
        fv["consent_complexity_score"] = st.number_input(
            f"Consent Complexity Score (1–10)  [{conf_badge(r)}]",
            1.0, 10.0, float(r.value), 0.5, key="iv_consent",
        )
        r = results["visit_burden_index"]
        fv["visit_burden_index"] = st.number_input(
            f"Visit Burden Index  [{conf_badge(r)}]",
            0.0, 20.0, float(r.value), 0.5, key="iv_vbi",
        )
        r = results["logistic_friction_score"]
        fv["logistic_friction_score"] = st.number_input(
            f"Logistic Friction Score  [{conf_badge(r)}]",
            0.0, 10.0, float(r.value), 0.5, key="iv_lfs",
        )

    # ── Confirmation ──────────────────────────────────────────────────────────
    st.divider()
    c_btn, c_status = st.columns([1, 2])
    with c_btn:
        confirmed = st.button(
            "✅ Confirm Extracted Data & Populate Sidebar",
            type="primary",
            use_container_width=True,
            key="intake_confirm",
        )
    with c_status:
        if st.session_state.get("intake_confirmed"):
            st.success(
                "✅ Sidebar populated from document. "
                "Switch to **Participant Retention Assessment** and click Run Retention Analysis."
            )

    if confirmed:
        # Determine which fields were edited vs extraction
        edited = [
            FIELD_LABELS[k] for k in EXTRACTION_ORDER
            if k in fv and str(fv[k]) != str(results[k].value)
        ]
        # Store values under a neutral key — Streamlit forbids writing widget-bound
        # session state keys (sb_*) after those widgets have already rendered.
        # main() reads _intake_pending BEFORE the sidebar renders and applies them.
        st.session_state["_intake_pending"] = {
            "sb_age":                           int(fv["age"]),
            "sb_gender":                        fv["gender"],
            "sb_bmi":                           float(fv["bmi"]),
            "sb_distance_from_site_km":         int(fv["distance_from_site_km"]),
            "sb_transportation_access":         fv["transportation_access"],
            "sb_insurance_status":              fv["insurance_status"],
            "sb_prior_trial_participation":     int(fv["prior_trial_participation"]),
            "sb_visit_frequency_per_month":     int(fv["visit_frequency_per_month"]),
            "sb_disease_severity_score":        float(fv["disease_severity_score"]),
            "sb_number_of_comorbidities":       int(fv["number_of_comorbidities"]),
            "sb_concomitant_medications":       int(fv["concomitant_medications"]),
            "sb_side_effect_severity_at_week2": float(fv["side_effect_severity_at_week2"]),
            "sb_trial_phase":                   int(fv["trial_phase"]),
            "sb_consent_complexity_score":      float(fv["consent_complexity_score"]),
        }
        st.session_state["doc_source"]          = f"Document Upload ({uploaded.name})"
        st.session_state["doc_extraction_method"] = extraction_method
        st.session_state["intake_confirmed"]    = True
        st.session_state["intake_edited_fields"] = edited
        st.rerun()

    # ── Audit log ─────────────────────────────────────────────────────────────
    with st.expander("📋 Extraction Audit Log"):
        log_rows = build_audit_log(
            filename=uploaded.name,
            extraction_method=extraction_method,
            results=results,
            edited_fields=st.session_state.get("intake_edited_fields", []),
        )
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
        chart_caption(
            "Audit log records document source, extraction engine, confidence breakdown, "
            "and any fields manually edited by the user prior to confirmation."
        )


# ── TAB 1: Participant Retention Assessment ───────────────────────────────────
def render_tab1(patient_df: pd.DataFrame, config: dict):
    run = st.button("🔍 Run Retention Analysis", type="primary", use_container_width=True)

    if not run:
        st.info(
            "**How to use:** Select a demo profile in the sidebar (one click), or configure a custom "
            "participant profile manually. Then click **Run Retention Analysis** above to generate a full "
            "explainable AI risk assessment with intervention recommendations and financial impact."
        )
        return

    try:
        model, preprocessor = load_model_artefacts()
    except Exception:
        st.error("Model artefacts not found. Run `python src/model.py` first.")
        return

    with st.spinner("Generating retention intelligence report..."):
        from agent import RetentionAgent
        agent    = RetentionAgent(model=model, preprocessor=preprocessor, config=config)
        analysis = agent.run(patient_df)

    risk_score = analysis["risk_score"]
    risk_cat   = analysis["risk_category"]
    risk_pct   = analysis["risk_pct"]
    rc         = risk_colour(risk_cat)

    # ── Row 1: KPI cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card risk-{risk_cat.lower()}">'
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Dropout Risk</div>'
        f'<div style="font-size:44px;font-weight:800;color:{rc};line-height:1.1">{risk_pct}%</div>'
        f'<div style="font-size:11px;color:#9CA3AF">AI-estimated probability</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card risk-{risk_cat.lower()}">'
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Risk Category</div>'
        f'<div style="font-size:28px;font-weight:800;color:{rc};line-height:1.2;margin:4px 0">{risk_cat}</div>'
        f'<div style="font-size:11px;color:#9CA3AF">Threshold-based classification</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Attrition Window</div>'
        f'<div style="font-size:15px;font-weight:700;color:#0D1B2A;margin:8px 0 4px">{analysis["dropout_window"]}</div>'
        f'<div style="font-size:11px;color:#9CA3AF">Estimated timing</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Participant Profile</div>'
        f'<div style="font-size:13px;font-weight:600;color:#1D9E75;margin:8px 0 4px">{analysis["persona"]}</div>'
        f'<div style="font-size:11px;color:#9CA3AF">Archetype classification</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # ── Row 2: Risk gauge (hero) ──────────────────────────────────────────────
    _, g_mid, _ = st.columns([1, 2, 1])
    with g_mid:
        st.plotly_chart(gauge_chart(risk_score, "Estimated Dropout Probability", rc, height=260),
                        use_container_width=True)
    chart_caption(
        "Green zone (0–30%) — Low risk, standard monitoring. "
        "Amber zone (30–60%) — Moderate risk, proactive engagement recommended. "
        "Red zone (60–100%) — High risk, intervention plan indicated. "
        "Modelled estimate — interpret alongside clinical judgement."
    )

    # ── Why Is This Participant At Risk? ──────────────────────────────────────
    section_header("Why Is This Participant At Risk?")
    risk_factors = analysis.get("top3_risk_factors", [])
    protective   = analysis.get("top3_protective_factors", [])

    if risk_factors or protective:
        rf_col, pf_col = st.columns(2)
        with rf_col:
            st.markdown("**🔴 Top Risk Drivers**", help="Factors increasing dropout probability.")
            for _, sv, label in risk_factors:
                pct_str = f"+{abs(sv)*100:.0f}%"
                icon    = _icon(label)
                st.markdown(
                    f'<div class="driver-card">'
                    f'<div class="driver-icon">{icon}</div>'
                    f'<div class="driver-label">{label}</div>'
                    f'<div class="driver-pct">{pct_str}</div></div>',
                    unsafe_allow_html=True,
                )
            if not risk_factors:
                st.info("No significant risk factors identified.")
        with pf_col:
            st.markdown("**🟢 Protective Factors**", help="Factors reducing dropout probability.")
            for _, sv, label in protective:
                pct_str = f"{sv*100:.0f}%"
                icon    = _icon(label)
                st.markdown(
                    f'<div class="protect-card">'
                    f'<div class="driver-icon">{icon}</div>'
                    f'<div class="driver-label">{label}</div>'
                    f'<div class="protect-pct">{pct_str}</div></div>',
                    unsafe_allow_html=True,
                )
            if not protective:
                st.info("No significant protective factors identified.")

        # SHAP bar chart in expander
        with st.expander("View Full SHAP Attribution Chart"):
            all_shap = [(label, sv) for _, sv, label in risk_factors] + \
                       [(label, sv) for _, sv, label in protective]
            if all_shap:
                fig_shap = go.Figure(go.Bar(
                    x=[x[1] for x in all_shap],
                    y=[x[0] for x in all_shap],
                    orientation="h",
                    marker_color=["#D9534F" if v > 0 else "#2E8B57" for _, v in all_shap],
                    hovertemplate="<b>%{y}</b><br>SHAP contribution: %{x:.3f}<extra></extra>",
                ))
                fig_shap.update_layout(
                    xaxis_title="Contribution to dropout risk (positive = increases risk)",
                    height=280,
                    margin=dict(l=0, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                chart_caption(
                    "SHAP (SHapley Additive exPlanations) values make AI decisions transparent and auditable. "
                    "Bar length = magnitude of each factor's contribution to this specific prediction."
                )

    # ── Evidence-Based Foundation ─────────────────────────────────────────────
    section_header("Evidence-Based Foundation")
    evidence_list = analysis.get("evidence", [])
    ev_shown = 0
    for item in evidence_list:
        ev = item.get("evidence")
        if ev and ev_shown < 3:
            with st.expander(f"📖 {item['intervention']}"):
                st.markdown(f"**Source:** {ev['source']}")
                st.markdown(f"**Evidence:** {ev['evidence']}")
                st.markdown(f"**Clinical Recommendation:** {ev['recommendation']}")
            ev_shown += 1
    if ev_shown == 0:
        st.info("No specific clinical evidence linked for this participant's current profile.")

    # ── Retention Action Plan ─────────────────────────────────────────────────
    section_header("Retention Action Plan")
    interventions = analysis.get("interventions", [])
    if interventions:
        priority_labels = ["HIGH", "HIGH", "MEDIUM", "MEDIUM", "LOW"]
        priority_css    = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}

        for i, iv in enumerate(interventions):
            priority = iv.get("priority", priority_labels[min(i, len(priority_labels) - 1)])
            badge_cls = priority_css.get(priority.upper(), "badge-low")
            cost_val  = iv.get("cost_usd", 0) or iv.get("cost", 0)
            benefit   = iv.get("potential_risk_reduction_pct", 0) or iv.get("risk_reduction", 0)
            owner     = iv.get("owner", iv.get("responsible_team", "Clinical Operations"))
            rationale = iv.get("pharmd_rationale", iv.get("rationale", ""))

            st.markdown(
                f'<div class="iv-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">'
                f'<div class="iv-title">{iv["name"]}</div>'
                f'<span class="{badge_cls}">{priority}</span>'
                f'</div>'
                f'<div class="iv-rationale">{rationale}</div>'
                f'<div class="iv-row">'
                f'<div class="iv-stat">Responsible Team<br><strong>{owner}</strong></div>'
                f'<div class="iv-stat">Est. Risk Reduction<br><strong>~{benefit:.0f}%</strong></div>'
                f'<div class="iv-stat">Est. Cost<br><strong>${cost_val:,}</strong></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        chart_caption(
            "Recommendations are generated from this participant's specific risk drivers. "
            "Costs and benefit estimates are modelled approximations. Clinical review required before action."
        )
    else:
        st.success(
            "No targeted interventions indicated at the current risk level. "
            "Continue standard monitoring and engagement protocols."
        )

    # ── Coordinator Copilot (V3) ──────────────────────────────────────────────
    try:
        render_coordinator_copilot(analysis, risk_cat)
    except Exception:
        pass

    # ── Protocol Change Simulator ─────────────────────────────────────────────
    section_header("Protocol Change Simulator")
    from scenario_simulator import PRESET_SCENARIOS, simulate_scenario

    st.markdown(
        "Select a protocol adjustment to model its estimated impact on this participant's attrition risk."
    )
    sc_cols = st.columns(len(PRESET_SCENARIOS))
    for i, (sc, col) in enumerate(zip(PRESET_SCENARIOS, sc_cols)):
        if col.button(sc["label"], key=f"sc_{i}", use_container_width=True):
            with st.spinner("Simulating..."):
                result = simulate_scenario(patient_df, model, preprocessor, sc["changes"])

            orig_pct = round(result["original_risk"] * 100)
            new_pct  = round(result["new_risk"] * 100)
            delta    = orig_pct - new_pct
            orig_cat = "HIGH" if result["original_risk"] >= 0.6 else "MEDIUM" if result["original_risk"] >= 0.3 else "LOW"
            new_cat  = "HIGH" if result["new_risk"] >= 0.6 else "MEDIUM" if result["new_risk"] >= 0.3 else "LOW"

            b_col, a_col, d_col = st.columns([2, 1, 2])
            with b_col:
                st.markdown(
                    f'<div class="wif-before">'
                    f'<div class="wif-label">Current Risk</div>'
                    f'<div class="wif-pct" style="color:{risk_colour(orig_cat)}">{orig_pct}%</div>'
                    f'<div style="font-size:12px;color:#6B7280;margin-top:4px">{orig_cat}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with a_col:
                delta_color = "#2E8B57" if delta > 0 else "#D9534F"
                delta_icon  = "⬇" if delta > 0 else "⬆"
                st.markdown(
                    f'<div class="wif-arrow">'
                    f'<div style="font-size:28px;color:{delta_color}">{delta_icon}</div>'
                    f'<div class="delta-badge" style="background:{delta_color}">'
                    f'{abs(delta)}%</div>'
                    f'<div style="font-size:10px;color:#9CA3AF;margin-top:4px">reduction</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with d_col:
                st.markdown(
                    f'<div class="wif-after">'
                    f'<div class="wif-label">After Change</div>'
                    f'<div class="wif-pct" style="color:{risk_colour(new_cat)}">{new_pct}%</div>'
                    f'<div style="font-size:12px;color:#6B7280;margin-top:4px">{new_cat}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if delta > 0:
                st.success(f"✅ {result['interpretation']}")
            else:
                st.warning(f"⚠️ {result['interpretation']}")
            chart_caption(
                "Simulation modifies selected input features and re-scores the AI model. "
                "Results are modelled estimates only. Validate protocol changes through clinical and operational review."
            )

    # ── Estimated Economic Impact ─────────────────────────────────────────────
    section_header("Estimated Economic Impact")
    impact = analysis["business_impact"]

    if risk_cat == "LOW":
        st.success(
            "**Low-risk participant — below intervention threshold.** "
            "Estimated dropout risk is below 30%. No urgent financial escalation indicated. "
            "If dropout were to occur, estimated replacement cost: ~\$18,000 "
            "*(Getz KA et al., Ther Innov Regul Sci, 2016)*."
        )
    else:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Replacement Cost Avoided", f"${impact['replacement_cost_avoided']:,.0f}",
                  help="Estimated cost avoided if dropout is prevented.")
        b2.metric("Intervention Cost",         f"${impact['intervention_total_cost']:,.0f}",
                  help="Total estimated cost of all recommended retention actions.")
        b3.metric("Net Savings",               f"${impact['net_savings']:,.0f}")
        roi = impact["roi_ratio"]
        roi_str = (
            f"{roi:.1f}x"
            if roi not in (float("inf"), 0) and impact["intervention_total_cost"] > 0
            else "N/A"
        )
        b4.metric("Return on Investment", roi_str)

    st.caption(
        "_Modelled estimates. Based on \\$18,000 average participant replacement cost "
        "_(Getz KA et al., Ther Innov Regul Sci, 2016)_. Actual outcomes vary by therapeutic area and site._"
    )

    # ── Generate Report ───────────────────────────────────────────────────────
    section_header("Participant Report")
    # Re-generate report with document source metadata if available
    doc_source      = st.session_state.get("doc_source", "Manual Entry")
    copilot_summary = st.session_state.get("_copilot_summary", None)
    try:
        from report_generator import generate_report
        report_path = generate_report(
            analysis,
            patient_id=patient_df["patient_id"].iloc[0],
            doc_source=doc_source,
            copilot_summary=copilot_summary,
        )
        analysis["report_path"] = str(report_path)
    except Exception:
        pass

    report_path = analysis.get("report_path")
    if report_path and Path(report_path).exists():
        with open(report_path, "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f,
                file_name=f"retention_assessment_{patient_df['patient_id'].iloc[0]}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        st.caption(
            "_Report includes: risk summary, SHAP explanation, intervention action plan, "
            "financial impact estimate, and supporting evidence citations._"
        )
    else:
        st.info("PDF generation requires fpdf2. Install via `pip install fpdf2`.")


# ── TAB 2: Trial Operations Dashboard ────────────────────────────────────────
def render_tab2(config: dict):
    if not DATA_PATH.exists():
        st.warning("Dataset not found. Run `python src/data_generator.py` first.")
        return
    try:
        model, preprocessor = load_model_artefacts()
        df = load_dataset()
    except Exception as e:
        st.error(f"Could not load data or model: {e}")
        return

    from feature_engineering import add_composite_features, get_feature_columns
    from business_impact import calculate_population_impact

    df_fe = add_composite_features(df.copy())
    cols  = get_feature_columns()
    feat  = [c for c in cols["numerical"] + cols["categorical"] + cols["composite"] if c in df_fe.columns]
    probs = model.predict_proba(preprocessor.transform(df_fe[feat]))[:, 1]
    df["risk_score"] = probs

    med_t    = config["thresholds"]["medium_risk"]
    high_n   = int((probs >= med_t).sum())
    dropouts = int(df["dropout"].sum())
    attr_pct = df["dropout"].mean() * 100
    cost_risk = dropouts * config["costs"]["patient_replacement_cost"]

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Total Participants</div>'
        f'<div class="kpi-value">{len(df):,}</div>'
        f'<div class="kpi-sub">Full cohort</div></div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        f'<div class="kpi-card" style="border-left-color:#D9534F">'
        f'<div class="kpi-label">High-Risk Participants</div>'
        f'<div class="kpi-value" style="color:#D9534F">{high_n:,}</div>'
        f'<div class="kpi-sub">Require priority attention &uarr;</div></div>',
        unsafe_allow_html=True,
    )
    k3.markdown(
        f'<div class="kpi-card" style="border-left-color:#F4B942">'
        f'<div class="kpi-label">Projected Attrition</div>'
        f'<div class="kpi-value" style="color:#F4B942">{attr_pct:.1f}%</div>'
        f'<div class="kpi-sub">Observed in cohort &mdash;</div></div>',
        unsafe_allow_html=True,
    )
    k4.markdown(
        f'<div class="kpi-card" style="border-left-color:#D9534F">'
        f'<div class="kpi-label">Financial Exposure</div>'
        f'<div class="kpi-value" style="color:#D9534F">${cost_risk:,.0f}</div>'
        f'<div class="kpi-sub">If unaddressed &uarr;</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

    # Site performance
    section_header("Site Performance Analysis")
    site_stats = (
        df.groupby("site_id")
        .agg(enrolled=("patient_id", "count"), dropouts=("dropout", "sum"),
             avg_risk=("risk_score", "mean"))
        .reset_index()
    )
    site_stats["dropout_rate"] = (site_stats["dropouts"] / site_stats["enrolled"] * 100).round(1)
    site_stats["Status"] = site_stats["dropout_rate"].apply(
        lambda r: "🔴 Review Recommended" if r > 35 else ("🟡 Monitor" if r > 25 else "🟢 On Track")
    )
    st.dataframe(
        site_stats[["site_id", "enrolled", "dropouts", "dropout_rate", "avg_risk", "Status"]]
        .rename(columns={
            "site_id": "Site", "enrolled": "Enrolled",
            "dropouts": "Observed Attrition", "dropout_rate": "Attrition Rate (%)",
            "avg_risk": "Avg Risk Score",
        })
        .round({"Avg Risk Score": 3}),
        use_container_width=True, hide_index=True,
    )
    chart_caption(
        "Sites above 35% attrition flagged for sponsor review. "
        "Average Risk Score = model-estimated participant vulnerability at each site."
    )

    # Operational alerts
    section_header("Operational Alert Panel")
    critical = site_stats[site_stats["dropout_rate"] > 35]["site_id"].tolist()
    monitor  = site_stats[(site_stats["dropout_rate"] > 25) & (site_stats["dropout_rate"] <= 35)]["site_id"].tolist()
    for s in critical:
        r = site_stats.loc[site_stats["site_id"] == s, "dropout_rate"].values[0]
        st.markdown(
            f'<div class="alert-critical">'
            f'<div style="font-size:18px">🔴</div>'
            f'<div><strong>{s}</strong> — Attrition rate {r:.1f}%. '
            f'Site performance review recommended. Consider sponsor support visit and enhanced monitoring.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    for s in monitor:
        r = site_stats.loc[site_stats["site_id"] == s, "dropout_rate"].values[0]
        st.markdown(
            f'<div class="alert-monitor">'
            f'<div style="font-size:18px">🟡</div>'
            f'<div><strong>{s}</strong> — Attrition rate {r:.1f}%. '
            f'Monitor closely. Proactive participant engagement and investigator check-in recommended.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if not critical and not monitor:
        st.markdown(
            '<div class="alert-ok">🟢 All sites within acceptable attrition thresholds. Continue standard monitoring.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Charts side-by-side
    ch1, ch2 = st.columns(2)
    with ch1:
        section_header("Attrition Rate by Site")
        fig_bar = px.bar(
            site_stats, x="site_id", y="dropout_rate",
            color="dropout_rate",
            color_continuous_scale=["#2E8B57", "#F4B942", "#D9534F"],
            text="dropout_rate",
            labels={"site_id": "Site", "dropout_rate": "Attrition (%)"},
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(
            height=380, showlegend=False, coloraxis_showscale=False,
            yaxis_title="Attrition %", xaxis_title="Site",
            margin=dict(l=50, r=10, t=20, b=50),
            yaxis=dict(range=[0, site_stats["dropout_rate"].max() * 1.28]),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        chart_caption("Red bars exceed the 35% attrition threshold and are flagged for review.")

    with ch2:
        section_header("Population Risk Distribution")
        risk_bins = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0],
                           labels=["Low (0–30%)", "Moderate (30–60%)", "High (60–100%)"])
        risk_counts = risk_bins.value_counts().reindex(
            ["Low (0–30%)", "Moderate (30–60%)", "High (60–100%)"]
        )
        fig_dist = go.Figure(go.Bar(
            x=risk_counts.index.tolist(),
            y=risk_counts.values,
            marker_color=["#2E8B57", "#F4B942", "#D9534F"],
            text=risk_counts.values,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Participants: %{y}<extra></extra>",
        ))
        fig_dist.update_layout(
            height=380,
            xaxis_title="Risk Tier", yaxis_title="Participants",
            margin=dict(l=50, r=10, t=20, b=50),
            yaxis=dict(range=[0, int(risk_counts.max() * 1.2)]),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        chart_caption("High-risk participants warrant priority retention outreach and intervention planning.")

    st.divider()

    # Population economic impact
    section_header("Population-Level Economic Impact")
    pop = calculate_population_impact(df, model_recall=0.75, config=config)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("High-Risk Identified",    pop["high_risk_identified"])
    p2.metric("Attrition Preventable",   pop["dropouts_prevented"],
              help="Modelled estimate: 60% intervention success rate.")
    p3.metric("Potential Total Savings", f"${pop['total_savings']:,.0f}")
    p4.metric("Estimated Net Benefit",   f"${pop['net_benefit']:,.0f}")
    chart_caption(
        "Modelled estimates using synthetic data. Assumes 75% model recall, 60% intervention success. "
        "Not a guarantee of financial outcomes."
    )


# ── TAB 3: AI Intelligence Engine ────────────────────────────────────────────
def render_tab3():
    def show_img(path: Path, explanation: str):
        if path.exists():
            st.image(str(path), use_container_width=True)
            chart_caption(explanation)
        else:
            st.info(f"`{path.name}` not yet generated — run the model training pipeline first.")

    section_header("Model Performance Overview")
    st.markdown(
        "Five candidate models evaluated on a held-out test set. "
        "**Recall is prioritised:** a missed dropout (false negative) carries higher cost than a false alarm."
    )

    c1, c2 = st.columns(2)
    with c1:
        show_img(OUTPUTS_DIR / "roc_curve_comparison.png",
                 "ROC Curve: shows sensitivity vs. specificity across all decision thresholds. "
                 "Higher AUC = stronger discrimination between dropouts and retainers.")
    with c2:
        show_img(OUTPUTS_DIR / "calibration_curve.png",
                 "Calibration Curve: shows whether predicted probabilities match real dropout frequencies. "
                 "Well-calibrated models are critical in clinical applications where scores drive intervention decisions.")

    c3, c4 = st.columns(2)
    with c3:
        show_img(OUTPUTS_DIR / "confusion_matrix.png",
                 "Confusion Matrix: True Positives (correctly identified dropouts) are the priority. "
                 "False Negatives (missed dropouts) = highest-cost error in retention contexts.")
    with c4:
        show_img(OUTPUTS_DIR / "precision_recall_curve.png",
                 "Precision-Recall Curve: especially informative for imbalanced datasets. "
                 "Shows the trade-off between recall (catching at-risk participants) and precision.")

    section_header("AI Transparency & Explainability")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) reveals which characteristics drive predictions "
        "across the entire population, supporting clinical validation, regulatory transparency, and stakeholder trust."
    )

    c5, c6 = st.columns(2)
    with c5:
        show_img(OUTPUTS_DIR / "shap_summary_beeswarm.png",
                 "SHAP Beeswarm: each dot = one participant. X-axis = impact on dropout risk. "
                 "Red = high feature value. Features ordered by population-level importance.")
    with c6:
        show_img(OUTPUTS_DIR / "shap_bar_plot.png",
                 "SHAP Bar Plot: mean absolute SHAP values rank global feature importance. "
                 "Week 2 Side Effect Severity dominates — consistent with pharmacovigilance literature.")

    c7, c8 = st.columns(2)
    with c7:
        show_img(OUTPUTS_DIR / "shap_dependence_side_effects.png",
                 "Dependence Plot — Week 2 Side Effects: positive trend confirms early adverse event burden "
                 "is the most critical and addressable intervention window (ICH E6(R2), 2016).")
    with c8:
        show_img(OUTPUTS_DIR / "shap_dependence_distance.png",
                 "Dependence Plot — Distance from Site: risk rises steeply beyond 50 km without transport. "
                 "Consistent with FDA participant convenience guidance (2012).")

    section_header("Survival Analysis — Attrition Timing")
    show_img(OUTPUTS_DIR / "survival_curve.png",
             "Kaplan-Meier Curves: probability of remaining in trial over time by risk tier. "
             "Clear separation confirms the model's risk scores predict not just WHO will drop out, but WHEN.")

    section_header("Model Comparison")
    model_tbl = pd.DataFrame([
        {"Model": "Logistic Regression ⭐ (primary)", "AUC": "0.694", "Recall": "0.779", "F1": "0.531", "Brier": "0.216", "Role": "Primary prediction model"},
        {"Model": "Random Forest",                    "AUC": "0.668", "Recall": "0.442", "F1": "0.435", "Brier": "0.200", "Role": "Benchmark"},
        {"Model": "XGBoost (Optuna-tuned)",           "AUC": "0.640", "Recall": "0.411", "F1": "0.429", "Brier": "0.243", "Role": "SHAP explainability model"},
        {"Model": "LightGBM",                         "AUC": "0.660", "Recall": "0.316", "F1": "0.387", "Brier": "0.219", "Role": "Benchmark"},
        {"Model": "CatBoost",                         "AUC": "0.663", "Recall": "0.432", "F1": "0.443", "Brier": "0.205", "Role": "Benchmark"},
    ])
    st.dataframe(model_tbl, use_container_width=True, hide_index=True)
    chart_caption(
        "Logistic Regression selected for highest recall (0.779). "
        "XGBoost used for SHAP TreeExplainer (exact, non-approximated values). "
        "All metrics approximate — synthetic data only."
    )


# ── TAB 4: About the Platform ─────────────────────────────────────────────────
def render_tab4():

    # Clinical Challenge — stat cards + short text
    section_header("The Clinical Challenge")
    s1, s2, s3 = st.columns(3)
    s1.markdown(
        '<div class="challenge-stat"><div class="challenge-num">20–30%</div>'
        '<div class="challenge-txt">Average attrition rate across clinical trials</div></div>',
        unsafe_allow_html=True,
    )
    s2.markdown(
        '<div class="challenge-stat"><div class="challenge-num">$18K+</div>'
        '<div class="challenge-txt">Estimated cost to replace one dropout participant</div></div>',
        unsafe_allow_html=True,
    )
    s3.markdown(
        '<div class="challenge-stat"><div class="challenge-num">$54M+</div>'
        '<div class="challenge-txt">Industry-wide annual cost exposure from attrition</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)
    st.markdown(
        "Current retention strategies are predominantly **reactive** — triggered after dropout occurs. "
        "This platform demonstrates how AI-driven early identification, combined with evidence-based "
        "interventions, can shift clinical trial operations from reactive to **proactive**."
    )
    st.caption("Sources: FDA (2012); Getz KA et al., Ther Innov Regul Sci (2016).")

    # Clinical Intelligence Framework
    section_header("Clinical Intelligence Framework")
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(
        '<div class="about-card"><div style="font-size:20px">⚠️</div>'
        '<div style="font-weight:700;margin:6px 0 4px;font-size:13px">Pharmacological Burden</div>'
        '<div style="font-size:12px;color:#6B7280">Side effect severity, polypharmacy risk, adverse event history</div></div>',
        unsafe_allow_html=True,
    )
    b2.markdown(
        '<div class="about-card"><div style="font-size:20px">🚗</div>'
        '<div style="font-weight:700;margin:6px 0 4px;font-size:13px">Logistical Friction</div>'
        '<div style="font-size:12px;color:#6B7280">Distance, visit frequency, transportation access, scheduling burden</div></div>',
        unsafe_allow_html=True,
    )
    b3.markdown(
        '<div class="about-card"><div style="font-size:20px">🧠</div>'
        '<div style="font-weight:700;margin:6px 0 4px;font-size:13px">Psychosocial Factors</div>'
        '<div style="font-size:12px;color:#6B7280">Consent complexity, participant-investigator relationship, anxiety</div></div>',
        unsafe_allow_html=True,
    )
    b4.markdown(
        '<div class="about-card"><div style="font-size:20px">📋</div>'
        '<div style="font-weight:700;margin:6px 0 4px;font-size:13px">Protocol Design</div>'
        '<div style="font-size:12px;color:#6B7280">Assessment load, Phase 1 uncertainty, complexity interactions</div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("View engineered composite features"):
        st.markdown(
            "| Feature | Clinical Purpose |\n|---------|------------------|\n"
            "| Visit Burden Index | Visit frequency × trial duration — captures participant fatigue |\n"
            "| Polypharmacy Risk Score | Multi-drug complexity and management burden |\n"
            "| Patient Burden Score | Aggregated physical, logistical, and clinical load |\n"
            "| Logistic Friction Score | Distance adjusted for transportation access |\n"
            "| Phase-Complexity Interaction | Risk amplification in early-phase, high-complexity trials |"
        )

    # Clinical Pharmacist's Perspective
    section_header("Clinical Pharmacist's Perspective")
    p1, p2, p3 = st.columns(3)
    p1.markdown(
        '<div class="about-card" style="border-left-color:#D9534F">'
        '<div style="font-size:18px">⚠️</div>'
        '<div style="font-weight:700;font-size:13px;margin:6px 0 4px">Week 2 Side Effects</div>'
        '<div style="font-size:12px;color:#6B7280">The strongest single predictor. SHAP contribution is ~3× larger than the next factor. '
        'Proactive pharmacovigilance contact at Week 2 is low-cost and high-impact.</div>'
        '<div style="font-size:11px;color:#9CA3AF;margin-top:6px">ICH E6(R2), 2016</div></div>',
        unsafe_allow_html=True,
    )
    p2.markdown(
        '<div class="about-card" style="border-left-color:#F4B942">'
        '<div style="font-size:18px">🚗</div>'
        '<div style="font-weight:700;font-size:13px;margin:6px 0 4px">Distance Barrier</div>'
        '<div style="font-size:12px;color:#6B7280">Beyond 50 km without transport: markedly elevated risk, independent of clinical profile. '
        'Transportation reimbursement delivers significant risk reduction at low cost.</div>'
        '<div style="font-size:11px;color:#9CA3AF;margin-top:6px">FDA, 2012</div></div>',
        unsafe_allow_html=True,
    )
    p3.markdown(
        '<div class="about-card" style="border-left-color:#1D9E75">'
        '<div style="font-size:18px">📋</div>'
        '<div style="font-weight:700;font-size:13px;margin:6px 0 4px">Protocol Complexity</div>'
        '<div style="font-size:12px;color:#6B7280">ICH E6(R2) supports eliminating non-critical assessments. '
        'Complexity reduction at design stage is the most upstream retention intervention available.</div>'
        '<div style="font-size:11px;color:#9CA3AF;margin-top:6px">Getz KA et al., 2016</div></div>',
        unsafe_allow_html=True,
    )

    # AI Development Framework
    section_header("AI Development Framework")
    dev_tbl = pd.DataFrame([
        {"Component": "Dataset",           "Detail": "2,000 synthetic participants, clinically-informed distributions"},
        {"Component": "Attrition Rate",    "Detail": "31.6% observed — aligned with industry literature"},
        {"Component": "Features",          "Detail": "25 inputs + 5 PharmD-engineered composite features"},
        {"Component": "Data Split",        "Detail": "70/15/15 train/val/test — stratified by dropout label"},
        {"Component": "Class Imbalance",   "Detail": "SMOTE on training set only — no data leakage"},
        {"Component": "Models Evaluated",  "Detail": "Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost"},
        {"Component": "Tuning",            "Detail": "Optuna (50 trials) applied to XGBoost"},
        {"Component": "Tracking",          "Detail": "MLflow with SQLite backend — all runs logged"},
        {"Component": "Selection",         "Detail": "Recall prioritised — false negative = higher cost than false alarm"},
    ])
    st.dataframe(dev_tbl, use_container_width=True, hide_index=True)

    # AI Transparency & Explainability
    section_header("AI Transparency & Explainability")
    e1, e2, e3 = st.columns(3)
    e1.markdown(
        '<div class="about-card"><div style="font-size:20px">🌍</div>'
        '<div style="font-weight:700;font-size:13px;margin:6px 0 4px">Global Explainability</div>'
        '<div style="font-size:12px;color:#6B7280">SHAP beeswarm and bar plots reveal population-level feature importance — '
        'supporting clinical validation and regulatory transparency.</div></div>',
        unsafe_allow_html=True,
    )
    e2.markdown(
        '<div class="about-card"><div style="font-size:20px">👤</div>'
        '<div style="font-weight:700;font-size:13px;margin:6px 0 4px">Per-Participant Explainability</div>'
        '<div style="font-size:12px;color:#6B7280">SHAP waterfall values show exactly which factors increased or '
        'decreased dropout risk for each individual — enabling personalised intervention targeting.</div></div>',
        unsafe_allow_html=True,
    )
    e3.markdown(
        '<div class="about-card"><div style="font-size:20px">⚡</div>'
        '<div style="font-weight:700;font-size:13px;margin:6px 0 4px">TreeExplainer</div>'
        '<div style="font-size:12px;color:#6B7280">Exact (non-approximated) SHAP values for XGBoost — '
        'computationally efficient for real-time inference in live dashboards.</div></div>',
        unsafe_allow_html=True,
    )

    # Evidence Sources
    section_header("Evidence Sources")
    ev_tbl = pd.DataFrame([
        {"Domain": "Transportation barriers",  "Reference": "FDA (2012) — Patient Retention in Clinical Trials", "Application": "Transportation support interventions"},
        {"Domain": "Adverse event management", "Reference": "ICH E6(R2) Good Clinical Practice (2016)",         "Application": "Week 2 pharmacovigilance calls"},
        {"Domain": "Protocol complexity",      "Reference": "Getz KA et al., Ther Innov Regul Sci (2016)",      "Application": "Protocol simplification recommendations"},
        {"Domain": "Visit burden / DCT",       "Reference": "FDA Decentralized Clinical Trials Guidance (2023)", "Application": "Visit frequency reduction scenarios"},
        {"Domain": "Investigator performance", "Reference": "ICH E6(R2) Section 4",                             "Application": "Site quality improvement alerts"},
        {"Domain": "Polypharmacy risk",        "Reference": "WHO Technical Report — Polypharmacy",              "Application": "Medication management support"},
        {"Domain": "Consent complexity",       "Reference": "FDA Plain Language Guidance (2014)",               "Application": "Consent simplification flagging"},
    ])
    st.dataframe(ev_tbl, use_container_width=True, hide_index=True)

    # System Architecture — inline diagram
    section_header("System Architecture")
    arch_img = OUTPUTS_DIR / "architecture.png"
    if arch_img.exists():
        st.image(str(arch_img), use_container_width=True)
    else:
        st.markdown(
            '<div class="arch-flow">'
            '<div class="arch-box">🗂️<br>Synthetic<br>Data</div>'
            '<div class="arch-arrow">→</div>'
            '<div class="arch-box">⚗️<br>Feature<br>Engineering</div>'
            '<div class="arch-arrow">→</div>'
            '<div class="arch-box arch-box-teal">🤖<br>Model<br>Training</div>'
            '<div class="arch-arrow">→</div>'
            '<div class="arch-box arch-box-teal">🔍<br>SHAP<br>Explainability</div>'
            '<div class="arch-arrow">→</div>'
            '<div class="arch-box">🎯<br>Intervention<br>Engine</div>'
            '<div class="arch-arrow">→</div>'
            '<div class="arch-box arch-box-amber">💰<br>Business<br>Impact</div>'
            '<div class="arch-arrow">→</div>'
            '<div class="arch-box">📄<br>Report<br>Generator</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    chart_caption(
        "End-to-end pipeline: data generation → PharmD feature engineering → "
        "multi-model training (MLflow) → SHAP explainability → evidence retrieval → "
        "intervention engine → business impact → 9-step agent → PDF report."
    )

    # Limitations
    section_header("Limitations & Scope")
    with st.expander("View full limitations statement"):
        st.markdown(
            "- **Synthetic data only.** No real patient records or sponsor datasets used.\n"
            "- **Modelled estimates.** Intervention effectiveness figures are approximate, not validated outcomes.\n"
            "- **External validity not established.** Performance on real trial populations may differ substantially.\n"
            "- **Proof of concept.** Demonstrates methodology — not a validated clinical decision support tool.\n"
            "- **Timing approximations.** Dropout windows derived from simulated distributions, not clinical predictions."
        )

    # Release History
    section_header("Release History")
    st.markdown("""
<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px">

<div style="flex:1;min-width:220px;background:#F0FDF4;border:1.5px solid #1D9E75;border-radius:10px;padding:16px">
<div style="font-size:11px;font-weight:700;color:#1D9E75;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">Version 1.0 — Deployed</div>
<div style="font-weight:700;font-size:14px;color:#0D1B2A;margin-bottom:8px">Manual Participant Entry</div>
<ul style="font-size:12px;color:#374151;margin:0;padding-left:18px;line-height:1.8">
<li>Manual participant data entry</li>
<li>XGBoost dropout risk prediction</li>
<li>SHAP per-participant explainability</li>
<li>7 evidence-based interventions</li>
<li>Business impact &amp; ROI calculator</li>
<li>What-if scenario simulator</li>
<li>Downloadable 2-page PDF report</li>
</ul>
</div>

<div style="flex:1;min-width:220px;background:#EFF6FF;border:1.5px solid #3B82F6;border-radius:10px;padding:16px">
<div style="font-size:11px;font-weight:700;color:#3B82F6;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">Version 2.0 — Current</div>
<div style="font-weight:700;font-size:14px;color:#0D1B2A;margin-bottom:8px">Clinical Document Intake & Auto-Population</div>
<ul style="font-size:12px;color:#374151;margin:0;padding-left:18px;line-height:1.8">
<li>Clinical PDF upload (CRF, screening forms)</li>
<li>Rule-based extraction of 16 clinical fields</li>
<li>Confidence scoring (High / Medium / Low)</li>
<li>Human-in-the-loop validation &amp; editing</li>
<li>Sidebar auto-population on confirmation</li>
<li>Extraction audit trail with timestamps</li>
<li>PDF report metadata: source &amp; method</li>
</ul>
</div>

<div style="flex:1;min-width:220px;background:#FFF7ED;border:1.5px dashed #F59E0B;border-radius:10px;padding:16px;opacity:0.85">
<div style="font-size:11px;font-weight:700;color:#D97706;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">Version 3.0 — Planned</div>
<div style="font-weight:700;font-size:14px;color:#0D1B2A;margin-bottom:8px">AI Clinical Intake Agent</div>
<ul style="font-size:12px;color:#374151;margin:0;padding-left:18px;line-height:1.8">
<li>Clinical entity recognition (NER)</li>
<li>Medical NLP for unstructured notes</li>
<li>Confidence scoring engine (model-driven)</li>
<li>Intelligent missing data detection</li>
<li>Agentic clinical intake workflow</li>
<li>Multi-document source reconciliation</li>
</ul>
<div style="font-size:11px;color:#D97706;font-weight:600;margin-top:8px">⚠️ Roadmap only — not implemented</div>
</div>

</div>
""", unsafe_allow_html=True)

    # Disclaimer
    section_header("Disclaimer")
    st.warning(
        "⚠️ **This platform is intended for educational, research, and portfolio demonstration purposes only. "
        "It does not provide clinical recommendations and must not be used for patient care decisions, "
        "regulatory submissions, or operational sponsor decision-making.**"
    )

    # Author
    section_header("Author")
    st.markdown(
        '<div class="about-card" style="border-left-color:#1D9E75;max-width:480px">'
        '<div style="font-size:16px;font-weight:800;color:#0D1B2A;margin-bottom:4px">'
        'Dr. Reema Mohamed Sulthan</div>'
        '<div style="font-size:13px;color:#1D9E75;font-weight:600;margin-bottom:8px">'
        'PharmD &nbsp;|&nbsp; Clinical Data Scientist &nbsp;|&nbsp; Certified AI Expert</div>'
        '<div style="display:flex;gap:16px;flex-wrap:wrap">'
        '<a href="mailto:reemahussain2097@gmail.com" style="font-size:12px;color:#1D9E75;text-decoration:none">📧 Email</a>'
        '<a href="https://github.com/reemahussain-pharmd" target="_blank" style="font-size:12px;color:#1D9E75;text-decoration:none">🐙 GitHub</a>'
        '<a href="https://linkedin.com/in/reema-mohamed-sulthan" target="_blank" style="font-size:12px;color:#1D9E75;text-decoration:none">💼 LinkedIn</a>'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ── Landing header ────────────────────────────────────────────────────────────
def render_landing():
    st.markdown(
        "<div style='padding:4px 0 2px'>"
        "<div style='font-size:26px;font-weight:800;color:#0D1B2A;line-height:1.2'>"
        "🧬 AI-Powered Clinical Trial Retention Intelligence</div>"
        "<div style='font-size:14px;color:#1D9E75;font-weight:600;margin:6px 0 4px'>"
        "Predict &nbsp;·&nbsp; Explain &nbsp;·&nbsp; Intervene &nbsp;·&nbsp; Simulate &nbsp;·&nbsp; Report"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # 5 capability cards
    cols = st.columns(5)
    caps = [
        ("🧠", "Predict",   "Dropout risk per participant"),
        ("🔍", "Explain",   "SHAP explainability"),
        ("🎯", "Intervene", "Evidence-based recommendations"),
        ("📈", "Simulate",  "What-if scenario analysis"),
        ("📄", "Report",    "Sponsor-ready PDF output"),
    ]
    for col, (icon, title, desc) in zip(cols, caps):
        col.markdown(
            f'<div class="cap-card">'
            f'<div class="cap-icon">{icon}</div>'
            f'<div class="cap-title">{title}</div>'
            f'<div class="cap-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='font-size:11px;color:#9CA3AF;margin:8px 0 4px'>"
        "⚠️ For educational, research, and portfolio demonstration purposes only. Not for clinical use.</div>",
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Apply intake-confirmed values BEFORE the sidebar renders its widgets.
    # This is required by Streamlit: widget-bound keys can only be set before
    # the widget appears in the current run.
    if "_intake_pending" in st.session_state:
        for k, v in st.session_state["_intake_pending"].items():
            st.session_state[k] = v
        del st.session_state["_intake_pending"]

    render_landing()
    render_landing_kpis()
    st.divider()

    config     = load_config()
    patient_df = render_sidebar()

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Clinical Document Intake",
        "🧬 Participant Retention Assessment",
        "📊 Trial Operations Dashboard",
        "🤖 AI Intelligence Engine",
        "📁 Multi-Participant Screening",
        "ℹ️ About the Platform",
    ])

    with tab0:
        render_tab_intake()
    with tab1:
        render_tab1(patient_df, config)
    with tab2:
        render_tab2(config)
    with tab3:
        render_tab3()
    with tab4:
        render_tab_batch()
    with tab5:
        render_tab4()


if __name__ == "__main__":
    main()
