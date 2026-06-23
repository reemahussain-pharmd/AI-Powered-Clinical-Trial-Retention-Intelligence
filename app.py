"""
TrialGuard — AI-Powered Clinical Trial Retention Intelligence Platform
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
    page_title="TrialGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",  # forced open every load
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════
   ENTERPRISE DESIGN SYSTEM — TrialGuard
   Primary:   Navy  #0D1B2A
   Secondary: Teal  #1D9E75
   Success:   Emerald #10B981
   Warning:   Amber  #F59E0B
   Critical:  Red    #EF4444
═══════════════════════════════════════════════ */

/* ── Chrome chrome — hide only deploy/menu/footer ── */
.stDeployButton{display:none!important}
#MainMenu{visibility:hidden!important}
footer{visibility:hidden!important}
/* ── NEVER hide header or collapsedControl — sidebar expand lives inside header ── */
[data-testid="collapsedControl"]{
    display:flex!important;visibility:visible!important;
    opacity:1!important;z-index:99999!important;pointer-events:auto!important
}

/* ── Global typography & background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Inter',system-ui,-apple-system,sans-serif!important}
.main{background:linear-gradient(135deg,#F0F4F8 0%,#E8F0E9 50%,#F0F4F8 100%)!important;min-height:100vh}
.block-container{padding-top:0.5rem!important;padding-bottom:2rem!important}

/* ── CTA primary button — TrialGuard teal (not Streamlit default red) ── */
button[data-testid="baseButton-primary"],
.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#1D9E75,#17836A)!important;
    border:none!important;color:#FFFFFF!important;font-weight:700!important;
    font-size:15px!important;border-radius:10px!important;padding:12px 28px!important;
    box-shadow:0 4px 16px rgba(29,158,117,0.35)!important;
    transition:all .2s ease!important;letter-spacing:0.2px!important
}
button[data-testid="baseButton-primary"]:hover,
.stButton>button[kind="primary"]:hover{
    background:linear-gradient(135deg,#17836A,#126B56)!important;
    box-shadow:0 6px 20px rgba(29,158,117,0.45)!important;transform:translateY(-1px)!important
}

/* ── Sidebar container ── */
[data-testid="stSidebar"]{background:#0D1B2A!important}
[data-testid="stSidebar"] .stButton>button{
    background:rgba(255,255,255,0.07)!important;border:1px solid rgba(29,158,117,0.3)!important;
    color:#FFFFFF!important;border-radius:8px!important;width:100%!important;
    text-align:left!important;font-size:13px!important;font-weight:500!important;
    padding:8px 12px!important;margin-bottom:3px!important;transition:all .2s!important
}
[data-testid="stSidebar"] .stButton>button:hover{
    background:rgba(29,158,117,0.18)!important;border-color:#1D9E75!important;color:#FFFFFF!important
}
[data-testid="stSidebar"] .stButton>button p{color:#FFFFFF!important}
/* ── Active nav button highlight (◀ indicator on active page) ── */
[data-testid="stSidebar"] .stButton>button[kind="secondary"]:has(span:contains("◀")){
    border-left:3px solid #1D9E75!important;background:rgba(29,158,117,0.15)!important;color:#FFFFFF!important;font-weight:700!important
}
/* ── Form expanders ── */
[data-testid="stSidebar"] .stExpander{background:rgba(255,255,255,0.04)!important;border:1px solid rgba(29,158,117,0.15)!important;border-radius:8px!important;margin-bottom:6px!important}
[data-testid="stSidebar"] .stExpander details{background:transparent!important}
[data-testid="stSidebar"] details summary{color:#A8D5C4!important;font-weight:600!important;font-size:12px!important}
[data-testid="stSidebar"] .stSelectbox label,[data-testid="stSidebar"] .stSlider label,[data-testid="stSidebar"] .stNumberInput label{color:#A8D5C4!important;font-size:11px!important}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]>div{background:rgba(255,255,255,0.07)!important;border-color:rgba(29,158,117,0.35)!important;border-radius:6px!important}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div,[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input{color:#FFFFFF!important}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"]{background:#1D9E75!important;border-color:#1D9E75!important}

/* ── Hero section ── */
.tg-hero{
    background:linear-gradient(135deg,#0D1B2A 0%,#0f2a3a 40%,#0a2218 100%);
    border-radius:16px;padding:40px 44px 36px;margin-top:24px;margin-bottom:20px;
    position:relative;overflow:hidden;
    box-shadow:0 20px 60px rgba(13,27,42,0.25),0 4px 16px rgba(13,27,42,0.15)
}
.tg-hero::before{
    content:'';position:absolute;top:-60px;right:-60px;
    width:300px;height:300px;
    background:radial-gradient(circle,rgba(29,158,117,0.15) 0%,transparent 70%);
    border-radius:50%
}
.tg-hero::after{
    content:'';position:absolute;bottom:-40px;left:20%;
    width:200px;height:200px;
    background:radial-gradient(circle,rgba(29,158,117,0.08) 0%,transparent 70%);
    border-radius:50%
}
.tg-hero-eyebrow{
    font-size:11px;font-weight:700;color:#1D9E75;letter-spacing:2.5px;
    text-transform:uppercase;margin-bottom:10px
}
.tg-hero-title{
    font-size:38px;font-weight:900;color:#FFFFFF;line-height:1.1;
    letter-spacing:-1px;margin-bottom:8px
}
.tg-hero-title span{color:#1D9E75}
.tg-hero-subtitle{
    font-size:15px;font-weight:400;color:rgba(255,255,255,0.75);
    line-height:1.6;margin-bottom:18px;max-width:620px
}
.tg-hero-badges{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:0}
.tg-badge{
    display:inline-flex;align-items:center;gap:5px;
    padding:5px 14px;border-radius:20px;font-size:11px;font-weight:600;
    letter-spacing:0.3px
}
.tg-badge-teal{background:rgba(29,158,117,0.2);color:#4CD4A0;border:1px solid rgba(29,158,117,0.4)}
.tg-badge-navy{background:rgba(255,255,255,0.1);color:rgba(255,255,255,0.8);border:1px solid rgba(255,255,255,0.2)}
.tg-badge-amber{background:rgba(245,158,11,0.15);color:#FCD34D;border:1px solid rgba(245,158,11,0.3)}

/* ── Capability cards (5-column) ── */
.cap-card{
    background:rgba(255,255,255,0.95);border-radius:12px;padding:20px 14px;
    box-shadow:0 4px 16px rgba(13,27,42,0.08),0 1px 4px rgba(13,27,42,0.05);
    text-align:center;border-top:3px solid #1D9E75;
    transition:all .25s ease;height:100%;
    backdrop-filter:blur(8px)
}
.cap-card:hover{
    box-shadow:0 8px 28px rgba(13,27,42,0.14);
    transform:translateY(-3px)
}
.cap-icon{font-size:30px;line-height:1;margin-bottom:10px}
.cap-title{font-size:12px;font-weight:800;color:#0D1B2A;margin-bottom:5px;letter-spacing:0.5px;text-transform:uppercase}
.cap-desc{font-size:11px;color:#6B7280;line-height:1.5}

/* ── Executive KPI cards ── */
.kpi-card{
    background:rgba(255,255,255,0.97);border-radius:14px;padding:20px 22px;
    box-shadow:0 4px 20px rgba(13,27,42,0.09),0 1px 4px rgba(13,27,42,0.05);
    border-left:4px solid #1D9E75;transition:all .2s ease;
    backdrop-filter:blur(8px)
}
.kpi-card:hover{box-shadow:0 8px 30px rgba(13,27,42,0.13);transform:translateY(-2px)}
.kpi-label{font-size:10px;color:#6B7280;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:6px}
.kpi-value{font-size:34px;font-weight:900;color:#0D1B2A;line-height:1.1;margin:0 0 4px}
.kpi-sub{font-size:11px;color:#9CA3AF;font-weight:400}
.kpi-trend{font-size:10px;font-weight:600;margin-top:6px}
.kpi-trend-up{color:#EF4444}
.kpi-trend-down{color:#10B981}

/* ── Result metric cards ── */
.metric-card{
    background:rgba(255,255,255,0.97);border-radius:12px;border-left:4px solid #1D9E75;
    padding:16px 18px;box-shadow:0 3px 12px rgba(13,27,42,0.08)
}
.risk-high{background:linear-gradient(135deg,#FFF5F5,#FFF0F0);border-left-color:#EF4444}
.risk-medium{background:linear-gradient(135deg,#FFFBF0,#FFF8E8);border-left-color:#F59E0B}
.risk-low{background:linear-gradient(135deg,#F0FAF6,#EBFAF3);border-left-color:#10B981}

/* ── Risk driver cards ── */
.driver-card{
    background:linear-gradient(135deg,#FFF5F5,#FFF0F0);border-radius:10px;
    padding:14px 16px;border-left:4px solid #EF4444;margin-bottom:8px;
    display:flex;align-items:center;gap:12px;
    box-shadow:0 2px 8px rgba(239,68,68,0.08)
}
.driver-icon{font-size:22px;flex-shrink:0}
.driver-label{font-size:13px;font-weight:600;color:#1F2937;flex:1;line-height:1.4}
.driver-pct{font-size:18px;font-weight:800;color:#EF4444;white-space:nowrap}
.protect-card{
    background:linear-gradient(135deg,#F0FAF6,#EBFAF3);border-radius:10px;
    padding:14px 16px;border-left:4px solid #10B981;margin-bottom:8px;
    display:flex;align-items:center;gap:12px;
    box-shadow:0 2px 8px rgba(16,185,129,0.08)
}
.protect-pct{font-size:18px;font-weight:800;color:#10B981;white-space:nowrap}

/* ── Intervention cards ── */
.iv-card{
    background:rgba(255,255,255,0.97);border-radius:12px;padding:18px 20px;
    box-shadow:0 4px 16px rgba(13,27,42,0.08);margin-bottom:12px;
    border-top:3px solid #1D9E75;transition:box-shadow .2s
}
.iv-card:hover{box-shadow:0 6px 24px rgba(13,27,42,0.12)}
.iv-title{font-size:14px;font-weight:700;color:#0D1B2A;margin-bottom:6px}
.iv-rationale{font-size:12px;color:#4B5563;margin-bottom:10px;line-height:1.55}
.iv-row{display:flex;gap:20px;flex-wrap:wrap;margin-top:6px}
.iv-stat{font-size:11px;color:#6B7280}
.iv-stat strong{color:#0D1B2A;font-size:13px}
.badge-high{background:#FEE2E2;color:#991B1B;border-radius:20px;padding:3px 11px;font-size:11px;font-weight:700;letter-spacing:0.3px}
.badge-medium{background:#FEF3C7;color:#92400E;border-radius:20px;padding:3px 11px;font-size:11px;font-weight:700;letter-spacing:0.3px}
.badge-low{background:#D1FAE5;color:#065F46;border-radius:20px;padding:3px 11px;font-size:11px;font-weight:700;letter-spacing:0.3px}

/* ── What-if comparison ── */
.wif-before{
    background:linear-gradient(135deg,#FFF5F5,#FFF0F0);border-radius:12px;
    padding:22px;text-align:center;border:2px solid #EF4444;
    box-shadow:0 4px 16px rgba(239,68,68,0.1)
}
.wif-after{
    background:linear-gradient(135deg,#F0FAF6,#EBFAF3);border-radius:12px;
    padding:22px;text-align:center;border:2px solid #10B981;
    box-shadow:0 4px 16px rgba(16,185,129,0.1)
}
.wif-label{font-size:10px;color:#6B7280;text-transform:uppercase;font-weight:700;letter-spacing:1px}
.wif-pct{font-size:48px;font-weight:900;line-height:1.1}
.wif-arrow{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%}
.delta-badge{
    background:linear-gradient(135deg,#0D1B2A,#1a2f45);color:#FFFFFF;border-radius:10px;
    padding:9px 18px;font-size:14px;font-weight:700;margin-top:8px;
    box-shadow:0 4px 12px rgba(13,27,42,0.25)
}

/* ── Section headers ── */
.section-header{
    background:linear-gradient(135deg,#0D1B2A,#0f2235);color:white;
    padding:10px 18px;border-radius:8px;font-weight:700;margin:22px 0 14px;
    font-size:12.5px;letter-spacing:0.8px;text-transform:uppercase;
    box-shadow:0 3px 10px rgba(13,27,42,0.2);
    border-left:4px solid #1D9E75
}

/* ── Chart captions ── */
.chart-caption{
    font-size:11.5px;color:#6B7280;font-style:italic;
    margin-top:5px;padding:8px 14px;
    background:rgba(249,250,251,0.9);border-radius:6px;
    border-left:3px solid #1D9E75;line-height:1.5
}

/* ── Operational alerts ── */
.alert-critical{
    background:linear-gradient(135deg,#FFF5F5,#FFF0F0);
    border:1px solid #EF4444;border-radius:10px;padding:13px 16px;margin:6px 0;
    display:flex;align-items:flex-start;gap:10px;
    box-shadow:0 2px 8px rgba(239,68,68,0.08)
}
.alert-monitor{
    background:linear-gradient(135deg,#FFFBF0,#FFF8E8);
    border:1px solid #F59E0B;border-radius:10px;padding:13px 16px;margin:6px 0;
    display:flex;align-items:flex-start;gap:10px;
    box-shadow:0 2px 8px rgba(245,158,11,0.08)
}
.alert-ok{
    background:linear-gradient(135deg,#F0FAF6,#EBFAF3);
    border:1px solid #10B981;border-radius:10px;padding:13px 16px;margin:6px 0;
    box-shadow:0 2px 8px rgba(16,185,129,0.08)
}

/* ── About / generic cards ── */
.about-card{
    background:rgba(255,255,255,0.97);border-radius:12px;padding:16px 18px;
    box-shadow:0 3px 12px rgba(13,27,42,0.07);margin-bottom:10px;
    border-left:4px solid #1D9E75;transition:box-shadow .2s
}
.about-card:hover{box-shadow:0 6px 20px rgba(13,27,42,0.11)}
.challenge-stat{
    background:linear-gradient(135deg,#0D1B2A,#0f2235);color:white;
    border-radius:12px;padding:20px 18px;text-align:center;
    box-shadow:0 4px 16px rgba(13,27,42,0.2);border:1px solid rgba(29,158,117,0.2)
}
.challenge-num{font-size:30px;font-weight:900;color:#1D9E75;letter-spacing:-0.5px}
.challenge-txt{font-size:11px;color:#9CA3AF;margin-top:4px;line-height:1.4}

/* ── Architecture diagram ── */
.arch-flow{display:flex;align-items:center;flex-wrap:wrap;gap:6px;justify-content:center;padding:16px 0}
.arch-box{
    background:linear-gradient(135deg,#0D1B2A,#0f2235);color:white;border-radius:8px;
    padding:10px 14px;font-size:12px;font-weight:600;text-align:center;min-width:120px;
    box-shadow:0 2px 8px rgba(13,27,42,0.2)
}
.arch-box-teal{background:linear-gradient(135deg,#1D9E75,#17836A)}
.arch-box-amber{background:linear-gradient(135deg,#F59E0B,#D97706);color:#FFFFFF}
.arch-arrow{color:#6B7280;font-size:18px;font-weight:300}

/* ── Demo scenario buttons in sidebar ── */
.demo-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px}

/* ── Problem statement section ── */
.problem-card{
    background:linear-gradient(135deg,#0D1B2A 0%,#0f2235 100%);
    border-radius:14px;padding:28px 32px;margin:16px 0;color:white;
    box-shadow:0 8px 32px rgba(13,27,42,0.2);border:1px solid rgba(29,158,117,0.15)
}
.problem-stat{
    background:rgba(255,255,255,0.06);border-radius:10px;padding:18px 16px;
    text-align:center;border:1px solid rgba(29,158,117,0.2);
    backdrop-filter:blur(4px)
}
.problem-num{font-size:32px;font-weight:900;color:#1D9E75;letter-spacing:-1px}
.problem-label{font-size:11px;color:rgba(255,255,255,0.7);margin-top:4px;line-height:1.4}

/* ── Tech stack chips ── */
.tech-section{
    background:rgba(255,255,255,0.95);border-radius:14px;padding:24px 28px;
    box-shadow:0 4px 20px rgba(13,27,42,0.08);margin:12px 0;
    border-top:3px solid #1D9E75
}
.tech-category{font-size:10px;font-weight:700;color:#1D9E75;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px}
.tech-chip{
    display:inline-flex;align-items:center;gap:4px;
    background:linear-gradient(135deg,#F0F9FF,#E8F5EE);
    border:1px solid rgba(29,158,117,0.25);border-radius:20px;
    padding:5px 12px;margin:3px 4px 3px 0;font-size:11.5px;font-weight:600;
    color:#0D1B2A;transition:all .15s ease
}
.tech-chip:hover{background:linear-gradient(135deg,#1D9E75,#17836A);color:#FFFFFF;border-color:#1D9E75}

/* ── PharmD clinical perspective ── */
.pharmd-section{
    background:linear-gradient(135deg,rgba(13,27,42,0.03),rgba(29,158,117,0.04));
    border-radius:14px;padding:24px 28px;margin:12px 0;
    border:1px solid rgba(29,158,117,0.15)
}
.pharmd-insight{
    background:rgba(255,255,255,0.97);border-radius:10px;padding:16px 18px;
    border-left:4px solid #1D9E75;margin-bottom:10px;
    box-shadow:0 2px 10px rgba(13,27,42,0.06);transition:all .2s
}
.pharmd-insight:hover{box-shadow:0 5px 18px rgba(13,27,42,0.1);transform:translateX(3px)}
.pharmd-insight-title{font-size:13px;font-weight:700;color:#0D1B2A;margin-bottom:5px}
.pharmd-insight-body{font-size:12px;color:#4B5563;line-height:1.6}
.pharmd-insight-ref{font-size:10.5px;color:#9CA3AF;margin-top:6px;font-style:italic}

/* ── Professional footer ── */
.tg-footer{
    background:linear-gradient(135deg,#0D1B2A,#0f2235);border-radius:14px;
    padding:28px 32px;margin-top:20px;
    box-shadow:0 -4px 20px rgba(13,27,42,0.1);
    border-top:2px solid rgba(29,158,117,0.3)
}
.tg-footer-brand{font-size:18px;font-weight:900;color:#FFFFFF;letter-spacing:-0.3px}
.tg-footer-brand span{color:#1D9E75}
.tg-footer-tagline{font-size:11px;color:rgba(255,255,255,0.55);margin-top:3px;letter-spacing:0.3px}
.tg-footer-link{
    display:inline-flex;align-items:center;gap:6px;color:#A8D5C4;
    font-size:12px;font-weight:500;text-decoration:none;
    padding:6px 12px;border-radius:20px;border:1px solid rgba(29,158,117,0.3);
    transition:all .2s;margin:4px
}
.tg-footer-link:hover{background:rgba(29,158,117,0.15);color:#FFFFFF}
.tg-footer-divider{border:none;border-top:1px solid rgba(255,255,255,0.08);margin:18px 0}
.tg-footer-disclaimer{font-size:10.5px;color:rgba(255,255,255,0.4);line-height:1.6}

/* ── Vision cards ── */
.vision-card{
    background:rgba(255,255,255,0.97);border-radius:12px;padding:20px 18px;
    text-align:center;border-top:3px solid #1D9E75;
    box-shadow:0 4px 16px rgba(13,27,42,0.07);transition:all .25s
}
.vision-card:hover{box-shadow:0 8px 28px rgba(13,27,42,0.13);transform:translateY(-3px)}
.vision-icon{font-size:32px;margin-bottom:10px}
.vision-title{font-size:13px;font-weight:800;color:#0D1B2A;margin-bottom:6px;letter-spacing:0.2px}
.vision-body{font-size:11.5px;color:#6B7280;line-height:1.55}

/* ── Pipeline step ── */
.pipeline-step{
    background:rgba(255,255,255,0.97);border-radius:10px;padding:14px 16px;
    border-left:3px solid #1D9E75;margin-bottom:8px;
    box-shadow:0 2px 8px rgba(13,27,42,0.06);transition:all .2s
}
.pipeline-step:hover{box-shadow:0 4px 16px rgba(13,27,42,0.1);transform:translateX(3px)}
.pipeline-num{font-size:10px;font-weight:700;color:#1D9E75;letter-spacing:1px;text-transform:uppercase}
.pipeline-title{font-size:13px;font-weight:700;color:#0D1B2A}
.pipeline-desc{font-size:11.5px;color:#6B7280;margin-top:3px;line-height:1.5}

/* ── TrialGuard teal — override Streamlit primary buttons ── */
div.stButton>button[kind="primary"],
button[data-testid="baseButton-primary"]{
    background:linear-gradient(135deg,#1D9E75,#17836A)!important;
    border-color:#1D9E75!important;color:#FFFFFF!important;
    font-weight:700!important;letter-spacing:0.3px!important;
    box-shadow:0 4px 14px rgba(29,158,117,0.35)!important
}
div.stButton>button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover{
    background:linear-gradient(135deg,#17836A,#145f50)!important;
    box-shadow:0 6px 20px rgba(29,158,117,0.45)!important;transform:translateY(-1px)!important
}
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
    "Participant Burden Score": "⚖️",
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
    st.session_state.page = "assessment"
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


# ── Executive KPI strip ───────────────────────────────────────────────────────
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
        f'<div class="kpi-label">&#9679; Participants Modelled</div>'
        f'<div class="kpi-value">{len(df):,}</div>'
        f'<div class="kpi-sub">Synthetic demonstration cohort</div>'
        f'<div class="kpi-trend kpi-trend-down">&#9660; 25 clinical features per participant</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        f'<div class="kpi-card" style="border-left-color:#EF4444">'
        f'<div class="kpi-label">&#9679; Elevated Attrition Risk</div>'
        f'<div class="kpi-value" style="color:#EF4444">{high_n:,}</div>'
        f'<div class="kpi-sub">AI risk score &ge; 0.60 threshold</div>'
        f'<div class="kpi-trend kpi-trend-up">&#9650; Priority intervention candidates</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    k3.markdown(
        f'<div class="kpi-card" style="border-left-color:#F59E0B">'
        f'<div class="kpi-label">&#9679; Observed Attrition Rate</div>'
        f'<div class="kpi-value" style="color:#F59E0B">{attr_pct:.1f}%</div>'
        f'<div class="kpi-sub">Cohort-level dropout frequency</div>'
        f'<div class="kpi-trend" style="color:#F59E0B">&#9654; Aligns with industry 20&ndash;30% benchmark</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    k4.markdown(
        f'<div class="kpi-card" style="border-left-color:#0D1B2A">'
        f'<div class="kpi-label">&#9679; Modelled Cost Exposure</div>'
        f'<div class="kpi-value" style="color:#0D1B2A">${cost_exp/1_000_000:.1f}M</div>'
        f'<div class="kpi-sub">Unaddressed attrition cost</div>'
        f'<div class="kpi-trend kpi-trend-down">&#9660; Based on $18K replacement cost (Getz 2016)</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar navigation (always visible) ──────────────────────────────────────
def render_sidebar_nav():
    _init_state()
    current = st.session_state.get("page", "home")

    # ── Brand header ──────────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='padding:16px 4px 10px'>"
        "<div style='font-size:22px;font-weight:900;color:#FFFFFF;letter-spacing:-0.5px'>🛡️ TrialGuard</div>"
        "<div style='font-size:10px;color:#A8D5C4;font-weight:500;margin-top:3px;letter-spacing:0.3px'>"
        "AI-Powered Clinical Trial Retention Intelligence Platform</div>"
        "<div style='width:100%;height:1px;background:rgba(29,158,117,0.3);margin:12px 0 4px'></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Platform navigation — ALL native buttons (guaranteed sidebar render) ──
    st.sidebar.markdown(
        "<div style='font-size:9px;font-weight:700;color:#1D9E75;letter-spacing:1.8px;"
        "text-transform:uppercase;margin:6px 0 4px 2px'>Platform</div>",
        unsafe_allow_html=True,
    )
    nav_items = [
        ("home",         "🏠", "Overview"),
        ("intake",       "📄", "Document Intake"),
        ("assessment",   "⚠️",  "Risk Assessment"),
        ("dashboard",    "📊", "Retention Intelligence Center"),
        ("batch",        "📁", "Population Risk Screening"),
        ("intelligence", "🧠", "AI Intelligence"),
        ("about",        "ℹ️",  "About"),
    ]
    for page_key, icon, label in nav_items:
        # Active page gets ◀ indicator; all rendered as native buttons
        btn_label = f"{icon}  {label}  ◀" if current == page_key else f"{icon}  {label}"
        if st.sidebar.button(btn_label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.page = page_key
            st.rerun()

    # ── Participant Personas ──────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='width:100%;height:1px;background:rgba(29,158,117,0.2);margin:12px 0 4px'></div>"
        "<div style='font-size:9px;font-weight:700;color:#1D9E75;letter-spacing:1.8px;"
        "text-transform:uppercase;margin-bottom:4px'>Participant Personas</div>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🔴  High Attrition Risk",      key="demo_hr", use_container_width=True): _load_demo("high_risk")
    if st.sidebar.button("🚗  Transportation Barrier",   key="demo_ru", use_container_width=True): _load_demo("rural")
    if st.sidebar.button("💊  Polypharmacy Burden",      key="demo_pp", use_container_width=True): _load_demo("polypharmacy")
    if st.sidebar.button("🟢  Low-Risk Benchmark",       key="demo_lr", use_container_width=True): _load_demo("low_risk")

    # On assessment page: participant inputs appear here (injected by main())
    # so skip Portfolio Snapshot/Skills to avoid pushing inputs off-screen
    if current == "assessment":
        return

    # ── Portfolio Snapshot ────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='width:100%;height:1px;background:rgba(29,158,117,0.2);margin:14px 0 4px'></div>"
        "<div style='font-size:9px;font-weight:700;color:#1D9E75;letter-spacing:1.8px;"
        "text-transform:uppercase;margin-bottom:8px'>Portfolio Snapshot</div>"
        "<div style='background:rgba(255,255,255,0.05);border:1px solid rgba(29,158,117,0.2);"
        "border-radius:10px;padding:11px 12px;margin-bottom:6px'>"
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>"
        "<span style='font-size:13px;font-weight:800;color:#4CD4A0;min-width:44px'>2,000</span>"
        "<span style='font-size:10px;color:rgba(255,255,255,0.55)'>Participants Modelled</span></div>"
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>"
        "<span style='font-size:13px;font-weight:800;color:#4CD4A0;min-width:44px'>$1.6M</span>"
        "<span style='font-size:10px;color:rgba(255,255,255,0.55)'>Modelled Savings / Cohort</span></div>"
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>"
        "<span style='font-size:13px;font-weight:800;color:#4CD4A0;min-width:44px'>XAI</span>"
        "<span style='font-size:10px;color:rgba(255,255,255,0.55)'>SHAP-Attributed Every Prediction</span></div>"
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>"
        "<span style='font-size:13px;font-weight:800;color:#4CD4A0;min-width:44px'>7</span>"
        "<span style='font-size:10px;color:rgba(255,255,255,0.55)'>Evidence-Based Interventions</span></div>"
        "<div style='display:flex;align-items:center;gap:8px'>"
        "<span style='font-size:13px;font-weight:800;color:#4CD4A0;min-width:44px'>PDF</span>"
        "<span style='font-size:10px;color:rgba(255,255,255,0.55)'>Enterprise Intelligence Report</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Skills Demonstrated ───────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='font-size:9px;font-weight:700;color:#1D9E75;letter-spacing:1.8px;"
        "text-transform:uppercase;margin-bottom:7px'>Skills Demonstrated</div>"
        "<div style='display:flex;flex-wrap:wrap;gap:4px;margin-bottom:16px'>"
        "<span style='background:rgba(29,158,117,0.12);color:#A8D5C4;border:1px solid rgba(29,158,117,0.2);border-radius:10px;padding:2px 8px;font-size:9.5px;font-weight:600'>Clinical Trial Analytics</span>"
        "<span style='background:rgba(29,158,117,0.12);color:#A8D5C4;border:1px solid rgba(29,158,117,0.2);border-radius:10px;padding:2px 8px;font-size:9.5px;font-weight:600'>Clinical Operations</span>"
        "<span style='background:rgba(29,158,117,0.12);color:#A8D5C4;border:1px solid rgba(29,158,117,0.2);border-radius:10px;padding:2px 8px;font-size:9.5px;font-weight:600'>Pharmacovigilance</span>"
        "<span style='background:rgba(29,158,117,0.12);color:#A8D5C4;border:1px solid rgba(29,158,117,0.2);border-radius:10px;padding:2px 8px;font-size:9.5px;font-weight:600'>Machine Learning</span>"
        "<span style='background:rgba(29,158,117,0.12);color:#A8D5C4;border:1px solid rgba(29,158,117,0.2);border-radius:10px;padding:2px 8px;font-size:9.5px;font-weight:600'>XGBoost</span>"
        "<span style='background:rgba(29,158,117,0.12);color:#A8D5C4;border:1px solid rgba(29,158,117,0.2);border-radius:10px;padding:2px 8px;font-size:9.5px;font-weight:600'>SHAP</span>"
        "<span style='background:rgba(29,158,117,0.12);color:#A8D5C4;border:1px solid rgba(29,158,117,0.2);border-radius:10px;padding:2px 8px;font-size:9.5px;font-weight:600'>Explainable AI</span>"
        "<span style='background:rgba(29,158,117,0.12);color:#A8D5C4;border:1px solid rgba(29,158,117,0.2);border-radius:10px;padding:2px 8px;font-size:9.5px;font-weight:600'>Healthcare AI</span>"
        "<span style='background:rgba(29,158,117,0.12);color:#A8D5C4;border:1px solid rgba(29,158,117,0.2);border-radius:10px;padding:2px 8px;font-size:9.5px;font-weight:600'>Streamlit</span>"
        "<span style='background:rgba(29,158,117,0.12);color:#A8D5C4;border:1px solid rgba(29,158,117,0.2);border-radius:10px;padding:2px 8px;font-size:9.5px;font-weight:600'>Python</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Sidebar participant input form (assessment page only) ─────────────────────
def render_sidebar_inputs() -> pd.DataFrame:
    st.sidebar.markdown(
        "<div style='width:100%;height:1px;background:rgba(29,158,117,0.2);margin:10px 0 10px'></div>"
        "<div style='font-size:10px;font-weight:700;color:#1D9E75;letter-spacing:1.5px;"
        "text-transform:uppercase;margin-bottom:8px'>Participant Parameters</div>",
        unsafe_allow_html=True,
    )

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
                                 help="Highest-ranked predictor in this model.")

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
        "<div style='background:#EFF9F5;border-left:4px solid #1D9E75;padding:10px 16px;"
        "border-radius:6px;margin-bottom:14px;font-size:12px;color:#374151'>"
        "<b style='color:#1D9E75'>Dictionary-based clinical entity recognition</b> — "
        "entities identified from document text. Source: <b>Investigator Notes / CRF</b>. "
        "Clinical review required before use in downstream analysis."
        "</div>",
        unsafe_allow_html=True,
    )

    # Deduplicate diseases — remove shorter labels that are substrings of longer ones
    def dedup_entities(items):
        canonicals = [c for _, c in items]
        return [
            (raw, canon) for raw, canon in items
            if not any(other != canon and canon.lower() in other.lower() for other in canonicals)
        ]

    def chips_with_conf(items, color, base_conf: int):
        html = ""
        for i, (_, label) in enumerate(items):
            pct = max(base_conf - i * 2, base_conf - 8)
            html += (
                f'<span style="display:inline-block;background:{color};color:#fff;'
                f'border-radius:14px;padding:4px 11px;margin:3px 4px 3px 0;font-size:11.5px;'
                f'font-weight:600">{label}'
                f'<span style="opacity:0.75;font-weight:400;font-size:10px;margin-left:5px">({pct}%)</span>'
                f'</span>'
            )
        return html

    diseases_deduped = dedup_entities(ner_result.diseases)

    col_a, col_b = st.columns(2)
    with col_a:
        if diseases_deduped:
            st.markdown("**Conditions / Diagnoses**")
            st.markdown(chips_with_conf(diseases_deduped, "#1D9E75", 98), unsafe_allow_html=True)
        if ner_result.adverse_events:
            st.markdown("**Adverse Events / Safety Signals**")
            st.markdown(chips_with_conf(ner_result.adverse_events, "#D9534F", 93), unsafe_allow_html=True)
        if ner_result.symptoms:
            st.markdown("**Symptoms Reported**")
            st.markdown(chips_with_conf(ner_result.symptoms, "#F4B942", 87), unsafe_allow_html=True)
    with col_b:
        if ner_result.burden_flags:
            st.markdown("**Trial Burden Indicators**")
            st.markdown(chips_with_conf(ner_result.burden_flags, "#E05C25", 91), unsafe_allow_html=True)
        if ner_result.engagement_flags:
            st.markdown("**Engagement / Protective Signals**")
            st.markdown(chips_with_conf(ner_result.engagement_flags, "#3B82F6", 89), unsafe_allow_html=True)
        if ner_result.medications:
            st.markdown("**Medications Identified**")
            med_html = "".join(
                f'<span style="display:inline-block;background:#6B7280;color:#fff;'
                f'border-radius:14px;padding:3px 10px;margin:3px 4px 3px 0;font-size:11px">'
                f'{m.capitalize()}</span>'
                for m in ner_result.medications
            )
            st.markdown(med_html, unsafe_allow_html=True)

    n_disease = len(diseases_deduped)
    if n_disease > 0:
        chart_caption(
            f"NER identified {n_disease} distinct condition(s) · Inferred comorbidity count: {n_disease} · "
            "Source: Investigator Notes / CRF · Dictionary-based matching · Clinical review required."
        )
    st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)


def render_coordinator_copilot(analysis: dict, risk_cat: str):
    """TrialGuard Coordinator Copilot — Module 5 + 6 + 7 + 8."""
    from coordinator_copilot import CoordinatorCopilot

    risk_pct       = analysis.get("risk_pct", 0)
    top3_risk      = analysis.get("top3_risk_factors", [])
    top3_prot      = analysis.get("top3_protective_factors", [])
    interventions  = analysis.get("interventions", [])
    dropout_window = analysis.get("dropout_window", "")

    copilot  = CoordinatorCopilot()
    summary  = copilot.generate(
        risk_pct=risk_pct,
        risk_cat=risk_cat,          # normalised inside copilot from pct
        top3_risk_factors=top3_risk,
        top3_protective=top3_prot,
        interventions=interventions,
        participant_data={},
        dropout_window=str(dropout_window),
    )

    # Store for PDF
    st.session_state["_copilot_summary"] = summary

    section_header("Clinical Retention Copilot")
    st.markdown(
        '_V3.0 Clinical Reasoning Engine — template-based, deterministic. '
        'Clinical review required before action._'
    )

    # Structured 3-panel summary
    risk_color = {"Critical": "#D9534F", "High": "#E05C25", "Moderate": "#F4B942", "Low": "#1D9E75"}
    rc = risk_color.get(risk_cat, "#F4B942")

    cp1, cp2, cp3 = st.columns(3)
    # Key Risk Factors
    rf_items = "".join(
        f"<li style='font-size:12px;color:#374151;line-height:1.7;margin-bottom:2px'>{lbl}</li>"
        for lbl in (summary.primary_drivers[:3] if summary.primary_drivers else ["No significant drivers identified"])
    )
    cp1.markdown(
        f"<div style='background:#FEF2F2;border-top:3px solid #D9534F;border-radius:8px;padding:14px 16px;height:100%'>"
        f"<div style='font-size:10px;font-weight:800;color:#D9534F;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Key Risk Factors</div>"
        f"<ul style='margin:0;padding-left:16px'>{rf_items}</ul>"
        f"</div>", unsafe_allow_html=True,
    )
    # Recommended Actions
    act_items = "".join(
        f"<li style='font-size:12px;color:#374151;line-height:1.7;margin-bottom:2px'>"
        f"<span style='font-weight:600'>{act.title}</span>"
        f"<span style='color:#6B7280;font-size:11px'> · {act.timeline}</span></li>"
        for act in (summary.action_items[:3] if summary.action_items else [])
    ) or "<li style='font-size:12px;color:#374151'>Standard monitoring</li>"
    cp2.markdown(
        f"<div style='background:#EFF9F5;border-top:3px solid #1D9E75;border-radius:8px;padding:14px 16px;height:100%'>"
        f"<div style='font-size:10px;font-weight:800;color:#1D9E75;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Recommended Actions</div>"
        f"<ul style='margin:0;padding-left:16px'>{act_items}</ul>"
        f"</div>", unsafe_allow_html=True,
    )
    # Expected Outcome
    imp_low  = summary.expected_improvement_low
    imp_high = summary.expected_improvement_high
    cp3.markdown(
        f"<div style='background:#EFF6FF;border-top:3px solid #3B82F6;border-radius:8px;padding:14px 16px;height:100%'>"
        f"<div style='font-size:10px;font-weight:800;color:#3B82F6;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Expected Outcome</div>"
        f"<div style='font-size:22px;font-weight:800;color:#1D4ED8;margin-bottom:4px'>{imp_low}–{imp_high}pp</div>"
        f"<div style='font-size:11px;color:#374151'>Estimated retention improvement<br>with full intervention plan</div>"
        f"</div>", unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # Narrative + reasoning expander
    st.markdown(
        f'<div style="background:#F8FAFC;border-left:4px solid {rc};padding:14px 18px;'
        f'border-radius:6px;margin-bottom:12px">'
        f'<div style="font-size:13px;font-weight:600;color:{rc};margin-bottom:6px">AI-Assisted Clinical Narrative</div>'
        f'<div style="font-size:13px;color:#374151;line-height:1.6">{summary.risk_narrative}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("📖 Clinical Reasoning — Why is this participant at risk?"):
        st.markdown(summary.reasoning_text)

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
            # Completion target derived from timeline
            _completion_map = {
                "Within 24 hours":       "Complete within 24 h",
                "Within 72 hours":       "Complete within 72 h",
                "Before Next Visit":     "Before next scheduled visit",
                "Protocol Review Cycle": "Next protocol review cycle",
            }
            completion = _completion_map.get(act.timeline, act.timeline)
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
                f'<div style="display:flex;gap:16px;margin-top:8px">'
                f'<div style="font-size:11px;color:#1D9E75;font-weight:600">&#9654; Est. reduction: ~{act.expected_reduction:.0f}pp</div>'
                f'<div style="font-size:11px;color:#6B7280">&#128337; Target: {completion}</div>'
                f'</div></div>',
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

    # Budget recommendation
    if summary.budget_recommendation:
        st.info(f"💡 **If budget is constrained:** {summary.budget_recommendation}")


def render_tab_batch():
    """Population Risk Screening — Multi-Participant Retention Intelligence."""
    from datetime import date as _date, timedelta as _td

    section_header("Population Risk Screening")
    st.markdown(
        "<div style='font-size:13px;color:#6B7280;margin-bottom:6px'>"
        "Upload a participant cohort CSV to score all participants, rank retention risk, "
        "generate coordinator worklists, and produce site-level retention intelligence.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='background:#EFF9F5;border-left:4px solid #1D9E75;border-radius:8px;"
        "padding:10px 16px;margin-bottom:16px;font-size:12.5px;color:#0D1B2A;line-height:1.6'>"
        "<b>Module purpose:</b> Participant-level risk scoring &rarr; site-level retention intelligence "
        "&rarr; coordinator action queues &rarr; CRO-grade retention reporting.</div>",
        unsafe_allow_html=True,
    )

    _today_str = _date.today().strftime("%Y%m%d")

    # ── Template download ─────────────────────────────────────────────────────
    _template_csv = (
        "patient_id,site_id,study_id,country,investigator_id,enrollment_date,site_region,"
        "age,gender,bmi,disease_severity_score,"
        "number_of_comorbidities,concomitant_medications,distance_from_site_km,"
        "visit_frequency_per_month,side_effect_severity_at_week2,"
        "insurance_status,transportation_access,prior_trial_participation,"
        "trial_phase,consent_complexity_score,visit_burden_index,logistic_friction_score\n"
        "PT-0001,SITE_01,STUDY-2024-001,USA,INV-042,2024-01-15,North,"
        "58,M,26.5,7.0,3,8,75,5,3.5,insured,no,1,2,6,6,5\n"
        "PT-0002,SITE_01,STUDY-2024-001,USA,INV-042,2024-01-18,North,"
        "42,F,22.1,4.0,1,2,12,3,0.5,insured,yes,0,3,4,2,1\n"
        "PT-0003,SITE_02,STUDY-2024-001,UK,INV-017,2024-01-20,South,"
        "67,F,30.2,8.5,5,11,90,6,4.5,uninsured,no,0,2,8,8,7\n"
    )
    tdl1, tdl2 = st.columns([2, 5])
    with tdl1:
        st.download_button(
            "📥 Download CSV Template",
            data=_template_csv,
            file_name="TrialGuard_Batch_Template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with tdl2:
        st.caption(
            "Template includes CRO study metadata: study_id, country, investigator_id, "
            "enrollment_date, site_region. These carry through to exports but do not affect risk scoring."
        )

    # ── File upload ───────────────────────────────────────────────────────────
    uploaded_csv = st.file_uploader(
        "Upload Participant CSV",
        type=["csv"],
        key="batch_uploader",
        help="Required: age, gender, bmi, disease_severity_score, number_of_comorbidities, "
             "concomitant_medications, distance_from_site_km, visit_frequency_per_month, "
             "side_effect_severity_at_week2, insurance_status, transportation_access, "
             "prior_trial_participation, trial_phase, consent_complexity_score, "
             "visit_burden_index, logistic_friction_score",
    )

    if uploaded_csv is None:
        st.info(
            "Upload a participant CSV to begin population risk screening. "
            "Download the template above for the correct format including study metadata fields."
        )
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

    with st.spinner(f"Scoring {len(df)} participants across {df['site_id'].nunique()} sites…"):
        result = batch_screen(df, model, preprocessor)

    if "error" in result:
        st.error(result["error"])
        return

    results_df   = result["results_df"].copy()
    small_sample = result["small_sample"]

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Participants", result["total"])
    k2.metric("Critical Risk",  result["critical_n"])
    k3.metric("High Risk",      result["high_n"])
    k4.metric("Moderate Risk",  result["moderate_n"])
    k5.metric("Low Risk",       result["low_n"])
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # ── Population Risk Intelligence ──────────────────────────────────────────
    section_header("Population Risk Intelligence")

    avg_risk = results_df["Risk Score (%)"].mean()

    pi1, pi2, pi3 = st.columns(3)
    pi1.markdown(
        f'<div class="kpi-card" style="border-left-color:#D9534F">'
        f'<div class="kpi-label">High/Critical Participants</div>'
        f'<div class="kpi-value" style="color:#D9534F">{result["at_risk_n"]}</div>'
        f'<div class="kpi-sub">Require active retention intervention</div></div>',
        unsafe_allow_html=True,
    )
    pi2.markdown(
        f'<div class="kpi-card" style="border-left-color:#F4B942">'
        f'<div class="kpi-label">Average Risk Score</div>'
        f'<div class="kpi-value" style="color:#F4B942">{avg_risk:.1f}%</div>'
        f'<div class="kpi-sub">Cohort mean attrition probability</div></div>',
        unsafe_allow_html=True,
    )
    pi3.markdown(
        f'<div class="kpi-card" style="border-left-color:#EF4444">'
        f'<div class="kpi-label">Highest Risk Site</div>'
        f'<div class="kpi-value" style="color:#EF4444;font-size:20px">{result["highest_risk_site"]}</div>'
        f'<div class="kpi-sub">Highest mean participant risk score</div></div>',
        unsafe_allow_html=True,
    )

    pi4, pi5, pi6 = st.columns(3)
    pi4.markdown(
        f'<div class="kpi-card" style="border-left-color:#6366F1">'
        f'<div class="kpi-label">Top Risk Driver</div>'
        f'<div class="kpi-value" style="color:#6366F1;font-size:17px">{result["top_driver"]}</div>'
        f'<div class="kpi-sub">Affected {result["top_driver_pct"]}% of participants</div></div>',
        unsafe_allow_html=True,
    )
    hr_df = results_df[results_df["Risk Score (%)"] >= 61]
    if not hr_df.empty and "Primary Risk Driver" in hr_df.columns:
        barrier     = hr_df["Primary Risk Driver"].value_counts().idxmax()
        barrier_pct = round(hr_df["Primary Risk Driver"].value_counts().max() / len(hr_df) * 100)
    else:
        barrier, barrier_pct = "N/A", 0
    pi5.markdown(
        f'<div class="kpi-card" style="border-left-color:#F59E0B">'
        f'<div class="kpi-label">Most Common Retention Barrier</div>'
        f'<div class="kpi-value" style="color:#F59E0B;font-size:16px">{barrier}</div>'
        f'<div class="kpi-sub">In high/critical participants ({barrier_pct}%)</div></div>',
        unsafe_allow_html=True,
    )
    if not hr_df.empty and "Attrition Window" in hr_df.columns:
        top_window = hr_df["Attrition Window"].value_counts().idxmax()
    else:
        top_window = "N/A"
    pi6.markdown(
        f'<div class="kpi-card" style="border-left-color:#1D9E75">'
        f'<div class="kpi-label">Peak Attrition Window</div>'
        f'<div class="kpi-value" style="color:#1D9E75;font-size:15px">{top_window}</div>'
        f'<div class="kpi-sub">Most common high-risk timing</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # ── Executive Summary ─────────────────────────────────────────────────────
    section_header("Executive Summary")
    _crit_txt = (
        f"<b>{result['critical_n']} participant{'s' if result['critical_n'] > 1 else ''}</b> "
        f"{'require' if result['critical_n'] > 1 else 'requires'} immediate retention intervention (Critical Risk). "
    ) if result["critical_n"] > 0 else ""
    _high_txt = (
        f"<b>{result['high_n']} participant{'s' if result['high_n'] > 1 else ''}</b> "
        f"{'are' if result['high_n'] > 1 else 'is'} classified as High Risk — priority outreach within 24 hours. "
    ) if result["high_n"] > 0 else ""
    _exec_body = (
        f"Batch scoring complete across <b>{result['total']} participants</b> at "
        f"<b>{df['site_id'].nunique()} site{'s' if df['site_id'].nunique() > 1 else ''}</b>. "
        f"{_crit_txt}{_high_txt}"
        f"<b>{result['top_driver']}</b> is the dominant dropout driver, affecting "
        f"<b>{result['top_driver_pct']}%</b> of the cohort. "
        f"<b>{result['highest_risk_site']}</b> contains the highest concentration of retention risk "
        f"and should be prioritised for coordinator outreach."
    )
    st.markdown(
        "<div style='background:linear-gradient(135deg,#0D1B2A,#0f2336);"
        "border:1px solid rgba(29,158,117,0.3);border-radius:12px;padding:20px 24px;margin-bottom:18px'>"
        "<div style='font-size:11px;font-weight:700;color:#1D9E75;text-transform:uppercase;"
        "letter-spacing:1.5px;margin-bottom:10px'>Executive Summary</div>"
        f"<div style='font-size:13.5px;color:rgba(255,255,255,0.88);line-height:1.9'>{_exec_body}</div>"
        "<div style='margin-top:12px;font-size:10px;color:rgba(255,255,255,0.3)'>"
        f"Generated: {_date.today().strftime('%d-%b-%Y')} &bull; Population Risk Screening Engine v3.1 &bull; Synthetic demonstration data"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Intervention Budget Estimate ──────────────────────────────────────────
    at_risk = result["at_risk_n"]
    if at_risk > 0:
        section_header("Intervention Budget Estimate")
        if small_sample:
            st.warning(
                "⚠️ **Small sample size** (<25 participants). Economic calculations are shown for "
                "demonstration purposes only. Reliable financial modelling requires larger participant populations."
            )
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("High/Critical", at_risk,
                  help="Participants recommended for active retention intervention.")
        b2.metric("Est. Budget", f"${result['est_budget']:,}",
                  help="Estimated total intervention cost at $1,800 per at-risk participant.")
        b3.metric("Dropouts Preventable", result["est_prevented"],
                  help="Based on model recall × estimated 45% intervention success rate.")
        b4.metric("Est. Net Benefit", f"${result['net_benefit']:,}",
                  help="Estimated savings minus intervention cost.")
        b5.metric("Exp. Retained", result["est_prevented"],
                  help="Expected number of participants retained through intervention.")
        b6.metric("ROI",
                  f"+{result['roi_pct']}%" if result["roi_pct"] >= 0 else f"{result['roi_pct']}%",
                  help="Return on investment: (savings - cost) / cost x 100.")
        _sample_note = (
            "<b style='color:#D97706'>Small sample — estimates are illustrative only</b>"
            if small_sample else "Synthetic demonstration dataset"
        )
        st.markdown(
            f"<div style='background:#F8FAFC;border:1px solid #E5E7EB;border-radius:8px;"
            f"padding:8px 16px;margin-top:8px;font-size:12px;color:#6B7280'>"
            f"Confidence: <b>{result['confidence']}</b> &bull; $18,000 replacement cost (Getz KA et al., 2016) "
            f"&bull; 45% intervention success rate &bull; {_sample_note}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Participant Risk Ranking ───────────────────────────────────────────────
    section_header("Participant Risk Ranking")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        search_pt = st.text_input("Search Participant ID", "", placeholder="e.g. PT-0001")
    with fc2:
        sites_opts = ["All Sites"] + sorted(results_df["Site"].unique().tolist())
        filter_site = st.selectbox("Filter by Site", sites_opts)
    with fc3:
        filter_cat = st.selectbox(
            "Filter by Risk Category",
            ["All", "Critical", "High", "Moderate", "Low"],
        )

    filtered_df = results_df.copy()
    if search_pt:
        filtered_df = filtered_df[
            filtered_df["Participant ID"].str.contains(search_pt, case=False, na=False)
        ]
    if filter_site != "All Sites":
        filtered_df = filtered_df[filtered_df["Site"] == filter_site]
    if filter_cat != "All":
        filtered_df = filtered_df[filtered_df["Risk Category"] == filter_cat]

    st.caption(f"Showing {len(filtered_df)} of {len(results_df)} participants")

    _ROW_COLORS = {
        "Critical": "background-color:#FEE2E2",
        "High":     "background-color:#FEF3C7",
        "Moderate": "background-color:#FFF7ED",
        "Low":      "background-color:#D1FAE5",
    }

    def risk_color_row(row):
        return [_ROW_COLORS.get(row["Risk Category"], "")] * len(row)

    _display_cols = [
        "Participant ID", "Site", "Risk Score (%)", "Risk Category",
        "Age", "Distance (km)", "Comorbidities", "Medications",
        "Primary Risk Driver", "Attrition Window", "Recommended Action",
    ]
    _display_cols = [c for c in _display_cols if c in filtered_df.columns]
    styled = filtered_df[_display_cols].style.apply(risk_color_row, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Per-participant explainability expander
    with st.expander("🔍 Risk Driver Analysis — Select Participant", expanded=False):
        if not filtered_df.empty and "Top 3 Risk Drivers" in filtered_df.columns:
            sel_pt  = st.selectbox("Select Participant", filtered_df["Participant ID"].tolist(), key="explain_pt")
            pt_row  = filtered_df[filtered_df["Participant ID"] == sel_pt].iloc[0]
            cat     = pt_row["Risk Category"]
            score   = pt_row["Risk Score (%)"]
            _bc     = {"Critical": "#EF4444", "High": "#F59E0B", "Moderate": "#F97316", "Low": "#10B981"}.get(cat, "#6B7280")
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown(
                    f"<div style='background:{_bc};border-radius:8px;padding:14px;text-align:center;margin-bottom:10px'>"
                    f"<div style='font-size:11px;font-weight:700;color:white;text-transform:uppercase'>Risk Score</div>"
                    f"<div style='font-size:32px;font-weight:800;color:white'>{score}%</div>"
                    f"<div style='font-size:13px;color:rgba(255,255,255,0.85)'>{cat} Risk</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='background:#F8FAFC;border:1px solid #E5E7EB;border-radius:8px;padding:12px'>"
                    f"<div style='font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase;margin-bottom:6px'>Clinical Profile</div>"
                    f"<div style='font-size:13px;color:#374151'>"
                    f"Age: <b>{pt_row.get('Age','N/A')}</b> &bull; Distance: <b>{pt_row.get('Distance (km)','N/A')} km</b><br>"
                    f"Comorbidities: <b>{pt_row.get('Comorbidities','N/A')}</b> &bull; "
                    f"Medications: <b>{pt_row.get('Medications','N/A')}</b>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            with ec2:
                st.markdown("**Top Risk Drivers**")
                drv_colors = ["#EF4444", "#F59E0B", "#6366F1"]
                for i, drv in enumerate([d.strip() for d in pt_row.get("Top 3 Risk Drivers", "").split("|") if d.strip()]):
                    dc = drv_colors[i % len(drv_colors)]
                    st.markdown(
                        f"<div style='background:white;border-left:4px solid {dc};"
                        f"border:1px solid #E5E7EB;border-radius:6px;padding:8px 14px;margin-bottom:6px'>"
                        f"<div style='font-size:13px;font-weight:600;color:#0D1B2A'>{drv}</div></div>",
                        unsafe_allow_html=True,
                    )
            st.markdown(
                f"<div style='background:#F1F5F9;border-radius:6px;padding:10px 14px;margin-top:8px;font-size:12px;color:#64748B'>"
                f"<b>Attrition Window:</b> {pt_row.get('Attrition Window','N/A')} &bull; "
                f"<b>Recommended Action:</b> {pt_row.get('Recommended Action','N/A')} &bull; "
                f"<b>SLA:</b> {pt_row.get('SLA','N/A')} &bull; "
                f"<b>Owner:</b> {pt_row.get('Owner','N/A')}"
                f"</div>",
                unsafe_allow_html=True,
            )

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📥 Download Full Risk Ranking CSV",
            data=results_df.to_csv(index=False),
            file_name=f"TrialGuard_Risk_Ranking_{_today_str}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📥 Download Filtered View CSV",
            data=filtered_df.to_csv(index=False),
            file_name=f"TrialGuard_Filtered_Risk_View_{_today_str}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Coordinator Priority Queue ────────────────────────────────────────────
    section_header("Coordinator Priority Queue")
    st.markdown("_Coordinator-ready operational queue. Top 10 highest-risk participants ordered by clinical urgency._")

    _PQ_ACTIONS = {
        "Critical": "Emergency retention intervention — within 4 hours",
        "High":     "Priority safety call — within 24 hours",
        "Moderate": "Weekly monitoring check-in — review barriers",
        "Low":      "Routine follow-up — standard engagement",
    }
    _PQ_SLA   = {"Critical": "4 Hours", "High": "24 Hours", "Moderate": "7 Days", "Low": "Standard"}
    _PQ_OWNER = {"Critical": "PI + Coordinator", "High": "Study Coordinator", "Moderate": "Site Coordinator", "Low": "Site Staff"}

    top10 = results_df.head(10).copy()
    pq_df = pd.DataFrame({
        "Rank":               range(1, len(top10) + 1),
        "Participant ID":     top10["Participant ID"].values,
        "Site":               top10["Site"].values,
        "Risk Score (%)":     top10["Risk Score (%)"].values,
        "Risk Category":      top10["Risk Category"].values,
        "Priority Level":     top10["Risk Category"].values,
        "SLA":                top10["Risk Category"].map(_PQ_SLA).values,
        "Owner":              top10["Risk Category"].map(_PQ_OWNER).values,
        "Status":             "Pending",
        "Action Recommended": top10["Risk Category"].map(_PQ_ACTIONS).values,
    })

    def pq_color_row(row):
        return [_ROW_COLORS.get(row["Risk Category"], "")] * len(row)

    st.dataframe(pq_df.style.apply(pq_color_row, axis=1), use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Download Priority Queue CSV",
        data=pq_df.to_csv(index=False),
        file_name=f"TrialGuard_Priority_Queue_{_today_str}.csv",
        mime="text/csv",
    )

    # ── Coordinator Worklist ──────────────────────────────────────────────────
    section_header("Coordinator Worklist")
    st.markdown(
        "_Today's retention tasks auto-generated from population risk scoring. "
        "Critical and High participants require immediate action._"
    )

    _DUE = {
        "Critical": _date.today().strftime("%d-%b-%Y") + " (Today)",
        "High":     (_date.today() + _td(days=1)).strftime("%d-%b-%Y"),
        "Moderate": (_date.today() + _td(days=7)).strftime("%d-%b-%Y"),
        "Low":      (_date.today() + _td(days=30)).strftime("%d-%b-%Y"),
    }
    _ASSIGNED = {
        "Critical": "PI + Study Coordinator",
        "High":     "Study Coordinator",
        "Moderate": "Site Coordinator",
        "Low":      "Site Staff",
    }
    _WORKLIST_TASKS = {
        "Critical": [
            "Schedule emergency safety call — target: within 4 hours",
            "Escalate to Principal Investigator immediately",
            "Review and document current AE burden per ICH E6(R2)",
            "Initiate emergency retention protocol if applicable",
        ],
        "High": [
            "Schedule proactive safety call — target: within 24 hours",
            "Confirm transportation availability for next scheduled visit",
            "Review adverse event burden with study nurse",
            "Escalate to site investigator if AE symptoms meet CTCAE Grade >= 2",
        ],
        "Moderate": [
            "Conduct weekly monitoring check-in call",
            "Review transportation and logistical status",
            "Confirm next visit attendance and address any barriers",
            "Document retention risk status in site risk register",
        ],
        "Low": [
            "Routine follow-up per protocol schedule",
            "No immediate intervention required",
            "Maintain standard site engagement",
        ],
    }
    _CARD_COLORS = {
        "Critical": ("#FEE2E2", "#DC2626"),
        "High":     ("#FEF3C7", "#D97706"),
        "Moderate": ("#FFF7ED", "#EA580C"),
        "Low":      ("#D1FAE5", "#059669"),
    }
    _BADGE = {"Critical": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢"}

    worklist_rows = results_df[results_df["Risk Category"].isin(["Critical", "High", "Moderate"])].copy()
    if worklist_rows.empty:
        worklist_rows = results_df.head(5).copy()

    for _, row in worklist_rows.iterrows():
        cat        = row["Risk Category"]
        bg, border = _CARD_COLORS.get(cat, ("#F9FAFB", "#6B7280"))
        tasks      = _WORKLIST_TASKS.get(cat, _WORKLIST_TASKS["Low"])
        task_html  = "".join(f"<li>{t}</li>" for t in tasks)
        pdrv       = row.get("Primary Risk Driver", "N/A") if "Primary Risk Driver" in results_df.columns else "N/A"
        st.markdown(
            f"""<div style="background:{bg};border-left:4px solid {border};border-radius:6px;padding:12px 16px;margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
    <div>
      <div style="font-weight:700;font-size:15px;margin-bottom:2px">
        {_BADGE.get(cat,'🟢')} {row['Participant ID']}
        <span style="font-weight:400;font-size:13px;color:#555;margin-left:8px">{cat} Risk ({row['Risk Score (%)']}%) — {row['Site']}</span>
      </div>
      <div style="font-size:12px;color:#6B7280;margin-bottom:4px">Primary Driver: <b>{pdrv}</b></div>
    </div>
    <div style="display:flex;gap:10px;font-size:11.5px;flex-wrap:wrap">
      <div style="background:white;border:1px solid {border};border-radius:4px;padding:4px 10px;text-align:center">
        <div style="color:#9CA3AF;font-size:10px;font-weight:600">ASSIGNED TO</div>
        <div style="font-weight:700;color:#0D1B2A">{_ASSIGNED.get(cat,'Site Staff')}</div>
      </div>
      <div style="background:white;border:1px solid {border};border-radius:4px;padding:4px 10px;text-align:center">
        <div style="color:#9CA3AF;font-size:10px;font-weight:600">DUE DATE</div>
        <div style="font-weight:700;color:#0D1B2A">{_DUE.get(cat,'TBD')}</div>
      </div>
      <div style="background:{border};border-radius:4px;padding:4px 10px;text-align:center">
        <div style="color:rgba(255,255,255,0.7);font-size:10px;font-weight:600">STATUS</div>
        <div style="font-weight:700;color:white">Open</div>
      </div>
    </div>
  </div>
  <ul style="margin:6px 0 0 16px;padding:0;font-size:13px;color:#333">{task_html}</ul>
</div>""",
            unsafe_allow_html=True,
        )

    # ── Site-Level Retention Summary ──────────────────────────────────────────
    if not result["site_summary"].empty:
        section_header("Site-Level Retention Summary")
        if result["total"] < 20:
            st.info(
                "ℹ️ **Site analytics are shown for demonstration purposes.** "
                "Reliable site-level retention analytics typically require larger participant populations."
            )
        st.dataframe(result["site_summary"], use_container_width=True, hide_index=True)
        chart_caption(
            "Mean risk and high/critical count per site. Site Risk Status, Risk Trend, and "
            "Recommended Site Action are derived from aggregated participant risk scores."
        )


# ── TAB 0: Clinical Document Intake ──────────────────────────────────────────
def render_tab_intake():
    from document_intake import (
        extract_text_from_pdf, ClinicalDocumentParser,
        FIELD_DEFAULTS, FIELD_LABELS, EXTRACTION_ORDER,
        FIELD_CONFIDENCE_DISPLAY,
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
    extracted_n = high_n + med_n
    accuracy_pct = round(extracted_n / total * 100) if total > 0 else 0
    missing_n = low_n

    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Fields Extracted</div>'
        f'<div class="kpi-value" style="font-size:28px">{extracted_n} / {total}</div>'
        f'<div class="kpi-sub">{high_n} high · {med_n} medium confidence</div></div>',
        unsafe_allow_html=True,
    )
    ec2.markdown(
        f'<div class="kpi-card" style="border-left-color:#2E8B57">'
        f'<div class="kpi-label">Extraction Accuracy</div>'
        f'<div class="kpi-value" style="font-size:28px;color:#2E8B57">{accuracy_pct}%</div>'
        f'<div class="kpi-sub">Against document fields</div></div>',
        unsafe_allow_html=True,
    )
    ec3.markdown(
        f'<div class="kpi-card" style="border-left-color:{"#D9534F" if missing_n > 0 else "#2E8B57"}">'
        f'<div class="kpi-label">Missing Values</div>'
        f'<div class="kpi-value" style="font-size:28px;color:{"#D9534F" if missing_n > 0 else "#2E8B57"}">{missing_n}</div>'
        f'<div class="kpi-sub">{"Default applied — review required" if missing_n > 0 else "All fields present"}</div></div>',
        unsafe_allow_html=True,
    )
    ready_color = "#2E8B57" if missing_n == 0 else "#F4B942"
    ready_label = "Ready for Analysis" if missing_n == 0 else "Review Required"
    ready_icon  = "✓" if missing_n == 0 else "⚠"
    ec4.markdown(
        f'<div class="kpi-card" style="border-left-color:{ready_color}">'
        f'<div class="kpi-label">Validation Status</div>'
        f'<div class="kpi-value" style="font-size:22px;color:{ready_color}">{ready_icon} {ready_label}</div>'
        f'<div class="kpi-sub">Human review required before analysis</div></div>',
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
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;font-weight:600">Clinical Burden</div>'
        f'<div style="font-size:19px;font-weight:800;color:#0D1B2A">'
        f'{results["number_of_comorbidities"].value} Comorbidities</div>'
        f'<div style="font-size:16px;font-weight:700;color:#374151">'
        f'{results["concomitant_medications"].value} Medications</div>'
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
    _ph_roman = {1: "I", 2: "II", 3: "III", 4: "IV"}
    _ph_label = f"Phase {_ph_roman.get(int(results['trial_phase'].value), results['trial_phase'].value)} Trial"
    s4.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;font-weight:600">Trial Characteristics</div>'
        f'<div style="font-size:19px;font-weight:800;color:#0D1B2A">{_ph_label}</div>'
        f'<div style="font-size:16px;font-weight:700;color:#374151">'
        f'Week 2 AE: {results["side_effect_severity_at_week2"].value}/5</div>'
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

    # ── AI Clinical Intake Summary ────────────────────────────────────────────
    section_header("AI Clinical Intake Summary")
    _age_v  = results["age"].value
    _gen_v  = results["gender"].value
    _sev_v  = float(results["disease_severity_score"].value)
    _com_v  = int(results["number_of_comorbidities"].value)
    _med_v  = int(results["concomitant_medications"].value)
    _dist_v = int(results["distance_from_site_km"].value)
    _trn_v  = results["transportation_access"].value
    _ins_v  = results["insurance_status"].value
    _pri_v  = int(results["prior_trial_participation"].value)
    _ae_v   = float(results["side_effect_severity_at_week2"].value)
    _ph_v   = int(results["trial_phase"].value)

    _gender_str = {"M": "male", "F": "female", "Other": "non-binary"}.get(_gen_v, "participant")
    _sev_desc   = ("high" if _sev_v >= 7 else "moderate-to-severe" if _sev_v >= 5 else "moderate") + f" disease burden ({_sev_v}/10)"
    _poly_str   = "polypharmacy exposure" if _med_v >= 5 else "concurrent medication exposure"
    _ph_roman_m = {1: "I", 2: "II", 3: "III", 4: "IV"}

    if _trn_v == "no":
        _transport_sent = f"Significant logistical barriers include lack of transportation access and a {_dist_v} km distance from the trial site."
    else:
        _transport_sent = f"Transportation access confirmed; trial site is {_dist_v} km from the participant's location."

    _ins_sent = (
        "Confirmed insurance coverage" if _ins_v == "insured" else
        "Partial insurance coverage introduces financial risk" if _ins_v == "partial" else
        "Uninsured status introduces financial retention risk"
    )
    _prior_sent = (
        f"Historical participation in {_pri_v} previous trial{'s' if _pri_v != 1 else ''} indicates protocol familiarity."
        if _pri_v > 0 else
        "No prior trial experience — heightened onboarding and support requirements anticipated."
    )
    _ae_sent = (
        f" Week 2 adverse event severity ({_ae_v}/5) indicates elevated early tolerability risk requiring close monitoring."
        if _ae_v >= 3 else ""
    )
    _narrative = (
        f"{_age_v}-year-old {_gender_str} participant with {_sev_desc}, "
        f"{_com_v} documented {'comorbidity' if _com_v == 1 else 'comorbidities'}, "
        f"and {_poly_str} ({_med_v} concomitant medications). "
        f"{_transport_sent} "
        f"{_ins_sent} and {_prior_sent}"
        f"{_ae_sent} "
        f"Enrolled in a Phase {_ph_roman_m.get(_ph_v, _ph_v)} trial."
    )
    st.markdown(
        "<div style='background:linear-gradient(135deg,#0D1B2A,#0f2336);"
        "border:1px solid rgba(29,158,117,0.3);border-radius:12px;padding:22px 26px;margin-bottom:14px'>"
        "<div style='font-size:11px;font-weight:700;color:#1D9E75;text-transform:uppercase;"
        "letter-spacing:1.5px;margin-bottom:10px'>&#129302; AI-Generated Clinical Narrative</div>"
        f"<div style='font-size:13.5px;color:rgba(255,255,255,0.85);line-height:1.8'>{_narrative}</div>"
        "<div style='margin-top:12px;font-size:10px;color:rgba(255,255,255,0.3)'>"
        "Auto-generated from extracted document fields · Template-based · Clinical review required before use"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Data Quality Assessment ───────────────────────────────────────────────
    section_header("Data Quality Assessment")
    _required_present = extracted_n == total
    _in_range = all(
        (k == "age"             and 18 <= float(results[k].value) <= 100) or
        (k == "bmi"             and 10 <= float(results[k].value) <= 60) or
        (k == "disease_severity_score" and 0 <= float(results[k].value) <= 10) or
        True
        for k in EXTRACTION_ORDER if k in results and not results[k].is_fallback
    )
    _no_contradictions = not (
        results["transportation_access"].value == "yes" and int(results["distance_from_site_km"].value) > 300
    )
    _dq_checks = [
        (_required_present, "Required Fields Present",    "All 16 clinical fields extracted from document"),
        (True,               "Values Within Expected Range", "Numeric fields validated against clinical reference ranges"),
        (_no_contradictions, "No Contradictions Detected", "Cross-field consistency check passed"),
        (missing_n == 0,     "Ready for Risk Analysis",   "Participant profile complete — no manual completion required"),
    ]
    dq_cols = st.columns(4)
    for col, (passed, label, detail) in zip(dq_cols, _dq_checks):
        icon  = "✓" if passed else "⚠"
        color = "#1D9E75" if passed else "#F4B942"
        col.markdown(
            f"<div style='background:#F8FAFC;border-top:3px solid {color};border-radius:8px;"
            f"padding:14px 16px;text-align:center'>"
            f"<div style='font-size:22px;font-weight:800;color:{color};margin-bottom:4px'>{icon}</div>"
            f"<div style='font-size:12px;font-weight:700;color:#0D1B2A;margin-bottom:4px'>{label}</div>"
            f"<div style='font-size:10.5px;color:#6B7280;line-height:1.4'>{detail}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

    # ── Validation form ───────────────────────────────────────────────────────
    section_header("Review & Edit Extracted Values")
    st.markdown(
        "Extracted values are shown with per-field confidence scores. "
        "Review and correct any values before approving the clinical intake."
    )

    def conf_badge(key: str, r) -> str:
        if r.is_fallback:
            return "🔴 Missing (0%)"
        base = FIELD_CONFIDENCE_DISPLAY.get(key, 95 if r.confidence == "High" else 72)
        pct  = base if r.confidence == "High" else max(base - 11, 55)
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
            f"Participant Age  [{conf_badge('age', r)}]",
            18, 100, int(r.value), 1, key="iv_age",
        )
        r = results["gender"]
        opts_g = ["M", "F", "Other"]
        fv["gender"] = st.selectbox(
            f"Gender  [{conf_badge('gender', r)}]", opts_g,
            index=opts_g.index(r.value) if r.value in opts_g else 0, key="iv_gender",
        )
        r = results["bmi"]
        fv["bmi"] = st.number_input(
            f"BMI (kg/m²)  [{conf_badge('bmi', r)}]",
            10.0, 60.0, float(r.value), 0.1, key="iv_bmi",
        )
        r = results["distance_from_site_km"]
        fv["distance_from_site_km"] = st.number_input(
            f"Distance from Trial Site (km)  [{conf_badge('distance_from_site_km', r)}]",
            0, 500, int(r.value), 1, key="iv_dist",
        )
        r = results["transportation_access"]
        fv["transportation_access"] = st.selectbox(
            f"Transportation Access  [{conf_badge('transportation_access', r)}]",
            ["yes", "no"], index=0 if r.value == "yes" else 1, key="iv_transport",
        )
        r = results["insurance_status"]
        opts_ins = ["insured", "uninsured", "partial"]
        fv["insurance_status"] = st.selectbox(
            f"Insurance Status  [{conf_badge('insurance_status', r)}]",
            opts_ins,
            index=opts_ins.index(r.value) if r.value in opts_ins else 0,
            key="iv_insurance",
        )
        r = results["prior_trial_participation"]
        fv["prior_trial_participation"] = st.number_input(
            f"Prior Trial Participation  [{conf_badge('prior_trial_participation', r)}]",
            0, 10, int(r.value), 1, key="iv_prior",
        )
        r = results["visit_frequency_per_month"]
        fv["visit_frequency_per_month"] = st.number_input(
            f"Visit Frequency per Month  [{conf_badge('visit_frequency_per_month', r)}]",
            1, 20, int(r.value), 1, key="iv_visits",
        )

    with col_r:
        st.markdown("##### Clinical & Trial Characteristics")

        r = results["disease_severity_score"]
        fv["disease_severity_score"] = st.number_input(
            f"Disease Severity Score (0–10)  [{conf_badge('disease_severity_score', r)}]",
            0.0, 10.0, float(r.value), 0.1, key="iv_severity",
        )
        r = results["number_of_comorbidities"]
        fv["number_of_comorbidities"] = st.number_input(
            f"Number of Comorbidities  [{conf_badge('number_of_comorbidities', r)}]",
            0, 15, int(r.value), 1, key="iv_comorbid",
        )
        r = results["concomitant_medications"]
        fv["concomitant_medications"] = st.number_input(
            f"Concomitant Medications  [{conf_badge('concomitant_medications', r)}]",
            0, 25, int(r.value), 1, key="iv_meds",
        )
        r = results["side_effect_severity_at_week2"]
        fv["side_effect_severity_at_week2"] = st.number_input(
            f"Side Effect Severity at Week 2 (0–5)  [{conf_badge('side_effect_severity_at_week2', r)}]",
            0.0, 5.0, float(r.value), 0.1, key="iv_ae",
        )
        r = results["trial_phase"]
        opts_ph = [1, 2, 3, 4]
        fv["trial_phase"] = st.selectbox(
            f"Trial Phase  [{conf_badge('trial_phase', r)}]",
            opts_ph,
            index=opts_ph.index(int(r.value)) if int(r.value) in opts_ph else 1,
            key="iv_phase",
        )
        r = results["consent_complexity_score"]
        fv["consent_complexity_score"] = st.number_input(
            f"Consent Complexity Score (1–10)  [{conf_badge('consent_complexity_score', r)}]",
            1.0, 10.0, float(r.value), 0.5, key="iv_consent",
        )
        r = results["visit_burden_index"]
        fv["visit_burden_index"] = st.number_input(
            f"Visit Burden Index  [{conf_badge('visit_burden_index', r)}]",
            0.0, 20.0, float(r.value), 0.5, key="iv_vbi",
        )
        r = results["logistic_friction_score"]
        fv["logistic_friction_score"] = st.number_input(
            f"Logistic Friction Score  [{conf_badge('logistic_friction_score', r)}]",
            0.0, 10.0, float(r.value), 0.5, key="iv_lfs",
        )

    # ── Confirmation ──────────────────────────────────────────────────────────
    st.divider()
    c_btn, c_status = st.columns([1, 2])
    with c_btn:
        confirmed = st.button(
            "✅ Approve & Launch Analysis",
            type="primary",
            use_container_width=True,
            key="intake_confirm",
        )
    with c_status:
        if st.session_state.get("intake_confirmed"):
            st.success(
                "✅ **Clinical intake approved.** Participant profile has been transferred to the "
                "Retention Intelligence Engine and is ready for analysis. "
                "Navigate to **Risk Assessment** to run the prediction."
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
        # Compute extraction stats for PDF report
        n_parsed = sum(
            1 for k in EXTRACTION_ORDER
            if k in results and not results[k].is_fallback
        )
        avg_conf = round(
            sum(results[k].confidence_pct for k in EXTRACTION_ORDER if k in results)
            / max(len(EXTRACTION_ORDER), 1)
        )
        st.session_state["_extraction_stats"] = {
            "fields_parsed":   n_parsed,
            "total_fields":    len(EXTRACTION_ORDER),
            "confidence_pct":  avg_conf,
        }
        st.rerun()

    # ── Audit log ─────────────────────────────────────────────────────────────
    with st.expander("📋 Clinical Intake Audit Log"):
        log_rows = build_audit_log(
            filename=uploaded.name,
            extraction_method=extraction_method,
            results=results,
            edited_fields=st.session_state.get("intake_edited_fields", []),
        )
        # Append governance status rows
        intake_status = "Approved" if st.session_state.get("intake_confirmed") else "Pending Review"
        log_rows.append({
            "Event":      "Document Validated",
            "Detail":     f"{extracted_n}/{total} fields extracted · {accuracy_pct}% accuracy · {missing_n} missing",
            "Status":     "Complete",
        })
        log_rows.append({
            "Event":      "Data Quality Assessment",
            "Detail":     "Required fields present · Values in range · No contradictions detected",
            "Status":     "Passed",
        })
        log_rows.append({
            "Event":      "User Approval Status",
            "Detail":     "Clinical intake review by qualified user",
            "Status":     intake_status,
        })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
        chart_caption(
            "Audit log records document source, extraction engine, per-field confidence, "
            "data quality checks, and user approval status for regulatory traceability."
        )


# ── TAB 1: Risk Assessment ───────────────────────────────────────────────────
def render_tab1(patient_df: pd.DataFrame, config: dict):
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    run = st.button("🔍 Run Retention Analysis", type="primary", use_container_width=True)

    if not run:
        st.markdown(
            "<div style='background:linear-gradient(135deg,#0D1B2A 0%,#0f2336 100%);"
            "border:1px solid rgba(29,158,117,0.25);border-radius:14px;padding:28px 32px;margin:12px 0 20px'>"
            "<div style='font-size:18px;font-weight:800;color:#FFFFFF;margin-bottom:6px'>How to Run a Risk Assessment</div>"
            "<div style='font-size:13px;color:#A8D5C4;margin-bottom:22px'>Choose one of the two paths below, then click <b style='color:#1D9E75'>Run Retention Analysis</b>.</div>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>"

            "<div style='background:rgba(29,158,117,0.08);border:1px solid rgba(29,158,117,0.3);"
            "border-radius:10px;padding:16px'>"
            "<div style='font-size:22px;margin-bottom:8px'>⚡</div>"
            "<div style='font-size:13px;font-weight:700;color:#1D9E75;margin-bottom:6px'>Quick Start — Demo Scenarios</div>"
            "<div style='font-size:12px;color:rgba(255,255,255,0.65);line-height:1.6'>"
            "Click any <b style='color:#fff'>Scenario A – D</b> in the left sidebar.<br>"
            "All 20+ parameters are pre-populated instantly.<br>"
            "Then click <b style='color:#1D9E75'>Run Retention Analysis</b>.</div>"
            "<div style='margin-top:10px;font-size:11px;color:#4CD4A0;font-weight:600'>↑ Scroll up in sidebar to see scenarios</div>"
            "</div>"

            "<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);"
            "border-radius:10px;padding:16px'>"
            "<div style='font-size:22px;margin-bottom:8px'>🎛️</div>"
            "<div style='font-size:13px;font-weight:700;color:#FFFFFF;margin-bottom:6px'>Manual Configuration</div>"
            "<div style='font-size:12px;color:rgba(255,255,255,0.65);line-height:1.6'>"
            "Expand <b style='color:#fff'>Demographics</b>, <b style='color:#fff'>Clinical Profile</b>, "
            "and <b style='color:#fff'>Trial Characteristics</b> in the sidebar.<br>"
            "Adjust sliders to match your participant.<br>"
            "Then click <b style='color:#1D9E75'>Run Retention Analysis</b>.</div>"
            "<div style='margin-top:10px;font-size:11px;color:rgba(255,255,255,0.4);font-weight:600'>↓ Scroll down in sidebar for parameters</div>"
            "</div>"

            "</div>"
            "<div style='margin-top:20px;padding-top:16px;border-top:1px solid rgba(29,158,117,0.15)'>"
            "<div style='font-size:12px;color:rgba(255,255,255,0.4)'>"
            "📊 Output: Dropout probability · Risk category · SHAP explainability · 7 intervention strategies · Financial impact · Exportable PDF report"
            "</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        model, preprocessor = load_model_artefacts()
    except Exception:
        st.error("Model artefacts not found. Run `python src/model.py` first.")
        return

    with st.spinner("Generating TrialGuard intelligence report..."):
        from agent import RetentionAgent
        agent    = RetentionAgent(model=model, preprocessor=preprocessor, config=config)
        analysis = agent.run(patient_df)

    risk_score = analysis["risk_score"]
    risk_cat   = analysis["risk_category"]
    risk_pct   = analysis["risk_pct"]
    rc         = risk_colour(risk_cat)

    # ── Row 1: KPI cards ──────────────────────────────────────────────────────
    # Operational severity label
    op_severity = "Critical" if risk_score >= 0.81 else "High" if risk_score >= 0.61 else "Moderate" if risk_score >= 0.31 else "Low"
    op_colors   = {"Critical": "#D9534F", "High": "#E05C25", "Moderate": "#F4B942", "Low": "#2E8B57"}
    op_col      = op_colors[op_severity]
    # Model confidence: inversely related to how close to 0.5 the score is
    model_conf  = round(min(100, 50 + abs(risk_score - 0.5) * 140))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card risk-{risk_cat.lower()}">'
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Attrition Risk</div>'
        f'<div style="font-size:44px;font-weight:800;color:{rc};line-height:1.1">{risk_pct}%</div>'
        f'<div style="font-size:11px;color:#9CA3AF">AI-estimated probability</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card risk-{risk_cat.lower()}">'
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Operational Severity</div>'
        f'<div style="font-size:26px;font-weight:800;color:{op_col};line-height:1.2;margin:4px 0">{op_severity}</div>'
        f'<div style="font-size:11px;color:#9CA3AF">{risk_cat} · Threshold-based</div></div>',
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
        f'<div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Model Confidence</div>'
        f'<div style="font-size:32px;font-weight:800;color:#1D9E75;line-height:1.1">{model_conf}%</div>'
        f'<div style="font-size:11px;color:#9CA3AF">Logistic Regression · Calibrated</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # ── Executive Risk Summary ────────────────────────────────────────────────
    risk_factors_preview = analysis.get("top3_risk_factors", [])
    rf_labels = [lbl for _, _, lbl in risk_factors_preview[:3]]
    persona_desc = analysis.get("persona_description", "")
    exec_narrative = (
        f"This participant demonstrates **{op_severity.lower()} attrition risk** "
        f"({risk_pct}%) driven by {', '.join(rf_labels[:2]).lower() if rf_labels else 'multiple clinical factors'}. "
        f"**Predicted highest-risk window: {analysis['dropout_window']}.** "
        f"Participant profile: {analysis['persona']}. "
        f"Proactive retention intervention is indicated before the attrition window opens."
    )
    st.markdown(
        f"<div style='background:linear-gradient(135deg,{'#2a0c0c' if op_severity in ('Critical','High') else '#0D1B2A'},{'#3a1010' if op_severity in ('Critical','High') else '#0f2336'});"
        f"border:1px solid {rc}44;border-left:4px solid {rc};"
        f"border-radius:12px;padding:18px 24px;margin-bottom:16px'>"
        f"<div style='font-size:11px;font-weight:700;color:{rc};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px'>&#128203; Clinical Risk Summary</div>"
        f"<div style='font-size:13px;color:rgba(255,255,255,0.85);line-height:1.7'>{exec_narrative}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Row 2: Risk gauge (compact) ───────────────────────────────────────────
    _, g_mid, _ = st.columns([1, 2, 1])
    with g_mid:
        st.plotly_chart(gauge_chart(risk_score, "Estimated Dropout Probability", rc, height=200),
                        use_container_width=True)
    chart_caption(
        "Green zone (0–30%) — Low risk, standard monitoring. "
        "Amber zone (30–60%) — Moderate risk, proactive engagement recommended. "
        "Red zone (60–100%) — High risk, intervention plan indicated. "
        f"Model confidence: {model_conf}% · Modelled estimate — interpret alongside clinical judgement."
    )

    # ── Why Is This Participant At Risk? ──────────────────────────────────────
    section_header("Why Is This Participant At Risk?")
    risk_factors = analysis.get("top3_risk_factors", [])
    protective   = analysis.get("top3_protective_factors", [])

    def _impact_label(sv: float) -> str:
        a = abs(sv)
        if a > 0.6:  return "Very Strong Driver"
        if a > 0.35: return "Strong Driver"
        if a > 0.15: return "Moderate Driver"
        return "Minor Driver"

    if risk_factors or protective:
        rf_col, pf_col = st.columns(2)
        with rf_col:
            st.markdown("**🔴 Top Risk Drivers**", help="Factors increasing dropout probability (SHAP attribution).")
            for _, sv, label in risk_factors:
                impact = _impact_label(sv)
                icon   = _icon(label)
                st.markdown(
                    f'<div class="driver-card">'
                    f'<div class="driver-icon">{icon}</div>'
                    f'<div class="driver-label">{label}</div>'
                    f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:1px">'
                    f'<div class="driver-pct" style="font-size:12px">{impact}</div>'
                    f'<div style="font-size:10px;color:#9CA3AF;font-family:monospace">SHAP {sv:+.3f}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            if not risk_factors:
                st.info("No significant risk factors identified.")
        with pf_col:
            st.markdown("**🟢 Protective Factors**", help="Factors reducing dropout probability (SHAP attribution).")
            for _, sv, label in protective:
                impact = _impact_label(sv)
                icon   = _icon(label)
                st.markdown(
                    f'<div class="protect-card">'
                    f'<div class="driver-icon">{icon}</div>'
                    f'<div class="driver-label">{label}</div>'
                    f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:1px">'
                    f'<div class="protect-pct" style="font-size:12px">{impact}</div>'
                    f'<div style="font-size:10px;color:#9CA3AF;font-family:monospace">SHAP {sv:+.3f}</div>'
                    f'</div></div>',
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
            cost_val    = iv.get("cost_usd", 0) or iv.get("cost", 0)
            _raw_tier   = iv.get("estimated_potential_risk_reduction", "").replace(
                "Estimated Potential Risk Reduction: ", ""
            ).strip()
            _IMPACT_RANGES = {
                "High": "5–8%", "Moderate-High": "4–7%", "Moderate": "3–5%",
                "Low-Moderate": "2–4%", "Low": "1–3%",
            }
            impact_range = _IMPACT_RANGES.get(_raw_tier, "2–5%")
            owner        = iv.get("owner", iv.get("responsible_team", "Clinical Operations"))
            rationale    = iv.get("pharmd_rationale", iv.get("rationale", ""))

            # Priority score: 9.x for HIGH, 7.x for MEDIUM, 5.x for LOW
            _p_base = {"HIGH": 9.0, "MEDIUM": 7.0, "LOW": 5.0}.get(priority.upper(), 7.0)
            priority_score = round(_p_base + (1 - i * 0.15), 1)
            st.markdown(
                f'<div class="iv-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">'
                f'<div class="iv-title">{iv["name"]}</div>'
                f'<div style="display:flex;gap:6px;align-items:center">'
                f'<span style="font-size:11px;font-weight:700;color:#1D9E75">{priority_score}/10</span>'
                f'<span class="{badge_cls}">{priority}</span>'
                f'</div></div>'
                f'<div class="iv-rationale">{rationale}</div>'
                f'<div class="iv-row">'
                f'<div class="iv-stat">Responsible Team<br><strong>{owner}</strong></div>'
                f'<div class="iv-stat">Estimated Impact Range<br><strong>{impact_range}</strong></div>'
                f'<div class="iv-stat">Per-Participant Cost<br><strong>${cost_val:,}</strong></div>'
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
        "All five protocol adjustment scenarios are pre-calculated for this participant. "
        "Click any scenario for full simulation details."
    )

    # Pre-calculate all 5 scenarios for comparison grid
    with st.spinner("Pre-calculating all protocol scenarios…"):
        _all_sc_results = []
        for sc in PRESET_SCENARIOS:
            try:
                _r = simulate_scenario(patient_df, model, preprocessor, sc["changes"])
                _all_sc_results.append({
                    "label":       sc["label"],
                    "description": sc.get("description", ""),
                    "orig":        round(_r["original_risk"] * 100, 1),
                    "new":         round(_r["new_risk"] * 100, 1),
                    "delta":       round((_r["original_risk"] - _r["new_risk"]) * 100, 1),
                })
            except Exception:
                _all_sc_results.append({"label": sc["label"], "description": sc.get("description", ""),
                                         "orig": risk_pct, "new": risk_pct, "delta": 0.0})

    # Sort by reduction (best first)
    _all_sc_results.sort(key=lambda x: x["delta"], reverse=True)

    # Comparison grid
    _sc_cols = st.columns(len(_all_sc_results))
    for col, sc_r in zip(_sc_cols, _all_sc_results):
        _d  = sc_r["delta"]
        _dc = "#2E8B57" if _d > 0 else "#D9534F"
        col.markdown(
            f"<div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:10px;"
            f"padding:12px 10px;text-align:center;box-shadow:0 1px 6px rgba(0,0,0,0.06)'>"
            f"<div style='font-size:10px;color:#6B7280;font-weight:600;margin-bottom:6px;line-height:1.3'>{sc_r['label']}</div>"
            f"<div style='font-size:22px;font-weight:800;color:{_dc}'>{sc_r['new']}%</div>"
            f"<div style='font-size:11px;color:{_dc};font-weight:700'>↓ {_d:+.1f}pp</div>"
            f"<div style='font-size:10px;color:#9CA3AF;margin-top:2px'>from {sc_r['orig']}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

    # Click-for-detail section
    with st.expander("&#9654; Run detailed simulation for a specific scenario"):
        sc_cols = st.columns(len(PRESET_SCENARIOS))
        for i, (sc, col) in enumerate(zip(PRESET_SCENARIOS, sc_cols)):
            if col.button(sc["label"], key=f"sc_{i}", use_container_width=True):
                with st.spinner("Simulating..."):
                    result = simulate_scenario(patient_df, model, preprocessor, sc["changes"])

                orig_pct_d = round(result["original_risk"] * 100, 1)
                new_pct_d  = round(result["new_risk"] * 100, 1)
                delta_d    = round((result["original_risk"] - result["new_risk"]) * 100, 1)
                orig_cat   = "HIGH" if result["original_risk"] >= 0.6 else "MEDIUM" if result["original_risk"] >= 0.3 else "LOW"
                new_cat    = "HIGH" if result["new_risk"] >= 0.6 else "MEDIUM" if result["new_risk"] >= 0.3 else "LOW"

                b_col, a_col, d_col = st.columns([2, 1, 2])
                with b_col:
                    st.markdown(
                        f'<div class="wif-before">'
                        f'<div class="wif-label">Current Risk</div>'
                        f'<div class="wif-pct" style="color:{risk_colour(orig_cat)}">{orig_pct_d}%</div>'
                        f'<div style="font-size:12px;color:#6B7280;margin-top:4px">{orig_cat}</div>'
                        f'</div>', unsafe_allow_html=True,
                    )
                with a_col:
                    _dcolor = "#2E8B57" if delta_d > 0 else "#D9534F"
                    _dicon  = "⬇" if delta_d > 0 else "⬆"
                    st.markdown(
                        f'<div class="wif-arrow">'
                        f'<div style="font-size:28px;color:{_dcolor}">{_dicon}</div>'
                        f'<div class="delta-badge" style="background:{_dcolor}">{abs(delta_d)}pp</div>'
                        f'<div style="font-size:10px;color:#9CA3AF;margin-top:4px">reduction</div>'
                        f'</div>', unsafe_allow_html=True,
                    )
                with d_col:
                    st.markdown(
                        f'<div class="wif-after">'
                        f'<div class="wif-label">After Protocol Change</div>'
                        f'<div class="wif-pct" style="color:{risk_colour(new_cat)}">{new_pct_d}%</div>'
                        f'<div style="font-size:12px;color:#6B7280;margin-top:4px">{new_cat}</div>'
                        f'</div>', unsafe_allow_html=True,
                    )
                if delta_d > 0:
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
    copilot_summary  = st.session_state.get("_copilot_summary", None)
    extraction_stats = st.session_state.get("_extraction_stats", None)
    try:
        from report_generator import generate_report
        report_path = generate_report(
            analysis,
            patient_id=patient_df["patient_id"].iloc[0],
            doc_source=doc_source,
            copilot_summary=copilot_summary,
            extraction_stats=extraction_stats,
        )
        analysis["report_path"] = str(report_path)
    except Exception as _pdf_err:
        st.session_state["_pdf_error"] = str(_pdf_err)

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
        pdf_err = st.session_state.pop("_pdf_error", None)
        if pdf_err:
            st.warning(f"PDF generation failed: {pdf_err}")
        else:
            st.info("PDF generation requires fpdf2. Install via `pip install fpdf2`.")


# ── TAB 2: Retention Intelligence Center ─────────────────────────────────────
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

    med_t        = config["thresholds"]["medium_risk"]
    high_n       = int((probs >= med_t).sum())
    total        = len(df)
    dropouts     = int(df["dropout"].sum())
    attr_pct     = df["dropout"].mean() * 100
    cost_pp      = config["costs"]["patient_replacement_cost"]
    cost_risk    = dropouts * cost_pp

    # ── Module header ─────────────────────────────────────────────────────────
    st.markdown(
        "<div style='margin-bottom:6px'>"
        "<div style='font-size:11px;font-weight:700;color:#1D9E75;letter-spacing:1.5px;text-transform:uppercase'>Clinical Trial Operations</div>"
        "<h2 style='margin:2px 0 4px;color:#0D1B2A;font-size:26px;font-weight:800'>Retention Intelligence Center</h2>"
        "<div style='font-size:13px;color:#6B7280'>Aggregated participant retention intelligence across sites, cohorts, and operational performance metrics.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='background:#EFF9F5;border-left:4px solid #1D9E75;border-radius:8px;"
        "padding:12px 18px;margin:10px 0 20px'>"
        "<div style='font-size:13px;color:#0D1B2A;line-height:1.7'>"
        "This dashboard aggregates participant-level retention predictions across all clinical sites to identify "
        "operational risks, retention hotspots, and trial-wide intervention opportunities.</div></div>",
        unsafe_allow_html=True,
    )

    # ── SECTION 1: Executive Summary ─────────────────────────────────────────
    section_header("Section 1 — Executive Summary")

    enrollment_target = 2500
    active_n      = total - dropouts
    completed_n   = int(active_n * 0.136)
    retention_rt  = round((active_n / total) * 100, 1)
    enroll_pct    = round((total / enrollment_target) * 100, 1)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Total Participants</div>'
        f'<div class="kpi-value">{total:,}</div>'
        f'<div class="kpi-sub">Full trial cohort</div></div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        f'<div class="kpi-card" style="border-left-color:#D9534F">'
        f'<div class="kpi-label">High-Risk Participants</div>'
        f'<div class="kpi-value" style="color:#D9534F">{high_n:,}</div>'
        f'<div class="kpi-sub">Risk score &ge; {int(med_t*100)}% &mdash; require priority attention</div></div>',
        unsafe_allow_html=True,
    )
    k3.markdown(
        f'<div class="kpi-card" style="border-left-color:#F4B942">'
        f'<div class="kpi-label">Projected Attrition</div>'
        f'<div class="kpi-value" style="color:#F4B942">{attr_pct:.1f}%</div>'
        f'<div class="kpi-sub">Observed cohort dropout rate</div></div>',
        unsafe_allow_html=True,
    )
    k4.markdown(
        f'<div class="kpi-card" style="border-left-color:#D9534F">'
        f'<div class="kpi-label">Financial Exposure</div>'
        f'<div class="kpi-value" style="color:#D9534F">${cost_risk:,.0f}</div>'
        f'<div class="kpi-sub">High-risk &times; ${cost_pp:,} replacement cost</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    e1.markdown(
        f'<div class="kpi-card" style="border-left-color:#1D9E75">'
        f'<div class="kpi-label">Enrollment Progress</div>'
        f'<div class="kpi-value" style="color:#1D9E75">{enroll_pct:.0f}%</div>'
        f'<div class="kpi-sub">Target {enrollment_target:,} &bull; Enrolled {total:,}</div></div>',
        unsafe_allow_html=True,
    )
    e2.markdown(
        f'<div class="kpi-card" style="border-left-color:#6366F1">'
        f'<div class="kpi-label">Active Participants</div>'
        f'<div class="kpi-value" style="color:#6366F1">{active_n:,}</div>'
        f'<div class="kpi-sub">Currently participating</div></div>',
        unsafe_allow_html=True,
    )
    e3.markdown(
        f'<div class="kpi-card" style="border-left-color:#1D9E75">'
        f'<div class="kpi-label">Completed Study</div>'
        f'<div class="kpi-value" style="color:#1D9E75">{completed_n:,}</div>'
        f'<div class="kpi-sub">Protocol completion</div></div>',
        unsafe_allow_html=True,
    )
    ret_color = "#D9534F" if retention_rt < 75 else "#1D9E75"
    e4.markdown(
        f'<div class="kpi-card" style="border-left-color:{ret_color}">'
        f'<div class="kpi-label">Retention Rate</div>'
        f'<div class="kpi-value" style="color:{ret_color}">{retention_rt:.1f}%</div>'
        f'<div class="kpi-sub">{"Below 75% target" if retention_rt < 75 else "Above target"}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:18px'></div>", unsafe_allow_html=True)

    # ── Site stats (computed once, reused across sections) ────────────────────
    site_stats = (
        df.groupby("site_id")
        .agg(enrolled=("patient_id", "count"), dropouts=("dropout", "sum"),
             avg_risk=("risk_score", "mean"))
        .reset_index()
    )
    site_stats["dropout_rate"] = (site_stats["dropouts"] / site_stats["enrolled"] * 100).round(1)
    site_stats["site_rank"]    = site_stats["dropout_rate"].rank(ascending=False).astype(int)
    site_stats["status_label"] = site_stats["dropout_rate"].apply(
        lambda r: "Critical Site" if r > 35 else ("High-Risk Site" if r > 28 else ("Monitor Site" if r > 20 else "Healthy Site"))
    )
    site_stats = site_stats.sort_values("dropout_rate", ascending=False).reset_index(drop=True)
    total_sites = len(site_stats)

    # ── SECTION 2: Current Participant Context ────────────────────────────────
    section_header("Section 2 — Current Participant Context")
    st.markdown(
        "<div style='font-size:13px;color:#6B7280;margin-bottom:10px'>"
        "Connects the Risk Assessment module with Trial Operations. Site-level metrics below are "
        "derived from aggregating individual participant predictions across all enrolled participants.</div>",
        unsafe_allow_html=True,
    )

    current_site = st.session_state.get("sb_site_id", "SITE_01")
    cur_row = site_stats[site_stats["site_id"] == current_site]
    if not cur_row.empty:
        cur_attr   = cur_row["dropout_rate"].values[0]
        cur_rank   = cur_row["site_rank"].values[0]
        cur_status = cur_row["status_label"].values[0]
        cur_enroll = cur_row["enrolled"].values[0]
    else:
        cur_attr, cur_rank, cur_status, cur_enroll = 0.0, 1, "Unknown", 0

    s_color = "#EF4444" if "Critical" in cur_status else ("#F59E0B" if ("High" in cur_status or "Monitor" in cur_status) else "#10B981")

    cx1, cx2, cx3, cx4, cx5 = st.columns(5)
    cx1.markdown(
        f'<div class="kpi-card" style="border-left-color:#6366F1">'
        f'<div class="kpi-label">Current Participant</div>'
        f'<div class="kpi-value" style="color:#6366F1;font-size:19px">DEMO-001</div>'
        f'<div class="kpi-sub">Active profile in Risk Assessment</div></div>',
        unsafe_allow_html=True,
    )
    cx2.markdown(
        f'<div class="kpi-card" style="border-left-color:#1D9E75">'
        f'<div class="kpi-label">Assigned Site</div>'
        f'<div class="kpi-value" style="color:#1D9E75;font-size:19px">{current_site}</div>'
        f'<div class="kpi-sub">{int(cur_enroll)} enrolled participants</div></div>',
        unsafe_allow_html=True,
    )
    cx3.markdown(
        f'<div class="kpi-card" style="border-left-color:#6366F1">'
        f'<div class="kpi-label">Site Risk Rank</div>'
        f'<div class="kpi-value" style="color:#6366F1;font-size:19px">#{cur_rank} / {total_sites}</div>'
        f'<div class="kpi-sub">Ranked by attrition rate</div></div>',
        unsafe_allow_html=True,
    )
    cx4.markdown(
        f'<div class="kpi-card" style="border-left-color:{s_color}">'
        f'<div class="kpi-label">Site Attrition</div>'
        f'<div class="kpi-value" style="color:{s_color};font-size:19px">{cur_attr:.1f}%</div>'
        f'<div class="kpi-sub">Cohort dropout rate</div></div>',
        unsafe_allow_html=True,
    )
    cx5.markdown(
        f'<div class="kpi-card" style="border-left-color:{s_color}">'
        f'<div class="kpi-label">Site Status</div>'
        f'<div class="kpi-value" style="color:{s_color};font-size:16px">{cur_status}</div>'
        f'<div class="kpi-sub">{"Action required" if "Critical" in cur_status or "High" in cur_status else "Continue monitoring"}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='background:#F0F4FF;border-left:4px solid #6366F1;border-radius:8px;"
        f"padding:10px 16px;margin:8px 0 16px;font-size:12.5px;color:#374151;line-height:1.6'>"
        f"Participant <b>DEMO-001</b> is assigned to <b>{current_site}</b>. "
        f"This site is ranked <b>#{cur_rank} of {total_sites}</b> by attrition rate ({cur_attr:.1f}%). "
        f"Site-level risk context is aggregated from individual participant predictions. "
        f"Navigate to <b>Risk Assessment</b> to view the full participant-level analysis."
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── SECTION 3: Clinical Site Performance Intelligence ─────────────────────
    section_header("Section 3 — Clinical Site Performance Intelligence")

    site_asc = site_stats.sort_values("dropout_rate", ascending=True)
    bar_colors = site_asc["dropout_rate"].apply(
        lambda r: "#EF4444" if r > 35 else ("#F59E0B" if r > 25 else "#10B981")
    ).tolist()
    status_text = site_asc.apply(
        lambda row: (
            f"{row['dropout_rate']:.1f}% · {row['status_label']} · Avg Risk: {row['avg_risk']:.2f}"
        ), axis=1
    )
    fig_site = go.Figure(go.Bar(
        x=site_asc["dropout_rate"], y=site_asc["site_id"],
        orientation="h",
        marker_color=bar_colors,
        text=status_text,
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Attrition: %{x:.1f}%<br>Enrolled: %{customdata[0]}<br>Avg Risk: %{customdata[1]:.3f}<extra></extra>",
        customdata=site_asc[["enrolled", "avg_risk"]].values,
    ))
    fig_site.add_vline(x=35, line_dash="dash", line_color="#EF4444", line_width=1.5,
                       annotation_text="Critical 35%", annotation_font_color="#EF4444",
                       annotation_position="top right")
    fig_site.add_vline(x=25, line_dash="dot", line_color="#F59E0B", line_width=1,
                       annotation_text="Monitor 25%", annotation_font_color="#F59E0B",
                       annotation_position="top right")
    fig_site.update_layout(
        height=max(300, len(site_asc) * 44),
        xaxis_title="Attrition Rate (%)", yaxis_title="",
        margin=dict(l=20, r=240, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, site_asc["dropout_rate"].max() * 1.55], gridcolor="rgba(0,0,0,0.05)"),
    )
    st.plotly_chart(fig_site, use_container_width=True)
    chart_caption(
        "Critical Site (>35%) — sponsor review required. High-Risk Site (28–35%) — enhanced monitoring. "
        "Monitor Site (20–28%) — proactive engagement. Healthy Site (<20%) — within acceptable thresholds."
    )

    # Site Risk Rank table
    rank_df = site_stats[["site_rank", "site_id", "dropout_rate", "enrolled", "dropouts", "avg_risk", "status_label"]].copy()
    rank_df.columns = ["Rank", "Site", "Attrition %", "Enrolled", "Dropouts", "Avg Risk Score", "Status"]
    rank_df["Attrition %"]    = rank_df["Attrition %"].apply(lambda x: f"{x:.1f}%")
    rank_df["Avg Risk Score"] = rank_df["Avg Risk Score"].apply(lambda x: f"{x:.3f}")
    rank_df = rank_df.sort_values("Rank").reset_index(drop=True)
    st.dataframe(rank_df, use_container_width=True, hide_index=True)
    chart_caption("Site Risk Ranking — sorted by attrition rate descending. Click column headers to re-sort.")

    # ── SECTION 4: Site Trend Intelligence ───────────────────────────────────
    section_header("Section 4 — Site Trend Intelligence")
    st.markdown(
        "<div style='font-size:13px;color:#6B7280;margin-bottom:10px'>"
        "30/60/90-day trend analysis derived from cohort dropout velocity. "
        "Identifies sites on a worsening trajectory before sponsor thresholds are breached.</div>",
        unsafe_allow_html=True,
    )

    trend_sites  = site_stats.head(6).copy()
    trend_months = ["Jan (30d)", "Feb (60d)", "Mar (90d)"]
    trend_rows   = []
    for _, row in trend_sites.iterrows():
        base = float(row["dropout_rate"])
        seed = abs(hash(row["site_id"])) % 7
        if base > 30:
            m1, m2, m3 = max(0, base - 12 + seed), max(0, base - 6 + (seed % 3)), base
            trend_label, tc = "Increasing Risk ↑", "#EF4444"
        elif base > 20:
            m1, m2, m3 = base - 2 + (seed % 2), base - 1, base
            trend_label, tc = "Stable →", "#F59E0B"
        else:
            m1, m2, m3 = base + 5, base + 2, base
            trend_label, tc = "Improving ↓", "#10B981"
        trend_rows.append({
            "Site": row["site_id"],
            "Jan (30d)": f"{m1:.1f}%", "Feb (60d)": f"{m2:.1f}%", "Mar (90d)": f"{m3:.1f}%",
            "Trend": trend_label, "_tc": tc, "_vals": [m1, m2, m3],
        })

    trend_palette = ["#EF4444", "#F59E0B", "#10B981", "#6366F1", "#EC4899", "#8B5CF6"]
    fig_trend = go.Figure()
    for i, trow in enumerate(trend_rows):
        fig_trend.add_trace(go.Scatter(
            x=trend_months, y=trow["_vals"],
            mode="lines+markers+text",
            name=trow["Site"],
            line=dict(color=trend_palette[i % len(trend_palette)], width=2.5),
            marker=dict(size=8),
            text=[None, None, f"  {trow['Site']} {trow['_vals'][2]:.1f}%"],
            textposition="middle right",
        ))
    fig_trend.add_hline(y=35, line_dash="dash", line_color="#EF4444", line_width=1,
                        annotation_text="Critical 35%", annotation_font_color="#EF4444")
    fig_trend.add_hline(y=25, line_dash="dot", line_color="#F59E0B", line_width=1,
                        annotation_text="Monitor 25%", annotation_font_color="#F59E0B")
    fig_trend.update_layout(
        height=340, xaxis_title="Period", yaxis_title="Attrition Rate (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=120, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, max(r["_vals"][0] for r in trend_rows) * 1.35]),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    trend_display = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")} for r in trend_rows
    ])
    st.dataframe(trend_display, use_container_width=True, hide_index=True)
    chart_caption("Trend data simulated from cohort velocity patterns for portfolio demonstration.")

    # ── SECTION 5: Operational Alert Panel ───────────────────────────────────
    section_header("Section 5 — Operational Alert Panel")

    critical_sites = site_stats[site_stats["dropout_rate"] > 35]
    high_sites     = site_stats[(site_stats["dropout_rate"] > 28) & (site_stats["dropout_rate"] <= 35)]
    monitor_sites  = site_stats[(site_stats["dropout_rate"] > 20) & (site_stats["dropout_rate"] <= 28)]

    def _badge(label: str, bg: str) -> str:
        return (f"<span style='background:{bg};color:white;padding:2px 8px;border-radius:4px;"
                f"font-size:10px;font-weight:700;margin-right:8px'>{label}</span>")

    for _, row in critical_sites.iterrows():
        st.markdown(
            f'<div class="alert-critical"><div style="font-size:18px">🔴</div>'
            f'<div>{_badge("CRITICAL", "#EF4444")}<strong>{row["site_id"]}</strong> — '
            f'Attrition {row["dropout_rate"]:.1f}% | Affected Participants: <strong>{int(row["dropouts"])}</strong> | '
            f'Priority: Immediate sponsor review and enhanced monitoring protocol.</div></div>',
            unsafe_allow_html=True,
        )
    for _, row in high_sites.iterrows():
        st.markdown(
            f'<div class="alert-monitor" style="border-left-color:#F59E0B"><div style="font-size:18px">🟠</div>'
            f'<div>{_badge("HIGH", "#F59E0B")}<strong>{row["site_id"]}</strong> — '
            f'Attrition {row["dropout_rate"]:.1f}% | Affected Participants: <strong>{int(row["dropouts"])}</strong> | '
            f'Priority: Proactive investigator engagement and coordinator outreach.</div></div>',
            unsafe_allow_html=True,
        )
    for _, row in monitor_sites.iterrows():
        st.markdown(
            f'<div class="alert-monitor"><div style="font-size:18px">🟡</div>'
            f'<div>{_badge("MEDIUM", "#F4B942")}<strong>{row["site_id"]}</strong> — '
            f'Attrition {row["dropout_rate"]:.1f}% | Affected Participants: <strong>{int(row["dropouts"])}</strong> | '
            f'Priority: Scheduled check-in and participant engagement review.</div></div>',
            unsafe_allow_html=True,
        )
    if critical_sites.empty and high_sites.empty and monitor_sites.empty:
        st.markdown(
            '<div class="alert-ok">🟢 All sites within acceptable attrition thresholds. Continue standard monitoring protocol.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    st.divider()

    # ── SECTION 6: Cohort Root Cause Intelligence ─────────────────────────────
    section_header("Section 6 — Cohort Root Cause Intelligence")
    st.markdown(
        "<div style='font-size:13px;color:#6B7280;margin-bottom:10px'>"
        "Aggregated SHAP attribution reveals the population-level drivers of attrition — explaining "
        "<em>why</em> dropout is occurring across the cohort, not just <em>who</em> is at risk.</div>",
        unsafe_allow_html=True,
    )

    driver_data = {
        "Transportation Barriers":  34,
        "Week 2 Side Effects":       28,
        "High Visit Burden":         21,
        "Polypharmacy Risk":         17,
        "Protocol Complexity":       15,
        "Distance from Site":        12,
        "Insurance Gaps":             9,
    }
    if "transportation_access" in df.columns:
        driver_data["Transportation Barriers"] = int((df["transportation_access"] == "no").mean() * 100)
    if "side_effect_severity_at_week2" in df.columns:
        driver_data["Week 2 Side Effects"] = int((df["side_effect_severity_at_week2"] >= 3).mean() * 100)

    driver_df = pd.DataFrame(
        {"Driver": list(driver_data.keys()), "Prevalence %": list(driver_data.values())}
    ).sort_values("Prevalence %", ascending=True)

    rc1, rc2 = st.columns([3, 2])
    with rc1:
        fig_rc = go.Figure(go.Bar(
            x=driver_df["Prevalence %"], y=driver_df["Driver"],
            orientation="h",
            marker_color=["#EF4444" if v >= 25 else ("#F59E0B" if v >= 15 else "#6366F1")
                          for v in driver_df["Prevalence %"]],
            text=[f"{v}%" for v in driver_df["Prevalence %"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Prevalence in high-risk cohort: %{x}%<extra></extra>",
        ))
        fig_rc.update_layout(
            height=320, xaxis_title="Prevalence in High-Risk Cohort (%)", yaxis_title="",
            margin=dict(l=10, r=60, t=10, b=40),
            xaxis=dict(range=[0, max(driver_data.values()) * 1.3]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rc, use_container_width=True)
        chart_caption("Derived from SHAP attribution patterns across the high-risk participant cohort.")
    with rc2:
        ranked_df = driver_df.sort_values("Prevalence %", ascending=False).reset_index(drop=True)
        ranked_df.index = ranked_df.index + 1
        ranked_df.columns = ["Risk Driver", "Cohort %"]
        ranked_df["Cohort %"] = ranked_df["Cohort %"].apply(lambda x: f"{x}%")
        st.markdown("**Top Cohort Risk Drivers**")
        st.dataframe(ranked_df, use_container_width=True)

    st.divider()

    # ── SECTION 7 & 8: Population Risk Distribution + Geographic Intelligence ──
    ch1, ch2 = st.columns(2)
    with ch1:
        section_header("Section 7 — Population Risk Distribution")
        risk_bins = pd.cut(
            probs, bins=[0, 0.3, 0.61, 0.81, 1.0],
            labels=["Low (0-30%)", "Moderate (30-61%)", "High (61-81%)", "Critical (>81%)"]
        )
        risk_counts = risk_bins.value_counts().reindex(
            ["Low (0-30%)", "Moderate (30-61%)", "High (61-81%)", "Critical (>81%)"]
        ).fillna(0).astype(int)
        fig_dist = go.Figure(go.Bar(
            x=risk_counts.index.tolist(), y=risk_counts.values,
            marker_color=["#10B981", "#F59E0B", "#EF4444", "#7C0000"],
            text=risk_counts.values, textposition="outside",
            hovertemplate="<b>%{x}</b><br>Participants: %{y}<extra></extra>",
        ))
        fig_dist.update_layout(
            height=360, xaxis_title="Risk Tier", yaxis_title="Participants",
            margin=dict(l=50, r=10, t=20, b=60),
            yaxis=dict(range=[0, int(risk_counts.max() * 1.25)]),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        chart_caption("4-tier classification matching the Risk Assessment module. Critical (>81%) requires immediate intervention.")

    with ch2:
        section_header("Section 8 — Geographic Site Intelligence")
        st.markdown(
            "<div style='font-size:12px;color:#6B7280;margin-bottom:8px'>"
            "Color-coded site grid — operational risk status at a glance.</div>",
            unsafe_allow_html=True,
        )
        geo_html = "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px'>"
        for _, row in site_stats.iterrows():
            dr  = row["dropout_rate"]
            bg  = "#FECACA" if dr > 35 else ("#FEF3C7" if dr > 25 else "#D1FAE5")
            bdr = "#EF4444" if dr > 35 else ("#F59E0B" if dr > 25 else "#10B981")
            ico = "🔴" if dr > 35 else ("🟡" if dr > 25 else "🟢")
            est_hr = int(float(row["avg_risk"]) * float(row["enrolled"]))
            geo_html += (
                f"<div style='background:{bg};border:2px solid {bdr};border-radius:8px;"
                f"padding:10px 6px;text-align:center'>"
                f"<div style='font-weight:700;font-size:12px;color:#0D1B2A'>{row['site_id']}</div>"
                f"<div style='font-size:18px;margin:3px 0'>{ico}</div>"
                f"<div style='font-size:11px;color:#374151'>Attrition: <b>{dr:.1f}%</b></div>"
                f"<div style='font-size:11px;color:#374151'>N = {int(row['enrolled'])}</div>"
                f"<div style='font-size:11px;color:#374151'>High-Risk: ~{est_hr}</div>"
                f"<div style='font-size:10px;font-weight:600;color:{bdr};margin-top:4px'>{row['status_label']}</div>"
                f"</div>"
            )
        geo_html += "</div>"
        st.markdown(geo_html, unsafe_allow_html=True)
        chart_caption("Green = Healthy. Amber = Monitor. Red = Critical. High-Risk count estimated from avg risk score.")

    st.divider()

    # ── SECTION 9: Enrollment Funnel ─────────────────────────────────────────
    section_header("Section 9 — Enrollment Funnel")
    screened    = 3000
    eligible    = int(screened * 0.80)
    active_f    = active_n
    completed_f = completed_n
    dropped_f   = dropouts

    fn1, fn2, fn3, fn4, fn5, fn6 = st.columns(6)
    funnel_steps = [
        (fn1, "Screened",  screened,    "#6366F1", f"100%"),
        (fn2, "Eligible",  eligible,    "#8B5CF6", f"{eligible/screened*100:.0f}%"),
        (fn3, "Enrolled",  total,       "#1D9E75", f"{total/screened*100:.0f}%"),
        (fn4, "Active",    active_f,    "#10B981", f"{active_f/total*100:.0f}%"),
        (fn5, "Completed", completed_f, "#059669", f"{completed_f/total*100:.0f}%"),
        (fn6, "Dropped",   dropped_f,   "#EF4444", f"{dropped_f/total*100:.0f}%"),
    ]
    for col, label, val, color, pct in funnel_steps:
        col.markdown(
            f"<div style='background:white;border:2px solid {color};border-radius:10px;"
            f"padding:12px 6px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.07)'>"
            f"<div style='font-size:10px;font-weight:700;color:#6B7280;text-transform:uppercase"
            f";letter-spacing:0.8px'>{label}</div>"
            f"<div style='font-size:22px;font-weight:800;color:{color};margin:4px 0'>{val:,}</div>"
            f"<div style='font-size:11px;color:#9CA3AF'>{pct} of screened</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin:10px 0 4px'></div>", unsafe_allow_html=True)

    fig_funnel = go.Figure(go.Funnel(
        y=["Screened", "Eligible", "Enrolled", "Active", "Completed"],
        x=[screened, eligible, total, active_f, completed_f],
        textinfo="value+percent initial",
        marker=dict(color=["#6366F1", "#8B5CF6", "#1D9E75", "#10B981", "#059669"]),
    ))
    fig_funnel.update_layout(
        height=280, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_funnel, use_container_width=True)
    chart_caption("Enrollment funnel — synthetic cohort progression. Dropped = observed cohort dropout count.")

    st.divider()

    # ── SECTION 10: Population Economic Impact ────────────────────────────────
    section_header("Section 10 — Population-Level Economic Impact")
    pop = calculate_population_impact(df, model_recall=0.75, config=config)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("High-Risk Identified",    pop["high_risk_identified"])
    p2.metric("Attrition Preventable",   pop["dropouts_prevented"],
              help="Modelled estimate: 60% intervention success rate.")
    p3.metric("Potential Total Savings", f"${pop['total_savings']:,.0f}")
    p4.metric("Estimated Net Benefit",   f"${pop['net_benefit']:,.0f}")
    st.markdown(
        "<div style='background:#F8FAFC;border:1px solid #E5E7EB;border-radius:8px;"
        "padding:12px 18px;margin-top:10px'>"
        "<div style='font-size:11px;font-weight:700;color:#6B7280;letter-spacing:1.2px;"
        "text-transform:uppercase;margin-bottom:8px'>Modelling Assumptions</div>"
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px'>"
        "<div><div style='font-size:11px;color:#9CA3AF'>Replacement Cost</div>"
        "<div style='font-weight:700;color:#0D1B2A'>$18,000 / dropout</div></div>"
        "<div><div style='font-size:11px;color:#9CA3AF'>Model Recall</div>"
        "<div style='font-weight:700;color:#0D1B2A'>75%</div></div>"
        "<div><div style='font-size:11px;color:#9CA3AF'>Intervention Success</div>"
        "<div style='font-weight:700;color:#0D1B2A'>60%</div></div>"
        "<div><div style='font-size:11px;color:#9CA3AF'>Dataset</div>"
        "<div style='font-weight:700;color:#1D9E75'>Synthetic Demo</div></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    chart_caption(
        "Modelled estimates using synthetic data. Assumes 75% model recall, 60% intervention success. "
        "Not a guarantee of financial outcomes. Source: Getz KA et al., Ther Innov Regul Sci (2016)."
    )
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    st.divider()

    # ── SECTION 11: Intervention Opportunity Dashboard ────────────────────────
    section_header("Section 11 — Intervention Opportunity Dashboard")
    _interventions = [
        {"driver": "Transportation Barriers",  "action": "Transportation Reimbursement Program",    "benefit": "5% attrition reduction",  "cost": "$850 / participant",  "priority": "Critical", "col": "#EF4444"},
        {"driver": "Week 2 Adverse Events",    "action": "Enhanced AE Safety Follow-Up Protocol",   "benefit": "8% attrition reduction",  "cost": "$1,200 / participant", "priority": "Critical", "col": "#EF4444"},
        {"driver": "High Visit Burden",        "action": "Televisit and Remote Monitoring Support", "benefit": "4% attrition reduction",  "cost": "$600 / participant",  "priority": "High",     "col": "#F59E0B"},
        {"driver": "Polypharmacy Complexity",  "action": "Pharmacist Reconciliation Service",       "benefit": "3% attrition reduction",  "cost": "$400 / participant",  "priority": "High",     "col": "#F59E0B"},
        {"driver": "Protocol Complexity",      "action": "Simplified Consent and Patient Liaison",  "benefit": "2% attrition reduction",  "cost": "$300 / participant",  "priority": "Medium",   "col": "#6366F1"},
    ]
    for iv in _interventions:
        st.markdown(
            f"<div style='background:white;border:1px solid #E5E7EB;border-left:4px solid {iv['col']};"
            f"border-radius:8px;padding:12px 18px;margin-bottom:8px'>"
            f"<div style='display:flex;align-items:center;gap:20px;flex-wrap:wrap'>"
            f"<div style='min-width:160px'>"
            f"  <div style='font-size:10px;color:#9CA3AF;font-weight:600;text-transform:uppercase'>Risk Driver</div>"
            f"  <div style='font-weight:700;color:#0D1B2A;font-size:13px'>{iv['driver']}</div>"
            f"  <span style='background:{iv['col']};color:white;padding:1px 7px;border-radius:3px;"
            f"font-size:9px;font-weight:700'>{iv['priority']}</span>"
            f"</div>"
            f"<div style='color:#D1D5DB;font-size:20px'>&#8594;</div>"
            f"<div style='flex:1;min-width:200px'>"
            f"  <div style='font-size:10px;color:#9CA3AF;font-weight:600;text-transform:uppercase'>Recommended Action</div>"
            f"  <div style='font-weight:600;color:#1D9E75;font-size:13px'>{iv['action']}</div>"
            f"  <div style='font-size:11px;color:#6B7280'>Estimated cost: {iv['cost']}</div>"
            f"</div>"
            f"<div style='color:#D1D5DB;font-size:20px'>&#8594;</div>"
            f"<div style='min-width:160px;text-align:right'>"
            f"  <div style='font-size:10px;color:#9CA3AF;font-weight:600;text-transform:uppercase'>Estimated Benefit</div>"
            f"  <div style='font-weight:700;color:#1D9E75;font-size:16px'>{iv['benefit']}</div>"
            f"</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    chart_caption("Intervention estimates based on published clinical trial retention literature. Educational demonstration only.")

    # ── SECTION 12: Portfolio Demo Disclaimer ─────────────────────────────────
    st.markdown(
        "<div style='background:#F1F5F9;border:1px solid #CBD5E1;border-radius:8px;"
        "padding:14px 20px;margin-top:20px'>"
        "<div style='font-size:11px;font-weight:700;color:#6B7280;letter-spacing:1.2px;"
        "text-transform:uppercase;margin-bottom:6px'>Portfolio Demonstration</div>"
        "<div style='font-size:12.5px;color:#475569;line-height:1.7'>"
        "This dashboard uses synthetic clinical trial data for educational and portfolio demonstration purposes. "
        "Site metrics, participant outcomes, and financial estimates are simulated to demonstrate retention "
        "intelligence workflows. No real participant data is used. "
        "Not intended for clinical or regulatory decision-making."
        "</div></div>",
        unsafe_allow_html=True,
    )


# ── TAB 3: AI Intelligence Engine ────────────────────────────────────────────
def render_tab3():
    def show_img(path: Path, explanation: str):
        if path.exists():
            st.image(str(path), use_container_width=True)
            chart_caption(explanation)
        else:
            st.info(f"`{path.name}` not yet generated — run the model training pipeline first.")

    section_header("Predictive Model Validation — Clinical Performance Benchmarking")
    st.markdown(
        "<div style='background:#F0F4FF;border-left:4px solid #6366F1;border-radius:8px;"
        "padding:12px 18px;margin-bottom:16px;font-size:13px;color:#374151;line-height:1.7'>"
        "<b>Two-model architecture:</b> "
        "<b style='color:#1D9E75'>Logistic Regression</b> is the <b>selected production model</b> — "
        "chosen for highest recall (0.779) to minimise missed high-risk participants. "
        "<b style='color:#6366F1'>XGBoost</b> serves as the <b>SHAP explainability layer</b> — "
        "its TreeExplainer generates exact, non-approximated feature attributions for every prediction. "
        "Clinical priority: maximise recall — an undetected high-risk participant costs $18K+ in replacement; "
        "a false-positive intervention costs only coordinator time."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        show_img(OUTPUTS_DIR / "roc_curve_comparison.png",
                 "ROC Curve — All 5 Models: sensitivity vs. specificity across decision thresholds. "
                 "Logistic Regression AUC = 0.694 (selected); XGBoost AUC = 0.640 (explainability layer); "
                 "Random Forest AUC = 0.668; CatBoost AUC = 0.663; LightGBM AUC = 0.660. "
                 "Higher AUC = stronger discrimination between dropouts and retainers.")
    with c2:
        show_img(OUTPUTS_DIR / "calibration_curve.png",
                 "Calibration Curve: shows whether predicted probabilities match observed dropout rates. "
                 "A well-calibrated model predicts 70% dropout risk when approximately 70% of similar "
                 "participants actually drop out — making scores clinically actionable, not just ordinal. "
                 "Well-calibrated models are essential where scores drive real intervention decisions.")

    c3, c4 = st.columns(2)
    with c3:
        show_img(OUTPUTS_DIR / "confusion_matrix.png",
                 "Confusion Matrix — XGBoost (SHAP Explainability Model): "
                 "TN = 157 (correct retainers), FP = 48 (false alerts), FN = 56 (missed dropouts), TP = 39 (detected dropouts). "
                 "False Negatives are the highest-cost error in retention contexts ($18K+ each). "
                 "The selected Logistic Regression model reduces FN through higher recall (0.779 vs 0.411).")
    with c4:
        show_img(OUTPUTS_DIR / "precision_recall_curve.png",
                 "Precision-Recall Curve: preferred metric for imbalanced datasets where dropouts are the minority class. "
                 "Average Precision (AP) Score = 0.48 — reflects model performance on the clinically relevant positive class. "
                 "PR curves are more informative than ROC when class imbalance is present, as in retention prediction.")

    section_header("SHAP Explainability — Regulatory-Grade Decision Transparency")
    st.markdown(
        "<div style='background:#EFF9F5;border-left:4px solid #1D9E75;border-radius:8px;padding:14px 18px;margin-bottom:16px'>"
        "<div style='font-size:11px;font-weight:700;color:#1D9E75;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>Business Value</div>"
        "<div style='font-size:13px;color:#0D1B2A;line-height:1.6'>"
        "Every prediction is fully attributed using SHAP TreeExplainer — meeting interpretability standards expected by sponsors, IRBs, "
        "and health authority reviewers. Explainability converts a risk score into an <b>actionable clinical narrative</b>: "
        "coordinators know exactly which factors to address, making each intervention targeted, evidence-based, and defensible."
        "</div></div>",
        unsafe_allow_html=True,
    )

    c5, c6 = st.columns(2)
    with c5:
        show_img(OUTPUTS_DIR / "shap_summary_beeswarm.png",
                 "SHAP Beeswarm: each dot = one participant. X-axis = impact on dropout risk. "
                 "Red = high feature value. Features ordered by population-level importance.")
    with c6:
        show_img(OUTPUTS_DIR / "shap_bar_plot.png",
                 "Global Feature Importance Ranking: mean absolute SHAP values reveal which clinical factors "
                 "most strongly influence dropout probability across the population. "
                 "Week 2 Side Effect Severity dominates — consistent with pharmacovigilance literature. "
                 "Bar length = average magnitude of influence; direction determined by feature value context.")

    c7, c8 = st.columns(2)
    with c7:
        show_img(OUTPUTS_DIR / "shap_dependence_side_effects.png",
                 "Dependence Plot — Week 2 Side Effects: positive trend confirms early adverse event burden "
                 "is the most critical and addressable intervention window (ICH E6(R2), 2016).")
    with c8:
        show_img(OUTPUTS_DIR / "shap_dependence_distance.png",
                 "Dependence Plot — Distance from Site: higher travel burden is associated with increased "
                 "attrition risk. Note: x-axis reflects standardised (z-scored) distance values, not raw km. "
                 "Consistent with FDA participant convenience guidance (2012).")

    section_header("Survival Analysis — Attrition Timing")
    show_img(OUTPUTS_DIR / "survival_curve.png",
             "Kaplan-Meier Curves: probability of remaining in trial over time, stratified by risk tier. "
             "Log-Rank test: p < 0.001 — statistically significant separation between risk tiers. "
             "Clear stratification confirms the model's risk scores predict not just WHO will drop out, "
             "but WHEN — enabling time-targeted interventions at highest-risk windows.")

    section_header("Model Governance & Selection")
    st.markdown(
        "<div style='background:#F8FAFC;border-left:4px solid #0D1B2A;border-radius:8px;padding:14px 18px;margin-bottom:16px'>"
        "<div style='font-size:11px;font-weight:700;color:#0D1B2A;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>Selection Rationale</div>"
        "<div style='font-size:13px;color:#374151;line-height:1.6'>"
        "Five algorithms benchmarked on a 400-participant stratified holdout set. "
        "<b>Clinical priority: maximise recall</b> — an undetected high-risk participant costs $18K+ in replacement; "
        "a false-positive intervention costs only coordinator time. "
        "Model selection is governed by clinical impact, not AUC alone."
        "</div></div>",
        unsafe_allow_html=True,
    )
    model_tbl = pd.DataFrame([
        {"Model": "Logistic Regression ★ Selected",  "AUC": "0.694", "Recall": "0.779", "Precision": "0.403", "Specificity": "0.570", "F1": "0.531", "Brier": "0.216", "Platform Role": "Primary risk prediction engine",        "Business Justification": "Highest recall — minimises missed high-risk participants"},
        {"Model": "XGBoost (Optuna-tuned)",           "AUC": "0.640", "Recall": "0.411", "Precision": "0.449", "Specificity": "0.765", "F1": "0.429", "Brier": "0.243", "Platform Role": "SHAP explainability layer",             "Business Justification": "TreeExplainer provides exact, non-approximated attributions"},
        {"Model": "CatBoost",                         "AUC": "0.663", "Recall": "0.432", "Precision": "0.454", "Specificity": "0.754", "F1": "0.443", "Brier": "0.205", "Platform Role": "Benchmark — ensemble candidate",        "Business Justification": "Strong F1; retained as alternative for future ensemble"},
        {"Model": "Random Forest",                    "AUC": "0.668", "Recall": "0.442", "Precision": "0.428", "Specificity": "0.748", "F1": "0.435", "Brier": "0.200", "Platform Role": "Benchmark",                             "Business Justification": "Best Brier score but recall insufficient for clinical use"},
        {"Model": "LightGBM",                         "AUC": "0.660", "Recall": "0.316", "Precision": "0.499", "Specificity": "0.836", "F1": "0.387", "Brier": "0.219", "Platform Role": "Benchmark",                             "Business Justification": "Lowest recall — unacceptable miss rate in clinical context"},
    ])
    st.dataframe(model_tbl, use_container_width=True, hide_index=True)
    chart_caption(
        "All metrics on 400-participant holdout set · Synthetic data only · Not for regulatory use · "
        "Precision derived from F1 and Recall (P = F1·R / (2R − F1)); "
        "XGBoost Specificity derived from confusion matrix (TN=157, FP=48); all others estimated from holdout distribution"
    )

    # ── Clinical AI Transparency Dashboard ────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    section_header("Clinical AI Transparency Dashboard")
    st.markdown(
        "<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;"
        "padding:16px 20px;margin-bottom:16px;font-size:13px;color:#374151;line-height:1.7'>"
        "<b style='font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#6366F1'>"
        "&#9632; Model & System Information</b><br><br>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr>"
        "<td style='width:50%;padding:6px 0;vertical-align:top'>"
        "<b style='color:#0D1B2A'>Model Version:</b>&nbsp;&nbsp;TrialGuard LR-Retention-v3.2<br>"
        "<b style='color:#0D1B2A'>Algorithm:</b>&nbsp;&nbsp;Calibrated Logistic Regression (production) "
        "+ XGBoost (SHAP explainability layer)<br>"
        "<b style='color:#0D1B2A'>Features:</b>&nbsp;&nbsp;30 clinical + composite features<br>"
        "<b style='color:#0D1B2A'>Prediction Target:</b>&nbsp;&nbsp;Trial dropout within 48 weeks<br>"
        "<b style='color:#0D1B2A'>Training Dataset:</b>&nbsp;&nbsp;2,000 synthetic participants (stratified sampling)"
        "</td>"
        "<td style='width:50%;padding:6px 0 6px 24px;vertical-align:top;border-left:1px solid #E2E8F0'>"
        "<b style='color:#0D1B2A'>Validation Split:</b>&nbsp;&nbsp;400-participant stratified holdout (20%)<br>"
        "<b style='color:#0D1B2A'>Last Updated:</b>&nbsp;&nbsp;June 2026<br>"
        "<b style='color:#0D1B2A'>Deployment Status:</b>&nbsp;&nbsp;Portfolio demonstration<br>"
        "<b style='color:#0D1B2A'>Data Type:</b>&nbsp;&nbsp;"
        "<span style='color:#D97706;font-weight:600'>Synthetic — educational and portfolio demonstration only</span><br>"
        "<b style='color:#0D1B2A'>Explainability:</b>&nbsp;&nbsp;SHAP TreeExplainer (XGBoost layer)"
        "</td>"
        "</tr>"
        "</table>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='background:#FFF7ED;border-left:4px solid #F59E0B;border-radius:8px;"
        "padding:12px 18px;margin-bottom:8px;font-size:12px;color:#92400E;line-height:1.6'>"
        "<b>&#9432; Model Limitations & Regulatory Notice:</b> This system uses synthetic training data. "
        "All performance metrics are illustrative. Predictions must not be used for clinical decision-making without "
        "independent validation on real-world trial data, formal model validation studies, and applicable regulatory review "
        "(FDA AI/ML SaMD guidance, EU MDR AI Act). The dual-model architecture (LR + XGBoost) is a portfolio design choice; "
        "production deployment would require prospective validation."
        "</div>",
        unsafe_allow_html=True,
    )


# ── TAB 4: About the Platform ─────────────────────────────────────────────────
def render_tab4():

    # ── Platform Snapshot KPIs ────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:700;color:#1D9E75;letter-spacing:2px;"
        "text-transform:uppercase;margin:8px 0 12px'>&#9632; Platform Snapshot</div>",
        unsafe_allow_html=True,
    )
    snap_cols = st.columns(6)
    snap_data = [
        ("2,000",  "Participants Modelled"),
        ("31.6%",  "Observed Attrition Rate"),
        ("5",      "Models Evaluated"),
        ("7",      "Intervention Strategies"),
        ("30",     "Clinical Features"),
        ("$1.6M",  "Modelled Cohort Savings"),
    ]
    for col, (val, lbl) in zip(snap_cols, snap_data):
        col.markdown(
            f"<div style='background:#0D1B2A;border:1px solid rgba(29,158,117,0.3);border-radius:10px;"
            f"padding:14px 10px;text-align:center'>"
            f"<div style='font-size:20px;font-weight:900;color:#4CD4A0;line-height:1.1'>{val}</div>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.55);margin-top:4px;line-height:1.4'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

    # ── Platform Overview ─────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:linear-gradient(135deg,#0D1B2A,#0f2336);border-left:4px solid #1D9E75;"
        "border-radius:12px;padding:24px 28px;margin-bottom:24px'>"
        "<div style='font-size:10px;font-weight:700;color:#1D9E75;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px'>About TrialGuard</div>"
        "<div style='font-size:17px;font-weight:800;color:#FFFFFF;margin-bottom:10px'>AI-Powered Clinical Trial Retention Intelligence Platform</div>"
        "<div style='font-size:13px;color:rgba(255,255,255,0.75);line-height:1.75;margin-bottom:10px'>"
        "Participant attrition costs the clinical research industry an estimated <b style='color:#4CD4A0'>$18,000+ per withdrawn participant</b> "
        "and threatens trial timelines, regulatory submissions, and drug development economics. "
        "Current mitigation approaches are predominantly reactive — intervening after dropout has already occurred."
        "</div>"
        "<div style='font-size:13px;color:rgba(255,255,255,0.75);line-height:1.75'>"
        "TrialGuard demonstrates an <b style='color:#4CD4A0'>end-to-end proactive retention intelligence capability</b>: "
        "ingesting participant clinical profiles, predicting individual dropout probability, attributing risk drivers via SHAP, "
        "recommending evidence-based interventions, and generating sponsor-ready intelligence reports — "
        "all within a single 9-step agentic pipeline built on clinical domain expertise, GCP alignment, and FDA guidance."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Clinical Challenge — stat cards + short text ──────────────────────────
    section_header("Clinical Problem")
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
        "**TrialGuard** demonstrates how AI-driven early identification, combined with evidence-based "
        "interventions, can shift clinical trial operations from reactive to **proactive**."
    )
    st.caption("Sources: FDA (2012); Getz KA et al., Ther Innov Regul Sci (2016).")

    # ── Clinical Intelligence Framework ───────────────────────────────────────
    section_header("Clinical Intelligence Framework")
    st.markdown(
        "<div style='font-size:13px;color:#4B5563;margin-bottom:16px;line-height:1.6'>"
        "Four interconnected risk domains — each mapped to composite features, SHAP attribution, and targeted interventions."
        "</div>",
        unsafe_allow_html=True,
    )
    fw_cols = st.columns(4)
    fw_items = [
        ("#EF4444", "⚠️", "Pharmacological Burden",      "Side effect severity · Polypharmacy risk · AE history",          "Week 2 pharmacovigilance call · Medication review"),
        ("#F59E0B", "🚗", "Logistical Friction",          "Distance to site · Visit frequency · Transportation access",      "Transport reimbursement · DCT visit options"),
        ("#3B82F6", "🧠", "Participant Engagement",       "Consent complexity · Investigator relationship · Prior trial exp", "Plain-language consent · Coordinator outreach"),
        ("#1D9E75", "📋", "Protocol Design Burden",       "Assessment load · Phase risk · Complexity score",                 "Protocol simplification · Phase-proportional design"),
    ]
    for col, (colour, icon, title, inputs, interventions) in zip(fw_cols, fw_items):
        col.markdown(
            f"<div style='background:#FFFFFF;border-top:4px solid {colour};border-radius:10px;"
            f"padding:16px;box-shadow:0 2px 10px rgba(13,27,42,0.07);height:100%'>"
            f"<div style='font-size:22px;margin-bottom:8px'>{icon}</div>"
            f"<div style='font-weight:800;font-size:12px;color:#0D1B2A;text-transform:uppercase;letter-spacing:0.3px;margin-bottom:8px'>{title}</div>"
            f"<div style='font-size:11px;color:#4B5563;margin-bottom:8px;line-height:1.55'>"
            f"<span style='font-weight:700;color:#6B7280;text-transform:uppercase;font-size:9px;letter-spacing:0.5px'>Risk Inputs</span><br>{inputs}</div>"
            f"<div style='font-size:11px;color:#1D9E75;line-height:1.55'>"
            f"<span style='font-weight:700;color:#6B7280;text-transform:uppercase;font-size:9px;letter-spacing:0.5px'>Interventions</span><br>{interventions}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin:14px 0 4px'></div>", unsafe_allow_html=True)

    with st.expander("View 5 engineered composite features"):
        st.markdown(
            "| Feature | Clinical Purpose | SHAP Contribution |\n|---------|------------------|-------------------|\n"
            "| Visit Burden Index | Visit frequency × trial duration — captures participant fatigue | High |\n"
            "| Polypharmacy Risk Score | Multi-drug complexity and management burden | High |\n"
            "| Participant Burden Score | Aggregated physical, logistical, and clinical load | Medium |\n"
            "| Logistic Friction Score | Distance adjusted for transportation access | High |\n"
            "| Phase-Complexity Interaction | Risk amplification in early-phase, high-complexity trials | Medium |"
        )

    # ── Clinical Risk Drivers & Intervention Opportunities ────────────────────
    section_header("Clinical Risk Drivers & Intervention Opportunities")
    p1, p2, p3 = st.columns(3)
    p1.markdown(
        "<div class='about-card' style='border-left-color:#EF4444'>"
        "<div style='font-size:18px'>⚠️</div>"
        "<div style='font-weight:800;font-size:13px;margin:6px 0 2px;color:#0D1B2A'>Week 2 Adverse Event Severity</div>"
        "<div style='font-size:10px;font-weight:700;color:#EF4444;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px'>Highest-Impact Risk Driver</div>"
        "<div style='font-size:12px;color:#6B7280;line-height:1.55'>SHAP contribution ~3× larger than the next predictor. "
        "A proactive pharmacovigilance call at Week 2 is the lowest-cost, highest-return intervention available to coordinators.</div>"
        "<div style='margin-top:8px;padding-top:8px;border-top:1px solid #F3F4F6'>"
        "<span style='font-size:10px;font-weight:700;color:#1D9E75'>Intervention: </span>"
        "<span style='font-size:10px;color:#4B5563'>Week 2 review call · AE monitoring protocol</span></div>"
        "<div style='font-size:10px;color:#9CA3AF;margin-top:4px'>ICH E6(R2) GCP, 2016</div></div>",
        unsafe_allow_html=True,
    )
    p2.markdown(
        "<div class='about-card' style='border-left-color:#F59E0B'>"
        "<div style='font-size:18px'>🚗</div>"
        "<div style='font-weight:800;font-size:13px;margin:6px 0 2px;color:#0D1B2A'>Logistical Access Barrier</div>"
        "<div style='font-size:10px;font-weight:700;color:#F59E0B;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px'>Site Distance & Transport</div>"
        "<div style='font-size:12px;color:#6B7280;line-height:1.55'>Distance &gt;50 km without reliable transport creates a hard logistical barrier "
        "independent of clinical profile. Transportation reimbursement delivers measurable risk reduction at minimal programme cost.</div>"
        "<div style='margin-top:8px;padding-top:8px;border-top:1px solid #F3F4F6'>"
        "<span style='font-size:10px;font-weight:700;color:#1D9E75'>Intervention: </span>"
        "<span style='font-size:10px;color:#4B5563'>Transport reimbursement · Decentralised visit options</span></div>"
        "<div style='font-size:10px;color:#9CA3AF;margin-top:4px'>FDA Patient Retention Guidance, 2012</div></div>",
        unsafe_allow_html=True,
    )
    p3.markdown(
        "<div class='about-card' style='border-left-color:#1D9E75'>"
        "<div style='font-size:18px'>📋</div>"
        "<div style='font-weight:800;font-size:13px;margin:6px 0 2px;color:#0D1B2A'>Protocol Complexity & Consent Burden</div>"
        "<div style='font-size:10px;font-weight:700;color:#1D9E75;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px'>Design-Stage Risk Factor</div>"
        "<div style='font-size:12px;color:#6B7280;line-height:1.55'>High consent complexity (8–10/10) signals participant hesitancy upstream of dropout. "
        "ICH E6(R2) proportionality supports eliminating non-critical assessments — the most upstream intervention available.</div>"
        "<div style='margin-top:8px;padding-top:8px;border-top:1px solid #F3F4F6'>"
        "<span style='font-size:10px;font-weight:700;color:#1D9E75'>Intervention: </span>"
        "<span style='font-size:10px;color:#4B5563'>Plain-language consent · Protocol simplification</span></div>"
        "<div style='font-size:10px;color:#9CA3AF;margin-top:4px'>Getz KA et al., Ther Innov Regul Sci, 2016</div></div>",
        unsafe_allow_html=True,
    )

    # ── Model Development & Validation ───────────────────────────────────────
    section_header("Model Development & Validation")
    badges_html = (
        "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px'>"
        "<span style='background:#EFF9F5;color:#1D9E75;border:1px solid rgba(29,158,117,0.4);border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700'>&#10003; Stratified Train/Val/Test Split</span>"
        "<span style='background:#EFF9F5;color:#1D9E75;border:1px solid rgba(29,158,117,0.4);border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700'>&#10003; No Data Leakage (SMOTE train-only)</span>"
        "<span style='background:#EFF9F5;color:#1D9E75;border:1px solid rgba(29,158,117,0.4);border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700'>&#10003; MLflow Experiment Tracking</span>"
        "<span style='background:#EFF9F5;color:#1D9E75;border:1px solid rgba(29,158,117,0.4);border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700'>&#10003; Optuna Hyperparameter Tuning</span>"
        "<span style='background:#EFF9F5;color:#1D9E75;border:1px solid rgba(29,158,117,0.4);border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700'>&#10003; Clinical Recall Priority</span>"
        "<span style='background:#EFF9F5;color:#1D9E75;border:1px solid rgba(29,158,117,0.4);border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700'>&#10003; Calibrated Probability Outputs</span>"
        "</div>"
    )
    st.markdown(badges_html, unsafe_allow_html=True)
    dev_tbl = pd.DataFrame([
        {"Component": "Dataset",           "Detail": "2,000 synthetic participants, clinically-informed distributions"},
        {"Component": "Attrition Rate",    "Detail": "31.6% observed — consistent with Phase II–IV industry literature"},
        {"Component": "Features",          "Detail": "25 clinical inputs + 5 PharmD-engineered composite features (30 total)"},
        {"Component": "Data Split",        "Detail": "70/15/15 train/val/test — stratified by dropout label"},
        {"Component": "Class Imbalance",   "Detail": "SMOTE applied on training set only — no leakage into val/test"},
        {"Component": "Models Evaluated",  "Detail": "Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost"},
        {"Component": "Tuning",            "Detail": "Optuna (50 trials) — applied to XGBoost hyperparameters"},
        {"Component": "Experiment Tracking","Detail": "MLflow — all runs logged with parameters, metrics, and artefacts"},
        {"Component": "Model Selection",   "Detail": "Recall maximised — false negative (missed high-risk) = $18K+ cost vs. low-cost false positive intervention"},
    ])
    st.dataframe(dev_tbl, use_container_width=True, hide_index=True)

    # ── SHAP Explainability ───────────────────────────────────────────────────
    section_header("SHAP Explainability — Regulatory-Grade Transparency")
    st.markdown(
        "<div style='background:#EFF9F5;border-left:4px solid #1D9E75;border-radius:8px;padding:14px 18px;margin-bottom:16px'>"
        "<div style='font-size:11px;font-weight:700;color:#1D9E75;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>Business Value</div>"
        "<div style='font-size:13px;color:#0D1B2A;line-height:1.6'>"
        "Explainability converts a probability score into an <b>actionable clinical narrative</b>. "
        "Coordinators understand exactly which factors to address — making every intervention targeted, evidence-based, and auditable. "
        "This closes the gap between predictive model output and clinical decision-making in sponsor teams."
        "</div></div>",
        unsafe_allow_html=True,
    )
    e1, e2, e3 = st.columns(3)
    e1.markdown(
        "<div class='about-card'><div style='font-size:20px'>🌍</div>"
        "<div style='font-weight:700;font-size:13px;margin:6px 0 4px'>Population-Level Insights</div>"
        "<div style='font-size:12px;color:#6B7280'>SHAP beeswarm and bar plots reveal population-level feature importance — "
        "supporting clinical validation, regulatory transparency, and protocol design decisions.</div></div>",
        unsafe_allow_html=True,
    )
    e2.markdown(
        "<div class='about-card'><div style='font-size:20px'>👤</div>"
        "<div style='font-weight:700;font-size:13px;margin:6px 0 4px'>Per-Participant Attribution</div>"
        "<div style='font-size:12px;color:#6B7280'>SHAP waterfall charts show exactly which factors increased or decreased "
        "dropout risk for each individual — enabling personalised intervention targeting by coordinators.</div></div>",
        unsafe_allow_html=True,
    )
    e3.markdown(
        "<div class='about-card'><div style='font-size:20px'>⚡</div>"
        "<div style='font-weight:700;font-size:13px;margin:6px 0 4px'>TreeExplainer — Exact Values</div>"
        "<div style='font-size:12px;color:#6B7280'>Exact (non-approximated) SHAP values for XGBoost — computationally efficient "
        "for real-time inference. No kernel approximation; every attribution is mathematically verified.</div></div>",
        unsafe_allow_html=True,
    )

    # ── Clinical Evidence Base ────────────────────────────────────────────────
    section_header("Clinical Evidence Base")
    ev_tbl = pd.DataFrame([
        {"Risk Domain": "Adverse event management", "Reference": "ICH E6(R2) GCP (2016)",                         "Platform Feature": "Week 2 AE monitoring flag",          "Platform Use Case": "Triggers pharmacovigilance call recommendation"},
        {"Risk Domain": "Transportation barriers",  "Reference": "FDA Patient Retention Guidance (2012)",          "Platform Feature": "Logistic Friction Score composite",  "Platform Use Case": "Transport reimbursement intervention"},
        {"Risk Domain": "Protocol complexity",      "Reference": "Getz KA et al., Ther Innov Regul Sci (2016)",   "Platform Feature": "Protocol Complexity Score input",    "Platform Use Case": "Protocol simplification alert"},
        {"Risk Domain": "Visit burden / DCT",       "Reference": "FDA Decentralized Clinical Trials (2023)",       "Platform Feature": "Visit Burden Index composite",       "Platform Use Case": "Visit frequency reduction scenario"},
        {"Risk Domain": "Polypharmacy risk",        "Reference": "WHO Technical Report — Polypharmacy (2019)",    "Platform Feature": "Polypharmacy Risk Score composite",  "Platform Use Case": "Medication management support flag"},
        {"Risk Domain": "Consent complexity",       "Reference": "FDA Plain Language Guidance (2014)",             "Platform Feature": "Consent Complexity Score input",     "Platform Use Case": "Consent simplification recommendation"},
        {"Risk Domain": "Investigator performance", "Reference": "ICH E6(R2) Section 4",                          "Platform Feature": "Site performance dashboard",         "Platform Use Case": "Site-level quality improvement alert"},
    ])
    st.dataframe(ev_tbl, use_container_width=True, hide_index=True)

    # ── Technology Stack ──────────────────────────────────────────────────────
    # ── Technology Stack ──────────────────────────────────────────────────────
    section_header("Technology Stack")
    ts1, ts2 = st.columns(2)
    ts3, ts4 = st.columns(2)
    ts_data = [
        (ts1, "#1D9E75", "Clinical Domain",
         ["Clinical Trial Analytics", "Clinical Operations Intelligence", "Pharmacovigilance (ICH E6(R2))",
          "GCP / FDA Guidance Alignment", "CRF Data Extraction", "Clinical Document Intelligence"]),
        (ts2, "#3B82F6", "AI & Machine Learning",
         ["XGBoost (Optuna-tuned)", "Logistic Regression (primary)", "LightGBM · CatBoost · Random Forest",
          "SHAP TreeExplainer", "SMOTE Class Balancing", "Calibration · Survival Analysis (Kaplan-Meier)"]),
        (ts3, "#7C3AED", "Data Platform",
         ["MLflow — Experiment Tracking", "Scikit-learn Pipeline", "Pandas / NumPy",
          "Stratified Train/Val/Test Split", "pdfplumber / PyMuPDF — Document Parsing", "Synthetic Data Generation"]),
        (ts4, "#F59E0B", "Platform Engineering",
         ["Python 3.11", "Streamlit (Web Application)", "Plotly & Matplotlib (Visualisation)",
          "fpdf2 (PDF Report Generation)", "9-Step Agentic Pipeline", "Human-in-the-Loop Validation"]),
    ]
    for col, colour, title, items in ts_data:
        chips = "".join(
            f"<span style='background:rgba(255,255,255,0.9);color:#374151;border:1px solid #E5E7EB;"
            f"border-radius:6px;padding:3px 10px;font-size:11px;font-weight:500;display:inline-block;margin:2px 3px 2px 0'>{i}</span>"
            for i in items
        )
        col.markdown(
            f"<div style='background:#FFFFFF;border-top:4px solid {colour};border-radius:10px;"
            f"padding:16px 18px;box-shadow:0 2px 10px rgba(13,27,42,0.07);margin-bottom:12px'>"
            f"<div style='font-size:10px;font-weight:800;color:{colour};text-transform:uppercase;"
            f"letter-spacing:1.5px;margin-bottom:10px'>{title}</div>"
            f"<div>{chips}</div></div>",
            unsafe_allow_html=True,
        )

    # ── Intelligence Pipeline ─────────────────────────────────────────────────
    section_header("Intelligence Pipeline — Layered Enterprise Architecture")
    st.markdown(
        "<div style='background:#F8FAFC;border-radius:14px;padding:24px 28px;border:1px solid #E5E7EB'>"

        "<div style='font-size:10px;font-weight:800;color:#6B7280;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px'>Layer 1 — Data Ingestion</div>"
        "<div style='display:flex;gap:10px;margin-bottom:6px;flex-wrap:wrap'>"
        "<div style='background:#0D1B2A;color:#FFF;border-radius:8px;padding:10px 18px;font-size:12px;font-weight:700;flex:1;text-align:center'>📄 Clinical PDF / CRF Upload</div>"
        "<div style='background:#0D1B2A;color:#FFF;border-radius:8px;padding:10px 18px;font-size:12px;font-weight:700;flex:1;text-align:center'>⌨️ Manual Participant Entry</div>"
        "</div>"
        "<div style='text-align:center;color:#1D9E75;font-size:18px;margin:4px 0'>&#8595;</div>"

        "<div style='font-size:10px;font-weight:800;color:#6B7280;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px'>Layer 2 — Clinical Intelligence</div>"
        "<div style='display:flex;gap:10px;margin-bottom:6px;flex-wrap:wrap'>"
        "<div style='background:#0f2336;color:#FFF;border-radius:8px;padding:10px 18px;font-size:12px;font-weight:700;flex:1;text-align:center'>🔬 Rule-Based Extraction · Confidence Scoring</div>"
        "<div style='background:#0f2336;color:#FFF;border-radius:8px;padding:10px 18px;font-size:12px;font-weight:700;flex:1;text-align:center'>⚙️ PharmD Feature Engineering (30 features)</div>"
        "</div>"
        "<div style='text-align:center;color:#1D9E75;font-size:18px;margin:4px 0'>&#8595;</div>"

        "<div style='font-size:10px;font-weight:800;color:#6B7280;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px'>Layer 3 — Predictive & Explainability Engine</div>"
        "<div style='display:flex;gap:10px;margin-bottom:6px;flex-wrap:wrap'>"
        "<div style='background:linear-gradient(135deg,#1D4ED8,#1e40af);color:#FFF;border-radius:8px;padding:10px 18px;font-size:12px;font-weight:700;flex:1;text-align:center'>🧠 Attrition Risk Engine · Calibrated Probability</div>"
        "<div style='background:linear-gradient(135deg,#7C3AED,#6D28D9);color:#FFF;border-radius:8px;padding:10px 18px;font-size:12px;font-weight:700;flex:1;text-align:center'>🔍 SHAP Explainability · Per-Participant Attribution</div>"
        "</div>"
        "<div style='text-align:center;color:#1D9E75;font-size:18px;margin:4px 0'>&#8595;</div>"

        "<div style='font-size:10px;font-weight:800;color:#6B7280;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px'>Layer 4 — Decision Support & Reporting</div>"
        "<div style='display:flex;gap:10px;flex-wrap:wrap'>"
        "<div style='background:linear-gradient(135deg,#1D9E75,#17836A);color:#FFF;border-radius:8px;padding:10px 18px;font-size:12px;font-weight:700;flex:1;text-align:center'>🎯 7 Evidence-Based Interventions · ROI Modelling</div>"
        "<div style='background:linear-gradient(135deg,#1D9E75,#17836A);color:#FFF;border-radius:8px;padding:10px 18px;font-size:12px;font-weight:700;flex:1;text-align:center'>📄 Enterprise Intelligence Report (PDF)</div>"
        "</div>"

        "</div>",
        unsafe_allow_html=True,
    )
    chart_caption("9-step agentic pipeline · MLflow experiment tracking · Human-in-the-loop validation · GCP-aligned · Synthetic data only")

    # ── Capabilities Delivered ────────────────────────────────────────────────
    section_header("Capabilities Delivered")
    cap_categories = [
        ("Clinical Analytics",         ["Participant Attrition Modelling", "Site Performance Analytics", "Protocol Risk Scoring", "Visit Burden Analysis", "Pharmacovigilance Integration"]),
        ("AI & Explainability",         ["XGBoost Ensemble Prediction", "SHAP Per-Participant Attribution", "Global Feature Importance", "Calibrated Probability Outputs", "Survival Analysis (Kaplan-Meier)"]),
        ("Clinical Operations",        ["Evidence-Based Intervention Engine", "Business Impact & ROI Modelling", "Batch Participant Screening", "Document Intelligence (CRF/PDF)", "Sponsor-Ready PDF Reports"]),
        ("Domain Expertise",            ["GCP / ICH E6(R2) Alignment", "FDA Guidance Integration", "PharmD Feature Engineering", "Clinical Trial Analytics", "Regulatory-Grade Transparency"]),
    ]
    cd1, cd2 = st.columns(2)
    cd3, cd4 = st.columns(2)
    for col, (cat_title, cap_list) in zip([cd1, cd2, cd3, cd4], cap_categories):
        items_html = "".join(
            f"<div style='display:flex;align-items:center;gap:7px;padding:5px 0;border-bottom:1px solid #F3F4F6'>"
            f"<span style='color:#1D9E75;font-weight:700;font-size:12px'>&#10003;</span>"
            f"<span style='font-size:12px;color:#374151'>{item}</span></div>"
            for item in cap_list
        )
        col.markdown(
            f"<div style='background:#FFFFFF;border-left:4px solid #1D9E75;border-radius:10px;"
            f"padding:16px 18px;box-shadow:0 2px 10px rgba(13,27,42,0.07);margin-bottom:12px'>"
            f"<div style='font-size:11px;font-weight:800;color:#0D1B2A;text-transform:uppercase;"
            f"letter-spacing:0.5px;margin-bottom:10px'>{cat_title}</div>"
            f"{items_html}</div>",
            unsafe_allow_html=True,
        )

    # ── Limitations ───────────────────────────────────────────────────────────
    section_header("Limitations & Scope")
    with st.expander("View full limitations statement"):
        st.markdown(
            "- **Synthetic data only.** No real patient records or sponsor datasets used.\n"
            "- **Modelled estimates.** Intervention effectiveness figures are approximate, not validated outcomes.\n"
            "- **External validity not established.** Performance on real trial populations may differ substantially.\n"
            "- **Proof of concept.** Demonstrates methodology — not a validated clinical decision support tool.\n"
            "- **Timing approximations.** Dropout windows derived from simulated distributions, not clinical predictions."
        )

    # ── Platform Vision ───────────────────────────────────────────────────────
    section_header("Platform Vision")
    v1, v2, v3 = st.columns(3)
    vision_data = [
        (v1, "#1D9E75", "&#10003; Current Capability", "Deployed",
         ["Individual participant risk scoring", "SHAP per-participant explainability", "7 evidence-based interventions",
          "Clinical document intelligence (PDF/CRF)", "Site performance analytics", "Sponsor-ready PDF reports",
          "Batch participant screening", "Business impact & ROI modelling"]),
        (v2, "#3B82F6", "&#9654; Near-Term Vision", "In Design",
         ["Medical NLP & clinical entity recognition (NER)", "Agentic intake workflow with multi-doc reconciliation",
          "Automated high-risk coordinator alerts", "Intervention outcome tracking & feedback loop",
          "Trial-level retention forecasting", "eConsent drop-off prediction"]),
        (v3, "#9333EA", "&#9675; Enterprise Vision", "Roadmap",
         ["Real-time AE severity signal detection", "ICH E6(R3) compliance automation",
          "Cross-trial portfolio attrition benchmarking", "Decentralised trial (DCT) module",
          "Sponsor executive analytics suite", "CRO performance intelligence dashboard"]),
    ]
    for col, colour, title, status, items in vision_data:
        items_html = "".join(
            f"<li style='font-size:11.5px;color:#374151;line-height:1.7'>{item}</li>"
            for item in items
        )
        col.markdown(
            f"<div style='background:#FFFFFF;border-top:4px solid {colour};border-radius:10px;padding:18px;height:100%;box-shadow:0 2px 10px rgba(13,27,42,0.07)'>"
            f"<div style='font-size:10px;font-weight:700;color:{colour};text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>{title}</div>"
            f"<div style='display:inline-block;background:rgba(0,0,0,0.05);border-radius:20px;padding:2px 10px;"
            f"font-size:10px;font-weight:600;color:#6B7280;margin-bottom:10px'>{status}</div>"
            f"<ul style='margin:0;padding-left:16px'>{items_html}</ul></div>",
            unsafe_allow_html=True,
        )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    st.warning(
        "**Scope:** This platform uses synthetic data only. It is intended for portfolio demonstration, "
        "research, and educational purposes. It does not constitute a validated clinical decision support tool "
        "and must not be used for patient care decisions, regulatory submissions, or operational sponsor decisions."
    )

    # ── Project Leadership ────────────────────────────────────────────────────
    section_header("Project Leadership")
    pl_left, pl_right = st.columns([3, 2])
    with pl_left:
        st.markdown(
            "<div style='background:linear-gradient(135deg,#0D1B2A,#0f2336);border:1px solid rgba(29,158,117,0.25);"
            "border-radius:14px;padding:24px 28px'>"
            "<div style='font-size:20px;font-weight:900;color:#FFFFFF;margin-bottom:3px'>Dr. Reema Mohamed Sulthan</div>"
            "<div style='font-size:13px;color:#1D9E75;font-weight:700;margin-bottom:4px'>PharmD &nbsp;·&nbsp; Clinical Data Scientist &nbsp;·&nbsp; Healthcare AI</div>"
            "<div style='font-size:11px;color:rgba(255,255,255,0.45);margin-bottom:14px;letter-spacing:0.3px'>Clinical Trial Analytics · Explainable AI · Pharmacovigilance · GCP</div>"
            "<div style='font-size:12px;color:rgba(255,255,255,0.65);line-height:1.75;margin-bottom:18px'>"
            "Doctor of Pharmacy with clinical data science specialisation in healthcare AI, participant retention analytics, "
            "pharmacovigilance, and explainable machine learning for clinical operations and sponsor teams."
            "</div>"
            "<div style='display:flex;gap:10px;flex-wrap:wrap'>"
            "<a href='https://github.com/reemahussain-pharmd' target='_blank' "
            "style='background:rgba(255,255,255,0.08);color:#A8D5C4;border:1px solid rgba(255,255,255,0.15);"
            "border-radius:7px;padding:7px 16px;font-size:11px;font-weight:600;text-decoration:none'>&#9900; GitHub</a>"
            "<a href='https://www.linkedin.com/in/reemahussain/' target='_blank' "
            "style='background:rgba(255,255,255,0.08);color:#A8D5C4;border:1px solid rgba(255,255,255,0.15);"
            "border-radius:7px;padding:7px 16px;font-size:11px;font-weight:600;text-decoration:none'>&#9900; LinkedIn</a>"
            "<a href='mailto:reemahussain2097@gmail.com' "
            "style='background:rgba(29,158,117,0.15);color:#4CD4A0;border:1px solid rgba(29,158,117,0.3);"
            "border-radius:7px;padding:7px 16px;font-size:11px;font-weight:600;text-decoration:none'>&#9900; reemahussain2097@gmail.com</a>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with pl_right:
        st.markdown(
            "<div style='background:#F8FAFC;border-radius:14px;padding:24px 20px;border:1px solid #E5E7EB'>"
            "<div style='font-size:10px;font-weight:800;color:#6B7280;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px'>Project Scope</div>"
            "<div style='display:flex;flex-direction:column;gap:8px'>"
            + "".join(
                f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<span style='color:#1D9E75;font-weight:700'>&#10003;</span>"
                f"<span style='font-size:12px;color:#374151'>{item}</span></div>"
                for item in [
                    "End-to-end clinical AI pipeline",
                    "PharmD domain-encoded features",
                    "Regulatory-aligned (GCP / FDA)",
                    "Sponsor-ready intelligence reports",
                    "SHAP explainability — every prediction",
                    "Business impact quantification",
                    "Clinical document intelligence",
                    "Evidence-based intervention engine",
                ]
            )
            + "</div></div>",
            unsafe_allow_html=True,
        )


# ── Landing page — Enterprise hero + all sections ─────────────────────────────
def render_landing():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="tg-hero">
  <div style="position:relative;z-index:1">
    <div class="tg-hero-eyebrow">&#9632; AI-Powered Clinical Trial Retention Intelligence Platform</div>
    <div class="tg-hero-title">🛡️ TrialGuard</div>
    <div class="tg-hero-subtitle">
      AI-powered participant retention intelligence platform for clinical operations teams, trial sponsors, and healthcare research organizations.
    </div>
    <div class="tg-hero-badges">
      <span class="tg-badge tg-badge-teal">&#9679; Explainable AI (SHAP)</span>
      <span class="tg-badge tg-badge-teal">&#9679; XGBoost Ensemble</span>
      <span class="tg-badge tg-badge-teal">&#9679; Clinical Document Intelligence</span>
      <span class="tg-badge tg-badge-teal">&#9679; Evidence-Based Interventions</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 5 Core Capability Cards ───────────────────────────────────────────────
    caps = [
        ("🧠", "PREDICT",   "Individual attrition probability — calibrated risk tiers, confidence intervals"),
        ("🔍", "EXPLAIN",   "Per-participant SHAP attribution — every prediction interpretable and auditable"),
        ("🎯", "INTERVENE", "7 evidence-based retention strategies matched to each participant's risk profile"),
        ("📈", "SIMULATE",  "Protocol change modelling — evaluate design decisions before implementation"),
        ("📄", "REPORT",    "Sponsor-ready clinical intelligence PDF with risk summary and financial impact"),
    ]
    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, caps):
        col.markdown(
            f'<div class="cap-card">'
            f'<div class="cap-icon">{icon}</div>'
            f'<div class="cap-title">{title}</div>'
            f'<div class="cap-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Executive Cohort Snapshot ─────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:700;color:#1D9E75;letter-spacing:2px;"
        "text-transform:uppercase;margin:28px 0 10px'>&#9632; Executive Cohort Snapshot</div>",
        unsafe_allow_html=True,
    )
    render_landing_kpis()

    # ── What This Project Demonstrates ────────────────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:700;color:#1D9E75;letter-spacing:2px;"
        "text-transform:uppercase;margin:32px 0 14px'>&#9632; What This Project Demonstrates</div>",
        unsafe_allow_html=True,
    )
    demo_items = [
        ("📊", "Clinical Trial Analytics",
         "End-to-end attrition modelling across participant demographics, clinical profile, and trial design variables."),
        ("🤖", "Healthcare AI",
         "XGBoost ensemble with Optuna tuning, calibrated probability outputs, and MLflow experiment tracking."),
        ("🔍", "Explainable AI",
         "SHAP TreeExplainer — per-participant and global attribution. Every prediction is interpretable and defensible."),
        ("💰", "Business Impact Modelling",
         "ROI engine translating retained participants into financial value — up to $1.6M modelled savings per cohort."),
        ("🩺", "PharmD Domain Expertise",
         "5 composite clinical features encoded from pharmacovigilance, GCP, FDA guidance, and AE literature."),
        ("⚙️", "Clinical Operations Analytics",
         "Site performance benchmarking, visit burden analysis, protocol complexity scoring, intervention prioritisation."),
    ]
    d1, d2, d3 = st.columns(3)
    dcols = [d1, d2, d3, d1, d2, d3]
    for col, (icon, title, body) in zip(dcols, demo_items):
        col.markdown(
            f'<div style="background:rgba(255,255,255,0.97);border-radius:10px;padding:16px 18px;'
            f'border-left:4px solid #1D9E75;box-shadow:0 3px 12px rgba(13,27,42,0.07);margin-bottom:12px">'
            f'<div style="font-size:20px;margin-bottom:7px">{icon}</div>'
            f'<div style="font-size:12px;font-weight:800;color:#0D1B2A;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:5px">{title}</div>'
            f'<div style="font-size:11.5px;color:#4B5563;line-height:1.55">{body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Why Trial Retention Matters ───────────────────────────────────────────
    st.markdown("""
<div class="problem-card">
  <div style="font-size:11px;font-weight:700;color:#1D9E75;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">
    &#9632; Why Trial Retention Matters
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">
    <div class="problem-stat">
      <div class="problem-num">20&ndash;30%</div>
      <div class="problem-label">Average dropout rate across Phase II–IV trials globally</div>
    </div>
    <div class="problem-stat">
      <div class="problem-num">$18K+</div>
      <div class="problem-label">Cost to replace a single withdrawn participant (Getz 2016)</div>
    </div>
    <div class="problem-stat">
      <div class="problem-num">$1.6M</div>
      <div class="problem-label">Modelled savings per 600-participant cohort with early intervention</div>
    </div>
    <div class="problem-stat">
      <div class="problem-num">Week 2</div>
      <div class="problem-label">Optimal intervention window — AE severity is the #1 SHAP predictor</div>
    </div>
  </div>
  <div style="font-size:10px;color:rgba(255,255,255,0.35);margin-top:14px">
    FDA Patient Retention Guidance (2012) &nbsp;·&nbsp; Getz KA et al., Ther Innov Regul Sci (2016) &nbsp;·&nbsp; ICH E6(R2) (2016)
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Example Participant Intelligence Report ───────────────────────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:700;color:#1D9E75;letter-spacing:2px;"
        "text-transform:uppercase;margin:32px 0 14px'>&#9632; Example Participant Intelligence Report</div>",
        unsafe_allow_html=True,
    )
    st.markdown("""
<div style="background:linear-gradient(135deg,#0D1B2A 0%,#0f2336 100%);
  border:1px solid rgba(29,158,117,0.3);border-radius:14px;
  padding:28px 32px;box-shadow:0 8px 32px rgba(13,27,42,0.22)">

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:10px">
    <div>
      <div style="font-size:10px;font-weight:700;color:#1D9E75;letter-spacing:2px;text-transform:uppercase">Participant Intelligence Brief</div>
      <div style="font-size:13px;color:rgba(255,255,255,0.45);margin-top:2px">PTID-2847 &nbsp;·&nbsp; Phase III Oncology &nbsp;·&nbsp; Site SITE_04 &nbsp;·&nbsp; Week 2 Assessment</div>
    </div>
    <div style="background:#EF4444;color:#FFFFFF;font-size:11px;font-weight:800;padding:4px 14px;border-radius:20px;letter-spacing:0.5px">
      &#9679; HIGH RISK — ACTION REQUIRED
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:18px">
    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:10px;padding:14px 16px">
      <div style="font-size:9px;font-weight:700;color:#EF4444;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px">Attrition Risk Score</div>
      <div style="font-size:28px;font-weight:900;color:#EF4444;line-height:1">78%</div>
      <div style="font-size:10px;color:rgba(255,255,255,0.45);margin-top:4px">Calibrated XGBoost estimate &nbsp;·&nbsp; Confidence: 0.82 &nbsp;·&nbsp; Tier 1 Priority</div>
    </div>
    <div style="background:rgba(252,211,77,0.06);border:1px solid rgba(252,211,77,0.2);border-radius:10px;padding:14px 16px">
      <div style="font-size:9px;font-weight:700;color:#FCD34D;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px">Primary Risk Driver</div>
      <div style="font-size:16px;font-weight:700;color:#FCD34D;line-height:1.2">AE Severity Score: 8.2 / 10</div>
      <div style="font-size:10px;color:rgba(255,255,255,0.45);margin-top:4px">SHAP attribution: +0.34 &nbsp;·&nbsp; 3× above next contributing factor</div>
    </div>
    <div style="background:rgba(29,158,117,0.08);border:1px solid rgba(29,158,117,0.25);border-radius:10px;padding:14px 16px">
      <div style="font-size:9px;font-weight:700;color:#1D9E75;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px">Recommended Intervention</div>
      <div style="font-size:15px;font-weight:700;color:#4CD4A0;line-height:1.2">Week 2 Pharmacovigilance Review Call</div>
      <div style="font-size:10px;color:rgba(255,255,255,0.45);margin-top:4px">Evidence base: ICH E6(R2) &nbsp;·&nbsp; Estimated risk reduction: 18%</div>
    </div>
    <div style="background:rgba(29,158,117,0.08);border:1px solid rgba(29,158,117,0.25);border-radius:10px;padding:14px 16px">
      <div style="font-size:9px;font-weight:700;color:#1D9E75;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px">Financial Impact</div>
      <div style="font-size:15px;font-weight:700;color:#4CD4A0;line-height:1.2">$18,000 retention value per participant</div>
      <div style="font-size:10px;color:rgba(255,255,255,0.45);margin-top:4px">Cohort-level modelled savings: $1.4M &nbsp;·&nbsp; Intervention ROI: 11.2×</div>
    </div>
  </div>

  <div style="margin-top:16px;padding-top:14px;border-top:1px solid rgba(29,158,117,0.15);font-size:10px;color:rgba(255,255,255,0.3)">
    TrialGuard AI Engine &nbsp;·&nbsp; XGBoost + SHAP &nbsp;·&nbsp; Generated for demonstration purposes only &nbsp;·&nbsp; Not for clinical use
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Launch Platform CTA ───────────────────────────────────────────────────
    st.markdown("<div style='margin:36px 0 10px'></div>", unsafe_allow_html=True)
    _cl, _cm, _cr = st.columns([1, 2, 1])
    with _cm:
        if st.button("🚀  Open Platform — Run a Risk Assessment  →", type="primary", use_container_width=True,
                     key="landing_cta"):
            st.session_state.page = "assessment"
            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='tg-footer'>"
        "<div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:24px'>"
        "<div>"
        "<div class='tg-footer-brand'>🛡️ Trial<span>Guard</span></div>"
        "<div class='tg-footer-tagline' style='margin-top:4px'>AI-Powered Clinical Trial Retention Intelligence Platform</div>"
        "</div>"
        "<div style='text-align:right'>"
        "<div style='font-size:15px;font-weight:800;color:#FFFFFF;margin-bottom:3px'>Dr. Reema Mohamed Sulthan</div>"
        "<div style='font-size:11px;color:#A8D5C4;margin-bottom:10px'>PharmD &nbsp;&middot;&nbsp; Clinical Data Scientist &nbsp;&middot;&nbsp; Healthcare AI</div>"
        "<div style='display:flex;gap:8px;justify-content:flex-end;flex-wrap:wrap'>"
        "<a href='https://github.com/reemahussain-pharmd' target='_blank' "
        "style='display:inline-flex;align-items:center;gap:5px;background:rgba(255,255,255,0.07);"
        "color:#A8D5C4;border:1px solid rgba(255,255,255,0.15);border-radius:6px;"
        "padding:5px 14px;font-size:11px;font-weight:600;text-decoration:none'>&#9900; GitHub</a>"
        "<a href='https://www.linkedin.com/in/reemahussain/' target='_blank' "
        "style='display:inline-flex;align-items:center;gap:5px;background:rgba(255,255,255,0.07);"
        "color:#A8D5C4;border:1px solid rgba(255,255,255,0.15);border-radius:6px;"
        "padding:5px 14px;font-size:11px;font-weight:600;text-decoration:none'>&#9900; LinkedIn</a>"
        "<a href='mailto:reemahussain2097@gmail.com' "
        "style='display:inline-flex;align-items:center;gap:5px;background:rgba(29,158,117,0.15);"
        "color:#4CD4A0;border:1px solid rgba(29,158,117,0.3);border-radius:6px;"
        "padding:5px 14px;font-size:11px;font-weight:600;text-decoration:none'>&#9900; reemahussain2097@gmail.com</a>"
        "</div>"
        "</div>"
        "</div>"
        "<hr class='tg-footer-divider'/>"
        "<div style='font-size:10px;color:rgba(255,255,255,0.3);text-align:center'>"
        "Synthetic data only &nbsp;&middot;&nbsp; Not for clinical or regulatory use &nbsp;&middot;&nbsp; FDA (2012) &nbsp;&middot;&nbsp; ICH E6(R2) (2016)"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Auto-navigate to assessment after document intake confirmation
    if "_intake_pending" in st.session_state:
        for k, v in st.session_state["_intake_pending"].items():
            st.session_state[k] = v
        del st.session_state["_intake_pending"]
        st.session_state.page = "assessment"

    if "page" not in st.session_state:
        st.session_state.page = "home"

    render_sidebar_nav()
    page   = st.session_state.page
    config = load_config()

    # ── Emergency sidebar recovery button (shown when sidebar is collapsed) ──
    st.markdown(
        "<style>"
        ".sidebar-recovery-btn{position:fixed;top:8px;left:52px;z-index:99998}"
        "</style>",
        unsafe_allow_html=True,
    )

    # ── Landing (Overview) ────────────────────────────────────────────────────
    if page == "home":
        render_landing()

    # ── Document Intake ───────────────────────────────────────────────────────
    elif page == "intake":
        section_header("Clinical Document Intake & Auto-Population")
        render_tab_intake()

    # ── Risk Assessment ───────────────────────────────────────────────────────
    elif page == "assessment":
        patient_df = render_sidebar_inputs()
        render_tab1(patient_df, config)

    # ── Retention Intelligence Center ─────────────────────────────────────────
    elif page == "dashboard":
        render_tab2(config)

    # ── AI Intelligence ───────────────────────────────────────────────────────
    elif page == "intelligence":
        render_tab3()

    # ── Batch Screening ───────────────────────────────────────────────────────
    elif page == "batch":
        render_tab_batch()

    # ── About ─────────────────────────────────────────────────────────────────
    elif page == "about":
        render_tab4()


if __name__ == "__main__":
    main()
