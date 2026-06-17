"""
PDF Clinical Report Generator.

Produces a professional 2-page PDF report for trial sponsors and clinical
operations teams. Uses fpdf2 for layout. All output is clearly labelled
as educational and portfolio demonstration material.
"""

import io
from fpdf import FPDF
from pathlib import Path
from datetime import date
from typing import Dict

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR  = PROJECT_ROOT / "reports"

GITHUB_URL = (
    "https://github.com/reemahussain-pharmd/"
    "AI-Powered-Clinical-Trial-Retention-Intelligence"
)

DISCLAIMER = (
    "This report is for educational and portfolio demonstration purposes only. "
    "It does not constitute clinical advice and should not be used for patient care decisions."
)
FOOTER_LINE2 = "AI-Powered Clinical Trial Retention Intelligence Platform  v3.0"
FOOTER_LINE3 = "Dr. Reema Mohamed Sulthan, PharmD  |  github.com/reemahussain-pharmd"

try:
    import qrcode as _qrcode
    _HAS_QR = True
except ImportError:
    _HAS_QR = False


def _make_qr_bytes(url: str):
    """Return QR code PNG as BytesIO, or None if qrcode is not installed."""
    if not _HAS_QR:
        return None
    qr = _qrcode.QRCode(version=1, box_size=4, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _safe(text: str) -> str:
    """Replace Unicode characters unsupported by Helvetica with ASCII equivalents."""
    return (
        str(text)
        .replace("–", "-")
        .replace("—", "--")
        .replace("→", "->")
        .replace("←", "<-")
        .replace("°", " deg")
        .replace("×", "x")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("é", "e")
        .replace("à", "a")
        .replace("∞", "inf")
        .replace("≥", ">=")
        .replace("≤", "<=")
        .replace("–", "-")
        .replace("—", "--")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("–",  "-")
        .replace("—",  "--")
        .replace("→",  "->")
        .replace("°",  " deg")
        .replace("×",  "x")
        .replace("’", "'")
        .replace("‘", "'")
    )


TEAL        = (29, 158, 117)
NAVY        = (13, 27, 42)
LIGHT_GRAY  = (245, 245, 245)
MID_GRAY    = (180, 180, 180)
WHITE       = (255, 255, 255)
RED_RISK    = (200, 50, 50)
AMBER_RISK  = (220, 150, 30)
ORANGE_RISK = (220, 100, 30)
GREEN_RISK  = (29, 158, 117)


def _risk_category_label(pct: int) -> str:
    """Map a dropout percentage to a 4-tier risk category label."""
    if pct >= 81:
        return "Critical"
    if pct >= 61:
        return "High"
    if pct >= 31:
        return "Moderate"
    return "Low"


def _risk_colour(category: str):
    return {
        "critical": RED_RISK,
        "high":     ORANGE_RISK,
        "moderate": AMBER_RISK,
        "low":      GREEN_RISK,
    }.get(category.lower(), AMBER_RISK)


def _shap_verbal(val: float) -> str:
    """Convert a SHAP value to a human-readable impact label."""
    a = abs(val)
    if a >= 0.4:
        return "Very High"
    if a >= 0.2:
        return "High"
    if a >= 0.1:
        return "Moderate"
    return "Low"


def _intervention_priority(iv: dict) -> str:
    """Derive a priority label from the intervention's estimated risk reduction."""
    reduction = iv.get("estimated_potential_risk_reduction", "").lower()
    if "high" in reduction:
        return "Critical"
    if "moderate" in reduction:
        return "High"
    return "Medium"


class RetentionReport(FPDF):
    """Custom FPDF subclass with branded header, footer, and helper layout methods."""

    def __init__(self, patient_id: str):
        super().__init__()
        self.patient_id = patient_id
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(15, 15, 15)

    def cell(self, w=0, h=0, text="", *args, **kwargs):
        return super().cell(w, h, _safe(str(text)), *args, **kwargs)

    def safe_multi_cell(self, w, h, text="", **kwargs):
        self.set_x(self.l_margin)
        if w == 0:
            w = self.epw
        return super().multi_cell(w, h, _safe(str(text)), **kwargs)

    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 18, "F")
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 12)
        self.set_xy(10, 4)
        self.cell(0, 10, "Clinical Trial Retention Intelligence Report", ln=False)
        self.set_font("Helvetica", "", 9)
        self.set_xy(10, 10)
        self.cell(
            0, 6,
            f"Participant: {self.patient_id}  |  Date: {date.today().strftime('%d %B %Y')}  |  CONFIDENTIAL",
        )
        self.ln(10)

    def footer(self):
        self.set_y(-30)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.4)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*MID_GRAY)
        self.multi_cell(0, 4, DISCLAIMER, align="C")
        self.set_font("Helvetica", "", 7)
        self.cell(0, 4, FOOTER_LINE2, ln=True, align="C")
        self.cell(0, 4, f"{FOOTER_LINE3}  |  {date.today().year}", align="C")

    def section_heading(self, text: str):
        self.set_fill_color(*TEAL)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 7, f"  {text}", ln=True, fill=True)
        self.ln(2)
        self.set_text_color(30, 30, 30)

    def kv_row(self, label: str, value: str, bold_value: bool = False):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(80, 80, 80)
        self.cell(65, 6, _safe(label) + ":", ln=False)
        if bold_value:
            self.set_font("Helvetica", "B", 9)
        self.set_text_color(20, 20, 20)
        self.cell(0, 6, _safe(value), ln=True)

    def colored_risk_badge(self, category: str, pct: int):
        r, g, b = _risk_colour(category)
        self.set_fill_color(r, g, b)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 14)
        self.cell(
            70, 14,
            f"{pct}%  {category.upper()}",
            border=1, ln=False, align="C", fill=True,
        )
        self.ln(16)
        self.set_text_color(20, 20, 20)

    def draw_risk_gauge(self, risk_pct: int):
        """Draw a 4-segment risk scale bar highlighting the current level."""
        current = _risk_category_label(risk_pct).upper()
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(80, 80, 80)
        self.cell(28, 6, "Risk Scale:", ln=False)
        segments = [
            ("LOW",      GREEN_RISK),
            ("MODERATE", AMBER_RISK),
            ("HIGH",     ORANGE_RISK),
            ("CRITICAL", RED_RISK),
        ]
        for label, color in segments:
            is_current = label == current
            if is_current:
                self.set_fill_color(*color)
                self.set_text_color(*WHITE)
                self.set_font("Helvetica", "B", 7)
                self.cell(37, 6, f"{label} ({risk_pct}%)", border=1, fill=True, ln=False, align="C")
            else:
                self.set_fill_color(240, 240, 240)
                self.set_text_color(160, 160, 160)
                self.set_font("Helvetica", "", 7)
                self.cell(37, 6, label, border=1, fill=True, ln=False, align="C")
        self.ln(9)
        self.set_text_color(20, 20, 20)

    def shap_table(self, factors, protective: bool = False):
        """Render a SHAP factor table with verbal impact labels."""
        col_w   = [8, 100, 38, 34]
        headers = ["#", "Driver", "Impact Level", "SHAP Value"]
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 8)
        for h, w in zip(headers, col_w):
            self.cell(w, 6, h, border=1, fill=True, ln=False, align="C")
        self.ln()

        self.set_text_color(20, 20, 20)
        for i, (feat, shap_val, label) in enumerate(factors, 1):
            fill = i % 2 == 0
            self.set_fill_color(*(LIGHT_GRAY if fill else WHITE))
            self.set_font("Helvetica", "", 8)
            self.cell(col_w[0], 6, str(i), border=1, fill=fill, ln=False, align="C")
            self.cell(col_w[1], 6, label, border=1, fill=fill, ln=False)
            verbal       = _shap_verbal(shap_val)
            impact_color = (
                RED_RISK   if verbal == "Very High" else
                AMBER_RISK if verbal == "High"      else
                TEAL       if protective            else
                ORANGE_RISK
            )
            self.set_text_color(*impact_color)
            self.set_font("Helvetica", "B", 8)
            self.cell(col_w[2], 6, verbal, border=1, fill=fill, ln=False, align="C")
            self.set_text_color(*TEAL if protective else RED_RISK)
            self.set_font("Helvetica", "", 8)
            prefix = "" if protective else "+"
            self.cell(col_w[3], 6, f"{prefix}{shap_val:.3f}", border=1, fill=fill, ln=False, align="C")
            self.ln()
            self.set_text_color(20, 20, 20)


def generate_report(
    analysis: Dict,
    patient_id: str = "DEMO",
    doc_source: str = "Manual Entry",
    copilot_summary=None,
    extraction_stats: dict = None,
) -> Path:
    """
    Generate an executive PDF retention intelligence report.

    Args:
        analysis: Full analysis dict returned by RetentionAgent.run().
        patient_id: Participant identifier for the report header.
        doc_source: Data source label (Manual Entry or Document Upload).
        copilot_summary: Optional CoordinatorSummary from CoordinatorCopilot.

    Returns:
        Path to the saved PDF file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    risk_pct       = analysis.get("risk_pct", 0)
    risk_cat_label = _risk_category_label(risk_pct)
    risk_factors   = analysis.get("top3_risk_factors", [])
    protective     = analysis.get("top3_protective_factors", [])
    interventions  = analysis.get("interventions", [])
    impact         = analysis.get("business_impact", {})
    evidence_list  = analysis.get("evidence", [])
    top_scenarios  = analysis.get("top_scenarios", [])

    pdf = RetentionReport(patient_id=patient_id)
    pdf.add_page()

    # ── PAGE 1 ───────────────────────────────────────────────────────────────

    # -- Author header --
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 8, "Dr. Reema Mohamed Sulthan", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "PharmD | Clinical Data Scientist | Certified AI Expert", ln=True)
    pdf.ln(3)

    # -- Metadata strip --
    strip_y = pdf.get_y()
    pdf.set_fill_color(*LIGHT_GRAY)
    pdf.set_draw_color(*MID_GRAY)
    pdf.set_line_width(0.3)
    pdf.rect(pdf.l_margin, strip_y, pdf.epw, 8, "FD")
    col_meta = pdf.epw / 4
    meta_items = [
        f"Report ID: CTRI-{patient_id}-001",
        "Platform Version: 3.0",
        f"Generated: {date.today().strftime('%d-%b-%Y')}",
        "Type: Participant Assessment",
    ]
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(80, 80, 80)
    pdf.set_xy(pdf.l_margin, strip_y + 1.5)
    for m in meta_items:
        pdf.cell(col_meta, 5, m, ln=False, align="C")
    pdf.ln(11)
    pdf.set_text_color(20, 20, 20)

    # -- Executive Summary --
    pdf.section_heading("Executive Summary")
    rf_names = [label for _, _, label in risk_factors[:2]] if risk_factors else []
    rf_text  = " and ".join(rf_names) if rf_names else "key clinical and logistical factors"
    if len(interventions) >= 2:
        iv_text = f"{interventions[0]['name']} and {interventions[1]['name']}"
    elif interventions:
        iv_text = interventions[0]["name"]
    else:
        iv_text = "targeted retention interventions"
    roi     = impact.get("roi_ratio", 0)
    roi_str = f"{roi:.1f}x" if roi not in (0, float("inf")) else "N/A"
    if risk_cat_label == "Low":
        exec_text = (
            f"Although {rf_text} contribute modestly to attrition risk, "
            f"multiple protective factors outweigh these, resulting in an overall low "
            f"retention risk profile for {patient_id} ({risk_pct}%). "
            f"No targeted interventions are recommended at this time. "
            f"Routine monitoring and standard site engagement are advised."
        )
    else:
        exec_text = (
            f"{patient_id} was classified as a {risk_cat_label} Retention Risk participant "
            f"({risk_pct}%). Primary dropout drivers are {rf_text}. "
            f"{iv_text} are recommended as priority interventions to reduce attrition risk. "
            f"Estimated intervention return on investment is {roi_str}."
        )
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)
    pdf.safe_multi_cell(0, 5, exec_text)
    pdf.ln(4)

    # -- Risk Assessment --
    pdf.section_heading("Participant Risk Assessment")
    pdf.colored_risk_badge(risk_cat_label, risk_pct)
    pdf.draw_risk_gauge(risk_pct)

    risk_score  = analysis.get("risk_score", risk_pct / 100)
    confidence  = int(round(max(risk_score, 1 - risk_score) * 100))
    conf_label  = "High" if confidence >= 80 else ("Moderate" if confidence >= 65 else "Low")

    pdf.kv_row("Participant ID",           patient_id)
    pdf.kv_row("Risk Category",            f"{risk_cat_label} ({risk_pct}%)", bold_value=True)
    pdf.kv_row("Prediction Confidence",    f"{conf_label} ({confidence}%)", bold_value=True)
    # Confidence explanation
    conf_explain = (
        f"{confidence}% confidence reflects calibrated probability output from the XGBoost classifier "
        f"(LogReg ensemble). Values above 80% indicate strong predictor signal alignment; "
        f"values between 60-80% indicate moderate agreement with clinically plausible uncertainty."
    )
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(100, 100, 100)
    pdf.safe_multi_cell(0, 4.5, conf_explain)
    pdf.set_text_color(20, 20, 20)
    pdf.ln(2)
    pdf.kv_row("Estimated Dropout Window", analysis.get("dropout_window", "—"))
    pdf.kv_row("Data Source",              doc_source)
    if doc_source != "Manual Entry":
        pdf.kv_row("Extraction Method",    "Rule-Based Clinical Parser")
        if extraction_stats:
            conf_pct    = extraction_stats.get("confidence_pct", 0)
            n_parsed    = extraction_stats.get("fields_parsed", 0)
            n_total     = extraction_stats.get("total_fields", 0)
            pdf.kv_row("Document Extraction Confidence", f"{conf_pct}%", bold_value=True)
            pdf.kv_row("Fields Successfully Parsed",     f"{n_parsed}/{n_total}")

    # Persona + characteristics
    persona_name = analysis.get("persona", "—")
    pdf.kv_row("Participant Persona",      persona_name)
    try:
        from personas import PERSONA_DESCRIPTIONS
        persona_desc = PERSONA_DESCRIPTIONS.get(persona_name, "")
    except Exception:
        persona_desc = analysis.get("persona_description", "")
    if persona_desc:
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(80, 80, 80)
        # Split on ". " to show as bullets
        for sentence in [s.strip() for s in persona_desc.split(". ") if s.strip()]:
            pdf.set_x(pdf.l_margin + 4)
            pdf.safe_multi_cell(pdf.epw - 4, 4.5, f"- {sentence.rstrip('.')}.")
        pdf.set_text_color(20, 20, 20)
    pdf.ln(3)

    # -- Risk Summary Box --
    def _risk_domain_label(pct: int) -> str:
        if pct >= 61: return "High"
        if pct >= 31: return "Moderate"
        return "Low"

    ae_pct    = risk_pct if any("side effect" in str(f).lower() or "adverse" in str(f).lower()
                                for f in risk_factors) else max(risk_pct - 15, 0)
    ops_pct   = risk_pct if any("visit" in str(f).lower() or "logistic" in str(f).lower()
                                or "protocol" in str(f).lower()
                                for f in risk_factors) else max(risk_pct - 20, 0)
    logi_pct  = risk_pct if any("transport" in str(f).lower() or "distance" in str(f).lower()
                                for f in risk_factors) else max(risk_pct - 20, 0)
    risk_box_rows = [
        ("Clinical Risk",     ae_pct),
        ("Operational Risk",  ops_pct),
        ("Logistical Risk",   logi_pct),
        ("Overall Retention Risk", risk_pct),
    ]
    pdf.set_draw_color(*MID_GRAY)
    pdf.set_line_width(0.3)
    box_x = pdf.l_margin
    box_w = pdf.epw
    box_row_h = 5.5
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.cell(box_w, 6, "Risk Domain Summary", border=0, fill=True, ln=True, align="C")
    for label, pct in risk_box_rows:
        domain_cat  = _risk_domain_label(pct)
        row_color   = _risk_colour(domain_cat)
        is_overall  = label.startswith("Overall")
        pdf.set_fill_color(*LIGHT_GRAY)
        pdf.set_text_color(20, 20, 20)
        pdf.set_font("Helvetica", "B" if is_overall else "", 7.5)
        pdf.cell(box_w * 0.55, box_row_h, f"  {label}", border=1, fill=True, ln=False)
        pdf.set_fill_color(*row_color)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.cell(box_w * 0.45, box_row_h,
                 f"{domain_cat} ({pct}%)",
                 border=1, fill=True, ln=True, align="C")
    pdf.ln(4)

    # Clinical timeline
    dropout_window = str(analysis.get("dropout_window", ""))
    pdf.section_heading("Estimated Clinical Risk Timeline")
    timeline_steps = [
        ("Week 0",  "Enrolment & Screening",                 TEAL),
        ("Week 2",  "AE Signal Window — pharmacovigilance contact recommended", AMBER_RISK),
        ("Week 4",  "Risk Escalation — logistical barriers compound",           ORANGE_RISK),
        (dropout_window or "Week 6", "Critical Retention Window — highest dropout probability", RED_RISK),
        ("Post-window", "Stabilisation if interventions deployed",              TEAL),
    ]
    for wk, desc, color in timeline_steps:
        r, g, b = color
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.cell(28, 5.5, _safe(wk), fill=True, ln=False, align="C")
        pdf.set_fill_color(*LIGHT_GRAY)
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 5.5, f"  {_safe(desc)}", fill=True, ln=True)
    pdf.ln(3)

    # -- SHAP Risk Factors --
    shap_heading = "Residual Risk Factors (SHAP Analysis)" if risk_cat_label == "Low" else "Top Dropout Risk Factors (SHAP Analysis)"
    pdf.section_heading(shap_heading)
    if risk_factors:
        pdf.shap_table(risk_factors, protective=False)
    pdf.ln(3)

    # -- Protective Factors --
    pdf.section_heading("Top Protective Factors")
    if protective:
        pdf.shap_table(protective, protective=True)
    pdf.ln(3)

    # -- Supporting Evidence --
    pdf.section_heading("Supporting Clinical Evidence")
    printed = 0
    for item in evidence_list:
        ev = item.get("evidence")
        if ev and printed < 2:
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(0, 5, f"Re: {item['intervention']}", ln=True)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.safe_multi_cell(0, 5, f"Source: {ev['source']}")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(20, 20, 20)
            pdf.safe_multi_cell(0, 5, ev["recommendation"])
            pdf.ln(2)
            printed += 1

    # ── PAGE 2 ───────────────────────────────────────────────────────────────
    pdf.add_page()

    # -- Interventions scorecard --
    pdf.section_heading("Intervention Scorecard")

    def _impact_score(reduction_text: str) -> str:
        t = reduction_text.lower()
        if "high" in t and "moderate" not in t and "low" not in t:
            return "9.5/10"
        if "moderate-high" in t or "high-moderate" in t:
            return "7.5/10"
        if "moderate" in t and "low" not in t:
            return "6.5/10"
        if "low-moderate" in t or "moderate-low" in t:
            return "5.0/10"
        return "3.5/10"

    def _cost_tier(cost: float) -> str:
        if cost == 0:       return "Zero"
        if cost <= 200:     return "Very Low"
        if cost <= 500:     return "Low"
        if cost <= 900:     return "Medium"
        return "High"

    col_w   = [44, 26, 26, 22, 28, 34]   # total = 180
    headers = ["Intervention", "Owner", "Impact", "Cost Tier", "Cost (USD)", "Priority"]

    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 7.5)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 7, h, border=1, fill=True, ln=False, align="C")
    pdf.ln()

    priority_color = {"Critical": RED_RISK, "High": AMBER_RISK, "Medium": TEAL}
    impact_color   = {"9.5/10": GREEN_RISK, "7.5/10": TEAL, "6.5/10": AMBER_RISK,
                      "5.0/10": ORANGE_RISK, "3.5/10": MID_GRAY}
    pdf.set_text_color(20, 20, 20)
    for i, iv in enumerate(interventions):
        fill      = i % 2 == 0
        bg        = LIGHT_GRAY if fill else WHITE
        pdf.set_fill_color(*bg)
        reduction  = iv["estimated_potential_risk_reduction"].replace(
            "Estimated Potential Risk Reduction: ", ""
        )
        iv_impact = _impact_score(reduction)
        cost_tier = _cost_tier(float(iv.get("cost", 0)))
        priority  = _intervention_priority(iv)

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(col_w[0], 6, _safe(iv["name"][:36]), border=1, fill=fill, ln=False)
        pdf.cell(col_w[1], 6, _safe(iv["owner"][:20]), border=1, fill=fill, ln=False)

        # Impact cell — coloured
        ir, ig, ib = impact_color.get(iv_impact, MID_GRAY)
        pdf.set_fill_color(ir, ig, ib)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(col_w[2], 6, iv_impact, border=1, fill=True, ln=False, align="C")
        pdf.set_fill_color(*bg)
        pdf.set_text_color(20, 20, 20)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(col_w[3], 6, cost_tier, border=1, fill=fill, ln=False, align="C")
        pdf.cell(col_w[4], 6, f"${float(iv.get('cost',0)):,.0f}", border=1, fill=fill, ln=False, align="C")

        pr, pg, pb = priority_color.get(priority, MID_GRAY)
        pdf.set_fill_color(pr, pg, pb)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(col_w[5], 6, priority, border=1, fill=True, ln=False, align="C")
        pdf.ln()
        pdf.set_text_color(20, 20, 20)
    pdf.ln(4)

    # -- PharmD Rationale --
    if interventions:
        pdf.section_heading("PharmD Rationale - Priority Intervention")
        top_iv = interventions[0]
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, top_iv["name"], ln=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(60, 60, 60)
        pdf.safe_multi_cell(0, 5, top_iv["pharmd_rationale"])
        pdf.set_text_color(20, 20, 20)
        pdf.ln(3)

    # -- Coordinator Copilot Summary (v3.0) --
    if copilot_summary is not None:
        pdf.section_heading("Retention Coordinator Recommendations (v3.0 Copilot)")
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(40, 40, 40)
        pdf.safe_multi_cell(0, 5, copilot_summary.risk_narrative)
        pdf.ln(3)
        if copilot_summary.action_items:
            priority_color = {
                "Critical": RED_RISK, "High": AMBER_RISK,
                "Medium": TEAL, "Low": MID_GRAY,
            }
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*NAVY)
            pdf.cell(0, 5, "Prioritised Action Plan:", ln=True)
            pdf.ln(1)
            for i, act in enumerate(copilot_summary.action_items, 1):
                pr, pg, pb = priority_color.get(act.priority, MID_GRAY)
                pdf.set_fill_color(pr, pg, pb)
                pdf.set_text_color(*WHITE)
                pdf.set_font("Helvetica", "B", 7.5)
                pdf.cell(22, 5.5, act.priority, border=0, fill=True, ln=False, align="C")
                pdf.set_fill_color(*LIGHT_GRAY)
                pdf.set_text_color(*NAVY)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(0, 5.5, f"  {i}. {act.title}", border=0, fill=True, ln=True)
                pdf.set_font("Helvetica", "", 7.5)
                pdf.set_text_color(60, 60, 60)
                pdf.set_x(pdf.l_margin + 24)
                pdf.multi_cell(pdf.epw - 24, 4.5, _safe(f"Timeline: {act.timeline}"), align="L")
                pdf.set_text_color(20, 20, 20)
                pdf.ln(1)
        if copilot_summary.expected_improvement_high > 0:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*TEAL)
            pdf.cell(
                0, 6,
                f"Expected retention improvement with full action plan: "
                f"{copilot_summary.expected_improvement_low}-{copilot_summary.expected_improvement_high} percentage points",
                ln=True,
            )
        pdf.set_text_color(20, 20, 20)
        pdf.ln(3)

    # -- What-If Scenarios (visual) --
    if top_scenarios:
        pdf.section_heading("What-If Scenario Analysis")
        for sc in top_scenarios:
            orig_pct  = int(round(sc.get("original_risk", 0) * 100))
            new_pct   = int(round(sc.get("new_risk", 0) * 100))
            reduction = sc.get("risk_reduction_pct", orig_pct - new_pct)

            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*NAVY)
            pdf.cell(0, 5, sc["label"], ln=True)

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*RED_RISK)
            pdf.cell(38, 6, f"Current: {orig_pct}%", ln=False, align="C")
            pdf.set_text_color(80, 80, 80)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(14, 6, "  ->  ", ln=False, align="C")
            pdf.set_text_color(*GREEN_RISK)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(32, 6, f"New: {new_pct}%", ln=False, align="C")
            pdf.set_text_color(*TEAL)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(0, 6, f"  Reduction: {reduction:.1f} percentage points", ln=True)

            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(60, 60, 60)
            pdf.safe_multi_cell(0, 5, sc["interpretation"])
            pdf.set_text_color(20, 20, 20)
            pdf.ln(2)

    # -- Business Impact --
    pdf.section_heading("Economic Impact Analysis")
    if risk_cat_label == "Low":
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*TEAL)
        pdf.cell(0, 6, "Intervention not recommended at current risk level.", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.safe_multi_cell(
            0, 5,
            f"Participant {patient_id} is within the acceptable retention threshold (risk < 30%). "
            "Economic impact modelling is not triggered. Continue routine monitoring and standard "
            "site engagement protocols."
        )
        pdf.ln(3)
    elif impact:
        col_a, col_b = 100, 0

        def impact_row(label, value, teal=False):
            if teal:
                pdf.set_text_color(*TEAL)
                pdf.set_font("Helvetica", "B", 9)
            else:
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(20, 20, 20)
            pdf.cell(col_a, 6, label, ln=False)
            pdf.cell(col_b, 6, value, ln=True)
            pdf.set_text_color(20, 20, 20)

        impact_row("Participant Replacement Cost (if dropped out):",
                   f"${impact.get('replacement_cost_avoided', 0):,.0f}")
        impact_row("Total Intervention Cost:",
                   f"${impact.get('intervention_total_cost', 0):,.0f}")
        impact_row("Estimated Net Savings (modelled):",
                   f"${impact.get('net_savings', 0):,.0f}", teal=True)
        roi     = impact.get("roi_ratio", 0)
        roi_str = f"{roi:.1f}x" if roi != float("inf") else "inf"
        impact_row("Return on Investment (modelled):", roi_str, teal=True)
        pdf.ln(3)

        # Assumptions
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Modelling Assumptions:", ln=True)
        pdf.set_font("Helvetica", "", 8)
        for a in [
            "Participant replacement cost estimated at $14,000-$22,000 per dropout (industry benchmark)",
            "Single-participant scenario; actual impact scales with cohort size",
            "Intervention effectiveness modelled based on published retention benchmarks",
            "All figures are educational estimates for portfolio demonstration only",
        ]:
            pdf.safe_multi_cell(0, 5, f"  - {a}")
        pdf.ln(4)

    # -- Technology Stack --
    pdf.section_heading("Technology Stack")
    tech_items = [
        ("Prediction Engine",    "XGBoost Gradient Boosting Classifier + Logistic Regression Ensemble"),
        ("Explainability",       "SHAP (SHapley Additive exPlanations) — feature-level attribution"),
        ("Clinical NER",         "Rule-Based Clinical Entity Recognition (Dictionary Pattern Matching)"),
        ("Document Intelligence","pdfplumber + PyMuPDF — multi-engine clinical CRF parser"),
        ("Data Processing",      "Scikit-learn pipeline — preprocessing, feature engineering, scaling"),
        ("Visualisation",        "Plotly, Matplotlib, Seaborn — interactive and static charts"),
        ("Report Generation",    "fpdf2 — programmatic PDF generation with branded clinical layout"),
        ("Application Layer",    "Streamlit Community Cloud — deployed web application"),
        ("Language",             "Python 3.11"),
    ]
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(30, 30, 30)
    for tech_label, tech_desc in tech_items:
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(52, 5.5, f"  {tech_label}:", border=0, ln=False)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 5.5, tech_desc, border=0, ln=True)
        pdf.set_text_color(30, 30, 30)
    pdf.ln(3)

    # -- References --
    pdf.section_heading("References")
    refs = [
        (
            "FDA Guidance for Industry: Patient Retention in Clinical Trials (2012). "
            "U.S. Food and Drug Administration."
        ),
        (
            "ICH E6(R2) Guideline for Good Clinical Practice (2016). "
            "International Council for Harmonisation."
        ),
        (
            "Gul, R.B. & Ali, P.A. (2010). Clinical trials: the challenge of recruitment "
            "and retention of participants. Journal of Clinical Nursing, 19(1-2), 227-233."
        ),
    ]
    pdf.set_text_color(40, 40, 40)
    for i, ref in enumerate(refs, 1):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(8, 5, f"{i}.", ln=False)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_x(pdf.l_margin + 8)
        super(RetentionReport, pdf).multi_cell(pdf.epw - 8, 5, _safe(ref))
        pdf.ln(1)

    # -- QR Code (GitHub) — bottom-right corner of page 2 --
    qr_bytes = _make_qr_bytes(GITHUB_URL)
    if qr_bytes:
        qr_size = 22
        qr_x    = 210 - pdf.r_margin - qr_size
        qr_y    = 297 - 38 - qr_size        # just above footer
        pdf.image(qr_bytes, x=qr_x, y=qr_y, w=qr_size, h=qr_size)
        pdf.set_xy(pdf.l_margin, qr_y + qr_size + 1)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*TEAL)
        pdf.cell(0, 4, "Project Repository:", ln=True)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(
            0, 4,
            "github.com/reemahussain-pharmd/AI-Powered-Clinical-Trial-Retention-Intelligence",
        )

    # Save
    out_path = REPORTS_DIR / f"retention_assessment_{patient_id}.pdf"
    pdf.output(str(out_path))
    return out_path


def generate_sample_report() -> Path:
    """Generate a sample PDF report using demo participant data."""
    sample_analysis = {
        "risk_score":     0.94,
        "risk_category":  "HIGH",
        "risk_pct":       94,
        "dropout_window": "Weeks 4-8 (Early Treatment Phase)",
        "persona":        "Long-Distance Rural Participant",
        "persona_description": (
            "Participant residing >80 km from site without transportation access."
        ),
        "top3_risk_factors": [
            ("side_effect_severity_at_week2", 0.805, "Week 2 Side Effect Severity"),
            ("logistic_friction_score",       0.507, "Visit Burden / Logistic Friction"),
            ("trial_phase",                   0.493, "Trial Phase Complexity"),
        ],
        "top3_protective_factors": [
            ("investigator_experience_years", -0.09, "Investigator Experience (Years)"),
            ("insurance_status_insured",      -0.05, "Insurance: Insured"),
            ("prior_trial_participation",     -0.04, "Prior Trial Participation"),
        ],
        "interventions": [
            {
                "name":  "Transportation Reimbursement Program",
                "owner": "Site Coordinator",
                "estimated_potential_risk_reduction": "Estimated Potential Risk Reduction: Moderate",
                "cost":  300.0,
                "pharmd_rationale": (
                    "Geographic barrier is a primary non-clinical dropout cause. "
                    "Reimbursing transport removes practical friction without protocol changes. "
                    "(Reference: FDA Guidance for Industry: Patient Retention in Clinical Trials, 2012)"
                ),
            },
            {
                "name":  "Dedicated Safety Monitoring Call (weekly)",
                "owner": "Principal Investigator + Study Nurse",
                "estimated_potential_risk_reduction": "Estimated Potential Risk Reduction: High",
                "cost":  150.0,
                "pharmd_rationale": (
                    "Early AE severity at week 2 is the strongest predictor of dropout in this analysis. "
                    "Proactive pharmacovigilance contact allows early AE management before discontinuation."
                ),
            },
            {
                "name":  "Protocol Complexity Counselling Session",
                "owner": "Study Coordinator",
                "estimated_potential_risk_reduction": "Estimated Potential Risk Reduction: Moderate",
                "cost":  100.0,
                "pharmd_rationale": (
                    "Phase III complexity is a known retention barrier. A structured counselling "
                    "session sets realistic expectations and improves informed consent quality."
                ),
            },
        ],
        "evidence": [
            {
                "intervention": "Transportation Reimbursement Program",
                "evidence": {
                    "source": "FDA (2012). Guidance for Industry: Patient Retention in Clinical Trials.",
                    "evidence": (
                        "Logistical barriers including transportation were identified as modifiable "
                        "factors affecting participant retention."
                    ),
                    "recommendation": (
                        "Trial sponsors should consider transportation reimbursement as a standard "
                        "retention strategy for participants residing more than 50 km from trial sites."
                    ),
                },
            },
            {
                "intervention": "Dedicated Safety Monitoring Call (weekly)",
                "evidence": {
                    "source": "ICH E6(R2) Good Clinical Practice Guidelines (2016).",
                    "evidence": (
                        "Proactive safety monitoring is mandated for early-phase trials "
                        "to protect participant welfare."
                    ),
                    "recommendation": (
                        "Weekly safety check-in calls should be implemented for participants "
                        "with emerging adverse events to prevent premature discontinuation."
                    ),
                },
            },
        ],
        "top_scenarios": [
            {
                "label":              "Add transportation reimbursement",
                "interpretation":     (
                    "Adding transportation support may reduce dropout risk from 94% to approximately "
                    "76% (estimated reduction: 18.2 percentage points)."
                ),
                "original_risk":      0.94,
                "new_risk":           0.76,
                "risk_reduction_pct": 18.2,
            },
            {
                "label":              "Add weekly safety monitoring call",
                "interpretation":     (
                    "Implementing weekly safety calls may reduce dropout risk from 94% to approximately "
                    "80% (estimated reduction: 14.0 percentage points)."
                ),
                "original_risk":      0.94,
                "new_risk":           0.80,
                "risk_reduction_pct": 14.0,
            },
        ],
        "business_impact": {
            "replacement_cost_avoided": 18000,
            "intervention_total_cost":  550,
            "net_savings":              17450,
            "roi_ratio":                31.7,
        },
    }
    return generate_report(sample_analysis, "DEMO_001")


if __name__ == "__main__":
    path = generate_sample_report()
    print(f"Sample report saved -> {path}")
