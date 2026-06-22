"""
PDF Clinical Report Generator — v3.0 Enterprise Layout.

Produces a professional 4-page PDF report for trial sponsors and clinical
operations teams. Uses fpdf2 for layout. All output is clearly labelled
as educational and portfolio demonstration material.
"""

import io
from fpdf import FPDF
from pathlib import Path
from datetime import date, datetime
from typing import Dict

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR  = PROJECT_ROOT / "reports"

GITHUB_URL = (
    "https://github.com/reemahussain-pharmd/"
    "AI-Powered-Clinical-Trial-Retention-Intelligence"
)

DISCLAIMER = (
    "For educational and portfolio demonstration purposes only. "
    "Not for clinical use or patient care decisions."
)

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
        .replace("–", "-").replace("—", "--")
        .replace("→", "->").replace("←", "<-")
        .replace("°", " deg").replace("×", "x")
        .replace("‘", "'").replace("’", "'")
        .replace("“", '"').replace("”", '"')
        .replace("é", "e").replace("à", "a")
        .replace("∞", "inf")
        .replace("≥", ">=").replace("≤", "<=")
        .replace("–", "-").replace("—", "--")
        .replace("→", "->").replace("°", " deg")
        .replace("×", "x").replace("'", "'").replace("'", "'")
    )


# ── Colour palette ────────────────────────────────────────────────────────────

TEAL        = (29, 158, 117)
NAVY        = (13, 27, 42)
LIGHT_GRAY  = (245, 245, 245)
MID_GRAY    = (180, 180, 180)
WHITE       = (255, 255, 255)
RED_RISK    = (200, 50, 50)
AMBER_RISK  = (220, 150, 30)
ORANGE_RISK = (220, 100, 30)
GREEN_RISK  = (29, 158, 117)

FOOTER_H    = 22          # footer zone height in mm
BREAK_GUARD = FOOTER_H + 6  # auto-page-break trigger margin


# ── Pure functions ────────────────────────────────────────────────────────────

def _risk_category_label(pct: int) -> str:
    if pct >= 81: return "Critical"
    if pct >= 61: return "High"
    if pct >= 31: return "Moderate"
    return "Low"


def _risk_colour(category: str):
    return {
        "critical": RED_RISK,
        "high":     ORANGE_RISK,
        "moderate": AMBER_RISK,
        "low":      GREEN_RISK,
    }.get(category.lower(), AMBER_RISK)


def _shap_verbal(val: float) -> str:
    a = abs(val)
    if a >= 0.4: return "Very High"
    if a >= 0.2: return "High"
    if a >= 0.1: return "Moderate"
    return "Low"


def _intervention_priority(iv: dict) -> str:
    reduction = iv.get("estimated_potential_risk_reduction", "").lower()
    if "high" in reduction: return "Critical"
    if "moderate" in reduction: return "High"
    return "Medium"


def _impact_label(reduction_text: str) -> str:
    t = reduction_text.lower()
    if "high" in t and "moderate" not in t and "low" not in t:
        return "Very High"
    if "moderate-high" in t or "high-moderate" in t:
        return "High"
    if "moderate" in t and "low" not in t:
        return "Moderate"
    if "low-moderate" in t or "moderate-low" in t:
        return "Low-Moderate"
    return "Low"


def _cost_tier(cost: float) -> str:
    if cost == 0:   return "Zero"
    if cost <= 200: return "Very Low"
    if cost <= 500: return "Low"
    if cost <= 900: return "Medium"
    return "High"


def _light(color):
    """Return a lightened version of a colour for value-card backgrounds."""
    return tuple(min(c + 50, 255) for c in color)


# ── PDF class ────────────────────────────────────────────────────────────────

class RetentionReport(FPDF):
    """Custom FPDF subclass — branded header, 3-column footer, layout helpers."""

    def __init__(self, patient_id: str):
        super().__init__()
        self.patient_id   = patient_id
        self.generated_at = datetime.now().strftime('%d-%b-%Y %H:%M')
        self.alias_nb_pages()
        self.set_auto_page_break(auto=True, margin=BREAK_GUARD)
        self.set_margins(15, 15, 15)

    # ── Core overrides ────────────────────────────────────────────────────────

    def cell(self, w=0, h=0, text="", *args, **kwargs):
        return super().cell(w, h, _safe(str(text)), *args, **kwargs)

    def safe_multi_cell(self, w, h, text="", **kwargs):
        self.set_x(self.l_margin)
        if w == 0:
            w = self.epw
        return super().multi_cell(w, h, _safe(str(text)), **kwargs)

    # ── Pagination helper ─────────────────────────────────────────────────────

    def check_page_space(self, needed: float):
        """Trigger a page break if fewer than `needed` mm remain above the footer."""
        remaining = self.h - self.get_y() - BREAK_GUARD
        if remaining < needed:
            self.add_page()

    # ── Header ────────────────────────────────────────────────────────────────

    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 17, "F")
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 11)
        self.set_xy(10, 3.5)
        self.cell(140, 7, "TrialGuard Clinical Intelligence Report", ln=False)
        self.set_font("Helvetica", "", 7.5)
        self.set_xy(10, 10)
        self.cell(
            0, 5,
            f"Participant: {self.patient_id}  |  {date.today().strftime('%d %B %Y')}  |  CONFIDENTIAL",
        )
        self.ln(8)

    # ── Footer ────────────────────────────────────────────────────────────────

    def footer(self):
        self.set_y(-FOOTER_H)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.4)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(1.5)
        self.set_font("Helvetica", "I", 6.5)
        self.set_text_color(*MID_GRAY)
        self.multi_cell(0, 3.5, DISCLAIMER, align="C")
        self.ln(1)
        col = self.epw / 3
        self.set_font("Helvetica", "", 7)
        self.set_x(self.l_margin)
        self.cell(col, 4, "TrialGuard  |  Clinical Trial Intelligence Platform", ln=False, align="L")
        self.cell(col, 4, f"Generated: {self.generated_at}", ln=False, align="C")
        self.cell(col, 4, f"Page {self.page_no()}/{{nb}}  |  Model v3.0", ln=False, align="R")

    # ── Section heading ───────────────────────────────────────────────────────

    def section_heading(self, text: str):
        self.set_fill_color(*TEAL)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9.5)
        self.cell(0, 6, f"  {text}", ln=True, fill=True)
        self.ln(1.5)
        self.set_text_color(30, 30, 30)

    # ── Key-value row ─────────────────────────────────────────────────────────

    def kv_row(self, label: str, value: str, bold_value: bool = False):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(80, 80, 80)
        self.cell(65, 5, _safe(label) + ":", ln=False)
        if bold_value:
            self.set_font("Helvetica", "B", 8)
        self.set_text_color(20, 20, 20)
        self.cell(0, 5, _safe(value), ln=True)

    # ── 4-card metric dashboard row ───────────────────────────────────────────

    def metric_cards(self, labels, values, colors):
        """Render a row of N equally-spaced metric cards."""
        n   = len(labels)
        cw  = self.epw / n
        cy  = self.get_y()
        # Header row
        for i, (lbl, col) in enumerate(zip(labels, colors)):
            self.set_fill_color(*col)
            self.set_text_color(*WHITE)
            self.set_font("Helvetica", "B", 6.5)
            self.set_xy(self.l_margin + i * cw, cy)
            self.cell(cw - 0.5, 5, lbl, fill=True, border=0, ln=False, align="C")
        self.ln(5)
        # Value row
        for i, (val, col) in enumerate(zip(values, colors)):
            self.set_fill_color(*_light(col))
            self.set_text_color(20, 20, 20)
            self.set_font("Helvetica", "B", 9)
            self.set_xy(self.l_margin + i * cw, self.get_y())
            self.cell(cw - 0.5, 9, _safe(val), border=1, fill=True, ln=False, align="C")
        self.ln(11)

    # ── Risk gauge ────────────────────────────────────────────────────────────

    def draw_risk_gauge(self, risk_pct: int):
        current = _risk_category_label(risk_pct).upper()
        self.set_font("Helvetica", "B", 7.5)
        self.set_text_color(80, 80, 80)
        label_w = 22
        self.cell(label_w, 5.5, "Risk Scale:", ln=False)
        seg_w = (self.epw - label_w) / 4
        for label, color in [
            ("LOW", GREEN_RISK), ("MODERATE", AMBER_RISK),
            ("HIGH", ORANGE_RISK), ("CRITICAL", RED_RISK),
        ]:
            is_cur = label == current
            self.set_fill_color(*(color if is_cur else (240, 240, 240)))
            self.set_text_color(*(WHITE if is_cur else (160, 160, 160)))
            self.set_font("Helvetica", "B" if is_cur else "", 6.5)
            txt = f"{label} ({risk_pct}%)" if is_cur else label
            self.cell(seg_w, 5.5, txt, border=1, fill=True, ln=False, align="C")
        self.ln(7)
        self.set_text_color(20, 20, 20)

    # ── SHAP table ────────────────────────────────────────────────────────────

    def shap_table(self, factors, protective: bool = False):
        col_w   = [8, 100, 36, 36]
        headers = ["#", "Driver", "Impact Level", "SHAP Value"]
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 7.5)
        for h, w in zip(headers, col_w):
            self.cell(w, 5.5, h, border=1, fill=True, ln=False, align="C")
        self.ln()
        self.set_text_color(20, 20, 20)
        for i, (feat, shap_val, label) in enumerate(factors, 1):
            fill = i % 2 == 0
            self.set_fill_color(*(LIGHT_GRAY if fill else WHITE))
            self.set_font("Helvetica", "", 7.5)
            self.cell(col_w[0], 5.5, str(i), border=1, fill=fill, ln=False, align="C")
            self.cell(col_w[1], 5.5, label, border=1, fill=fill, ln=False)
            verbal = _shap_verbal(shap_val)
            impact_color = (
                RED_RISK   if verbal == "Very High" else
                AMBER_RISK if verbal == "High"      else
                TEAL       if protective            else ORANGE_RISK
            )
            self.set_text_color(*impact_color)
            self.set_font("Helvetica", "B", 7.5)
            self.cell(col_w[2], 5.5, verbal, border=1, fill=fill, ln=False, align="C")
            self.set_text_color(*TEAL if protective else RED_RISK)
            self.set_font("Helvetica", "", 7.5)
            prefix = "" if protective else "+"
            self.cell(col_w[3], 5.5, f"{prefix}{shap_val:.3f}", border=1, fill=fill, ln=False, align="C")
            self.ln()
            self.set_text_color(20, 20, 20)


# ── Main report generator ─────────────────────────────────────────────────────

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
        extraction_stats: Optional dict with fields_parsed / total_fields / confidence_pct.

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

    risk_score     = analysis.get("risk_score", risk_pct / 100)
    confidence     = int(round(max(risk_score, 1 - risk_score) * 100))
    conf_label     = "High" if confidence >= 80 else ("Moderate" if confidence >= 65 else "Low")
    dropout_window = str(analysis.get("dropout_window", "—"))
    roi            = impact.get("roi_ratio", 0)
    roi_str        = f"{roi:.1f}x" if roi not in (0, float("inf")) else "N/A"

    rf_names = [label for _, _, label in risk_factors[:2]] if risk_factors else []
    rf_text  = " and ".join(rf_names) if rf_names else "key clinical and logistical factors"
    if len(interventions) >= 2:
        iv_text = f"{interventions[0]['name']} and {interventions[1]['name']}"
    elif interventions:
        iv_text = interventions[0]["name"]
    else:
        iv_text = "targeted retention interventions"

    pdf = RetentionReport(patient_id=patient_id)
    pdf.add_page()

    # ── Author + metadata strip ───────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 7, "Dr. Reema Mohamed Sulthan", ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 4, "PharmD | Clinical Data Scientist | Certified AI Expert", ln=True)
    pdf.ln(1.5)

    strip_y = pdf.get_y()
    pdf.set_fill_color(*LIGHT_GRAY)
    pdf.set_draw_color(*MID_GRAY)
    pdf.set_line_width(0.3)
    pdf.rect(pdf.l_margin, strip_y, pdf.epw, 6.5, "FD")
    meta_items = [
        f"Report ID: CTRI-{date.today().strftime('%Y%m%d')}-{patient_id.replace('_','').replace('-','')[:8].upper()}",
        "Platform Version: 3.0",
        f"Generated: {date.today().strftime('%d-%b-%Y')}",
        "Type: Participant Assessment",
    ]
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(80, 80, 80)
    pdf.set_xy(pdf.l_margin, strip_y + 1)
    col_meta = pdf.epw / 4
    for m in meta_items:
        pdf.cell(col_meta, 5, m, ln=False, align="C")
    pdf.ln(9)
    pdf.set_text_color(20, 20, 20)

    # ── Executive Summary ─────────────────────────────────────────────────────
    pdf.check_page_space(42)
    pdf.section_heading("Executive Summary")

    # Dashboard: 4 metric cards
    risk_col = _risk_colour(risk_cat_label)
    conf_col = GREEN_RISK if confidence >= 80 else (AMBER_RISK if confidence >= 65 else RED_RISK)
    pdf.metric_cards(
        labels=["Risk Score", "Risk Category", "Prediction Confidence", "Dropout Window"],
        values=[f"{risk_pct}%", risk_cat_label.upper(), f"{conf_label} ({confidence}%)", dropout_window],
        colors=[risk_col, risk_col, conf_col, NAVY],
    )

    # Narrative
    if risk_cat_label == "Low":
        exec_text = (
            f"Although {rf_text} contribute modestly to attrition risk, multiple protective "
            f"factors outweigh these, resulting in a low retention risk profile for {patient_id} "
            f"({risk_pct}%). No targeted interventions recommended. Routine monitoring advised."
        )
    else:
        exec_text = (
            f"{patient_id} was classified as a {risk_cat_label} Retention Risk participant "
            f"({risk_pct}%). Highest-ranked risk drivers identified by the model were {rf_text}. "
            f"{iv_text} are recommended as priority interventions to reduce attrition risk. "
            f"Estimated intervention return on investment is {roi_str}."
        )
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 30, 30)
    pdf.safe_multi_cell(0, 4.5, exec_text)
    pdf.ln(2)

    # ── Participant Risk Assessment ───────────────────────────────────────────
    pdf.check_page_space(52)
    pdf.section_heading("Participant Risk Assessment")

    # Risk badge (left column) + KV metadata (right column)
    badge_y = pdf.get_y()
    r, g, b = _risk_colour(risk_cat_label)
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(pdf.l_margin, badge_y)
    pdf.cell(62, 14, f"{risk_pct}%  {risk_cat_label.upper()}", border=1, fill=True, align="C", ln=False)

    kv_x = pdf.l_margin + 66
    kv_y = badge_y
    kv_items = [
        ("Participant ID",  patient_id,          False),
        ("Data Source",     doc_source,           False),
        ("Model Used",      "Logistic Regression (Primary Retention Model)", False),
        ("Explainability",  "SHAP Analysis",      False),
    ]
    if doc_source != "Manual Entry":
        kv_items.append(("Extraction Method", "Clinical Document Intelligence Engine", False))
        if extraction_stats:
            conf_pct = extraction_stats.get("confidence_pct", 0)
            n_parsed = extraction_stats.get("fields_parsed", 0)
            n_total  = extraction_stats.get("total_fields", 0)
            kv_items.append(("Extraction Confidence", f"{conf_pct}%", True))
            kv_items.append(("Fields Parsed",         f"{n_parsed}/{n_total}", False))

    for lbl, val, bold in kv_items:
        pdf.set_xy(kv_x, kv_y)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(52, 4.5, _safe(lbl) + ":", ln=False)
        pdf.set_font("Helvetica", "B" if bold else "", 7.5)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(pdf.epw - 52 - 66, 4.5, _safe(val), ln=False)
        kv_y += 4.5

    pdf.set_xy(pdf.l_margin, badge_y + 15)
    pdf.ln(1.5)

    # Risk gauge
    pdf.draw_risk_gauge(risk_pct)

    # Confidence note
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(100, 100, 100)
    pdf.safe_multi_cell(0, 3.5,
        "Prediction confidence reflects model certainty based on agreement across multiple "
        "clinical risk signals and calibration performance during validation.")
    pdf.set_text_color(20, 20, 20)
    pdf.ln(1.5)

    # Persona
    persona_name = analysis.get("persona", "—")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(65, 4.5, "Participant Persona:", ln=False)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 4.5, _safe(persona_name), ln=True)
    try:
        from personas import PERSONA_DESCRIPTIONS
        persona_desc = PERSONA_DESCRIPTIONS.get(persona_name, "")
    except Exception:
        persona_desc = analysis.get("persona_description", "")
    if persona_desc:
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(80, 80, 80)
        for sentence in [s.strip() for s in persona_desc.split(". ") if s.strip()]:
            pdf.set_x(pdf.l_margin + 4)
            pdf.safe_multi_cell(pdf.epw - 4, 4, f"- {sentence.rstrip('.')}.")
        pdf.set_text_color(20, 20, 20)
    pdf.ln(1.5)

    # ── Risk Domain Summary (4-card row) ──────────────────────────────────────
    def _risk_domain_label(pct: int) -> str:
        if pct >= 61: return "High"
        if pct >= 31: return "Moderate"
        return "Low"

    ae_pct   = risk_pct if any("side effect" in str(f).lower() or "adverse" in str(f).lower()
                               for f in risk_factors) else max(risk_pct - 15, 0)
    ops_pct  = risk_pct if any("visit" in str(f).lower() or "logistic" in str(f).lower()
                               or "protocol" in str(f).lower()
                               for f in risk_factors) else max(risk_pct - 20, 0)
    logi_pct = risk_pct if any("transport" in str(f).lower() or "distance" in str(f).lower()
                               for f in risk_factors) else max(risk_pct - 20, 0)
    dom_labels = ["Clinical Risk", "Operational Risk", "Logistical Risk", "Overall Risk"]
    dom_pcts   = [ae_pct, ops_pct, logi_pct, risk_pct]
    dom_cats   = [_risk_domain_label(p) for p in dom_pcts]
    dom_colors = [_risk_colour(c) for c in dom_cats]
    pdf.metric_cards(
        labels=dom_labels,
        values=[f"{cat} ({pct}%)" for cat, pct in zip(dom_cats, dom_pcts)],
        colors=dom_colors,
    )

    # ── Clinical Risk Timeline ────────────────────────────────────────────────
    pdf.check_page_space(36)
    pdf.section_heading("Estimated Clinical Risk Timeline")
    timeline_steps = [
        ("Week 0",               "Enrolment & Screening",                                   TEAL),
        ("Week 2",               "AE Signal Window -- pharmacovigilance contact recommended",AMBER_RISK),
        ("Week 4",               "Risk Escalation -- logistical barriers compound",          ORANGE_RISK),
        (dropout_window or "Week 6", "Critical Retention Window -- highest dropout probability", RED_RISK),
        ("Post-window",          "Stabilisation if interventions deployed",                  TEAL),
    ]
    for wk, desc, color in timeline_steps:
        pdf.set_fill_color(*color)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(34, 5, _safe(wk), fill=True, ln=False, align="C")
        pdf.set_fill_color(*LIGHT_GRAY)
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.cell(0, 5, f"  {_safe(desc)}", fill=True, ln=True)
    pdf.ln(2)

    # ── SHAP Risk Factors ─────────────────────────────────────────────────────
    pdf.check_page_space(36)
    shap_heading = (
        "Residual Risk Factors (SHAP Analysis)"
        if risk_cat_label == "Low"
        else "Top Dropout Risk Factors (SHAP Analysis)"
    )
    pdf.section_heading(shap_heading)
    if risk_factors:
        pdf.shap_table(risk_factors, protective=False)
    pdf.ln(2)

    # ── Protective Factors ────────────────────────────────────────────────────
    pdf.check_page_space(28)
    pdf.section_heading("Top Protective Factors")
    if protective:
        pdf.shap_table(protective, protective=True)
    pdf.ln(2)

    # ── Supporting Clinical Evidence (compact 3-column table) ─────────────────
    pdf.check_page_space(28)
    pdf.section_heading("Supporting Clinical Evidence")
    ev_w = [55, 68, 57]   # Domain | Source | Application  (total = 180)
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 7.5)
    for hdr, w in zip(["Domain / Intervention", "Source", "Application"], ev_w):
        pdf.cell(w, 5.5, hdr, border=1, fill=True, ln=False, align="C")
    pdf.ln()

    printed = 0
    for item in evidence_list:
        ev = item.get("evidence")
        if not ev or printed >= 3:
            continue
        fill   = printed % 2 == 0
        bg     = LIGHT_GRAY if fill else WHITE
        row_y  = pdf.get_y()

        # Domain
        pdf.set_xy(pdf.l_margin, row_y)
        pdf.set_fill_color(*bg)
        pdf.set_text_color(20, 20, 20)
        pdf.set_font("Helvetica", "B", 6.5)
        super(RetentionReport, pdf).multi_cell(
            ev_w[0], 4, _safe(item["intervention"][:50]), border=1, fill=fill)
        end_y = pdf.get_y()

        # Source
        pdf.set_xy(pdf.l_margin + ev_w[0], row_y)
        pdf.set_font("Helvetica", "I", 6.5)
        super(RetentionReport, pdf).multi_cell(
            ev_w[1], 4, _safe(ev.get("source", "")[:80]), border=1, fill=fill)
        end_y = max(end_y, pdf.get_y())

        # Application
        pdf.set_xy(pdf.l_margin + ev_w[0] + ev_w[1], row_y)
        pdf.set_font("Helvetica", "", 6.5)
        super(RetentionReport, pdf).multi_cell(
            ev_w[2], 4, _safe(ev.get("recommendation", "")[:110]), border=1, fill=fill)
        end_y = max(end_y, pdf.get_y())

        pdf.set_y(end_y)
        pdf.set_text_color(20, 20, 20)
        printed += 1
    pdf.ln(2)

    # ── Intervention Scorecard ────────────────────────────────────────────────
    pdf.check_page_space(28)
    pdf.section_heading("Intervention Scorecard")

    sc_col_w = [50, 28, 26, 22, 26, 28]   # total = 180
    sc_hdrs  = ["Intervention", "Owner", "Impact", "Cost Tier", "Cost (USD)", "Priority"]
    priority_color = {"Critical": RED_RISK, "High": AMBER_RISK, "Medium": TEAL}
    impact_color   = {
        "Very High": GREEN_RISK, "High": TEAL, "Moderate": AMBER_RISK,
        "Low-Moderate": ORANGE_RISK, "Low": MID_GRAY,
    }

    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 7)
    for hdr, w in zip(sc_hdrs, sc_col_w):
        pdf.cell(w, 6, hdr, border=1, fill=True, ln=False, align="C")
    pdf.ln()

    for i, iv in enumerate(interventions):
        fill     = i % 2 == 0
        bg       = LIGHT_GRAY if fill else WHITE
        reduction = iv["estimated_potential_risk_reduction"].replace(
            "Estimated Potential Risk Reduction: ", ""
        )
        iv_impact = _impact_label(reduction)
        cost_val  = float(iv.get("cost", 0))
        priority  = _intervention_priority(iv)

        pdf.set_fill_color(*bg)
        pdf.set_text_color(20, 20, 20)
        pdf.set_font("Helvetica", "", 6.5)
        pdf.cell(sc_col_w[0], 5.5, _safe(iv["name"][:42]), border=1, fill=fill, ln=False)
        pdf.cell(sc_col_w[1], 5.5, _safe(iv["owner"][:22]), border=1, fill=fill, ln=False)

        ir, ig, ib = impact_color.get(iv_impact, MID_GRAY)
        pdf.set_fill_color(ir, ig, ib)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 6.5)
        pdf.cell(sc_col_w[2], 5.5, iv_impact, border=1, fill=True, ln=False, align="C")

        pdf.set_fill_color(*bg)
        pdf.set_text_color(20, 20, 20)
        pdf.set_font("Helvetica", "", 6.5)
        pdf.cell(sc_col_w[3], 5.5, _cost_tier(cost_val), border=1, fill=fill, ln=False, align="C")
        pdf.cell(sc_col_w[4], 5.5, f"${cost_val:,.0f}", border=1, fill=fill, ln=False, align="C")

        pr, pg, pb = priority_color.get(priority, MID_GRAY)
        pdf.set_fill_color(pr, pg, pb)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 6.5)
        pdf.cell(sc_col_w[5], 5.5, priority, border=1, fill=True, ln=False, align="C")
        pdf.ln()
        pdf.set_text_color(20, 20, 20)
    pdf.ln(2)

    # ── PharmD Rationale ─────────────────────────────────────────────────────
    if interventions:
        pdf.check_page_space(20)
        pdf.section_heading("PharmD Rationale - Priority Intervention")
        top_iv = interventions[0]
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(0, 5, top_iv["name"], ln=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(60, 60, 60)
        pdf.safe_multi_cell(0, 4.5, top_iv["pharmd_rationale"])
        pdf.set_text_color(20, 20, 20)
        pdf.ln(2)

    # ── AI-Assisted Coordinator Summary ──────────────────────────────────────
    if copilot_summary is not None:
        pdf.check_page_space(30)
        pdf.section_heading("AI-Assisted Coordinator Summary")
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(40, 40, 40)
        pdf.safe_multi_cell(0, 4.5, copilot_summary.risk_narrative)
        pdf.ln(1.5)
        if copilot_summary.action_items:
            _pc = {"Critical": RED_RISK, "High": AMBER_RISK, "Medium": TEAL, "Low": MID_GRAY}
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(*NAVY)
            pdf.cell(0, 4.5, "Prioritised Action Plan:", ln=True)
            pdf.ln(0.5)
            for i, act in enumerate(copilot_summary.action_items, 1):
                pr, pg, pb = _pc.get(act.priority, MID_GRAY)
                pdf.set_fill_color(pr, pg, pb)
                pdf.set_text_color(*WHITE)
                pdf.set_font("Helvetica", "B", 7)
                pdf.cell(20, 5, act.priority, border=0, fill=True, ln=False, align="C")
                pdf.set_fill_color(*LIGHT_GRAY)
                pdf.set_text_color(*NAVY)
                pdf.set_font("Helvetica", "B", 7.5)
                pdf.cell(0, 5,
                         f"  {i}. {act.title}  -  Timeline: {act.timeline}",
                         border=0, fill=True, ln=True)
                pdf.set_text_color(20, 20, 20)
                pdf.ln(0.5)
        if copilot_summary.expected_improvement_high > 0:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*TEAL)
            pdf.cell(
                0, 5,
                f"Expected retention improvement: "
                f"{copilot_summary.expected_improvement_low}"
                f"-{copilot_summary.expected_improvement_high} percentage points",
                ln=True,
            )
        pdf.set_text_color(20, 20, 20)
        pdf.ln(2)

    # ── What-If Scenario Analysis ─────────────────────────────────────────────
    if top_scenarios:
        pdf.check_page_space(32)
        pdf.section_heading("What-If Scenario Analysis")
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(100, 100, 100)
        pdf.safe_multi_cell(0, 4,
            "Note: Individual intervention effects are modelled estimates and are not additive.")
        pdf.set_text_color(20, 20, 20)
        pdf.ln(1.5)

        if len(top_scenarios) == 2:
            # Side-by-side cards
            sc_w = (pdf.epw - 4) / 2
            sc_y = pdf.get_y()
            for si, sc in enumerate(top_scenarios):
                orig_pct  = int(round(sc.get("original_risk", 0) * 100))
                new_pct   = int(round(sc.get("new_risk", 0) * 100))
                reduction = sc.get("risk_reduction_pct", orig_pct - new_pct)
                sx = pdf.l_margin + si * (sc_w + 4)

                pdf.set_xy(sx, sc_y)
                pdf.set_fill_color(*NAVY)
                pdf.set_text_color(*WHITE)
                pdf.set_font("Helvetica", "B", 7)
                pdf.cell(sc_w, 5, _safe(sc["label"]), fill=True, border=0, ln=False, align="C")

                pdf.set_xy(sx, sc_y + 5)
                vw = sc_w / 3
                pdf.set_fill_color(*LIGHT_GRAY)
                pdf.set_text_color(*RED_RISK)
                pdf.set_font("Helvetica", "B", 8.5)
                pdf.cell(vw, 7, f"{orig_pct}%", border=1, fill=True, ln=False, align="C")
                pdf.set_text_color(80, 80, 80)
                pdf.set_font("Helvetica", "", 7)
                pdf.cell(vw * 0.6, 7, "->", border=0, fill=True, ln=False, align="C")
                pdf.set_text_color(*GREEN_RISK)
                pdf.set_font("Helvetica", "B", 8.5)
                pdf.cell(vw, 7, f"{new_pct}%", border=1, fill=True, ln=False, align="C")
                pdf.set_text_color(*TEAL)
                pdf.set_font("Helvetica", "B", 7)
                pdf.cell(sc_w - vw * 2 - vw * 0.6, 7,
                         f"-{reduction:.1f}pp", border=1, fill=True, ln=False, align="C")

                pdf.set_xy(sx, sc_y + 13)
                pdf.set_text_color(60, 60, 60)
                pdf.set_font("Helvetica", "I", 6.5)
                super(RetentionReport, pdf).multi_cell(
                    sc_w, 3.8, _safe(sc["interpretation"][:130]))

            pdf.set_y(sc_y + 30)
        else:
            for sc in top_scenarios:
                orig_pct  = int(round(sc.get("original_risk", 0) * 100))
                new_pct   = int(round(sc.get("new_risk", 0) * 100))
                reduction = sc.get("risk_reduction_pct", orig_pct - new_pct)
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(*NAVY)
                pdf.cell(0, 5, sc["label"], ln=True)
                pdf.set_font("Helvetica", "B", 8.5)
                pdf.set_text_color(*RED_RISK)
                pdf.cell(36, 5.5, f"Current: {orig_pct}%", ln=False, align="C")
                pdf.set_text_color(80, 80, 80)
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(12, 5.5, "->", ln=False, align="C")
                pdf.set_text_color(*GREEN_RISK)
                pdf.set_font("Helvetica", "B", 8.5)
                pdf.cell(28, 5.5, f"New: {new_pct}%", ln=False, align="C")
                pdf.set_text_color(*TEAL)
                pdf.set_font("Helvetica", "", 7.5)
                pdf.cell(0, 5.5, f"  Reduction: {reduction:.1f} pp", ln=True)
                pdf.set_font("Helvetica", "I", 7.5)
                pdf.set_text_color(60, 60, 60)
                pdf.safe_multi_cell(0, 4.5, sc["interpretation"])
                pdf.set_text_color(20, 20, 20)
                pdf.ln(1.5)
        pdf.ln(1)

    # ── Economic Impact Analysis ──────────────────────────────────────────────
    pdf.check_page_space(38)
    pdf.section_heading("Economic Impact Analysis")
    if risk_cat_label == "Low":
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*TEAL)
        pdf.cell(0, 5.5, "Intervention not recommended at current risk level.", ln=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(60, 60, 60)
        pdf.safe_multi_cell(
            0, 4.5,
            f"Participant {patient_id} is within the acceptable retention threshold (risk < 30%). "
            "Continue routine monitoring and standard site engagement protocols."
        )
        pdf.ln(2)
    elif impact:
        pdf.metric_cards(
            labels=["Replacement Cost", "Intervention Cost", "Net Savings (Modelled)", "ROI (Modelled)"],
            values=[
                f"${impact.get('replacement_cost_avoided', 0):,.0f}",
                f"${impact.get('intervention_total_cost', 0):,.0f}",
                f"${impact.get('net_savings', 0):,.0f}",
                roi_str,
            ],
            colors=[RED_RISK, AMBER_RISK, GREEN_RISK, TEAL],
        )
        # Assumptions (compact)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 4.5, "Modelling Assumptions:", ln=True)
        pdf.set_font("Helvetica", "", 7.5)
        for a in [
            "Replacement cost: $14,000-$22,000 per dropout (industry benchmark)",
            "Single-participant scenario; scales with cohort size",
            "Intervention effectiveness modelled from published retention benchmarks",
            "All figures are educational estimates for portfolio demonstration only",
        ]:
            pdf.safe_multi_cell(0, 4, f"  - {a}")
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(80, 80, 80)
        pdf.safe_multi_cell(0, 4, "Source: Getz KA et al., Ther Innov Regul Sci (2016).")
        pdf.set_text_color(20, 20, 20)
        pdf.ln(2)

    # ── Technology Stack (2-column layout) ────────────────────────────────────
    pdf.check_page_space(34)
    pdf.section_heading("Technology Stack")

    left_tech = [
        ("Prediction Engine",         "XGBoost + Logistic Regression Ensemble"),
        ("Explainability",            "SHAP -- feature-level attribution"),
        ("Clinical Info. Extraction", "Rule-Based Entity Extraction Pipeline"),
        ("Report Generation",         "fpdf2 -- branded clinical PDF layout"),
    ]
    right_tech = [
        ("Document Intelligence",     "pdfplumber + PyMuPDF -- CRF parser"),
        ("Data Processing",           "Scikit-learn -- preprocessing & scaling"),
        ("Visualisation",             "Plotly, Matplotlib -- interactive charts"),
        ("Application Layer",         "Streamlit Community Cloud"),
        ("Language",                  "Python 3.11"),
    ]
    col_half = pdf.epw / 2 - 2
    lbl_w    = 44
    val_w    = col_half - lbl_w
    row_h    = 4.5
    ts_y     = pdf.get_y()

    for row_i in range(max(len(left_tech), len(right_tech))):
        ry = ts_y + row_i * row_h
        if row_i < len(left_tech):
            lbl, val = left_tech[row_i]
            pdf.set_xy(pdf.l_margin, ry)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(lbl_w, row_h, f"  {lbl}:", ln=False)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(val_w, row_h, _safe(val), ln=False)
        if row_i < len(right_tech):
            lbl, val = right_tech[row_i]
            pdf.set_xy(pdf.l_margin + col_half + 4, ry)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(lbl_w, row_h, f"  {lbl}:", ln=False)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(val_w, row_h, _safe(val), ln=False)

    pdf.set_y(ts_y + max(len(left_tech), len(right_tech)) * row_h)
    pdf.set_text_color(20, 20, 20)
    pdf.ln(2)

    # ── References ────────────────────────────────────────────────────────────
    pdf.check_page_space(20)
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
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.cell(7, 4.5, f"{i}.", ln=False)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_x(pdf.l_margin + 7)
        super(RetentionReport, pdf).multi_cell(pdf.epw - 7, 4.5, _safe(ref))
        pdf.ln(0.5)
    pdf.ln(1)

    # ── QR Code + Repository link ─────────────────────────────────────────────
    qr_bytes  = _make_qr_bytes(GITHUB_URL)
    remaining = pdf.h - pdf.get_y() - BREAK_GUARD
    qr_size   = 20
    if qr_bytes and remaining >= qr_size + 6:
        cur_y = pdf.get_y() + 1
        pdf.image(qr_bytes, x=pdf.l_margin, y=cur_y, w=qr_size, h=qr_size)
        pdf.set_xy(pdf.l_margin + qr_size + 4, cur_y + 3)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(*TEAL)
        pdf.cell(0, 4.5, "Project Repository:", ln=True)
        pdf.set_xy(pdf.l_margin + qr_size + 4, pdf.get_y())
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(
            0, 4.5,
            "github.com/reemahussain-pharmd/AI-Powered-Clinical-Trial-Retention-Intelligence",
            link=GITHUB_URL,
        )
    elif qr_bytes:
        pdf.add_page()
        cur_y = pdf.get_y()
        pdf.image(qr_bytes, x=pdf.l_margin, y=cur_y, w=qr_size, h=qr_size)
        pdf.set_xy(pdf.l_margin + qr_size + 4, cur_y + 3)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*TEAL)
        pdf.cell(0, 5, "Project Repository:", ln=True)
        pdf.set_xy(pdf.l_margin + qr_size + 4, pdf.get_y())
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(
            0, 5,
            "github.com/reemahussain-pharmd/AI-Powered-Clinical-Trial-Retention-Intelligence",
            link=GITHUB_URL,
        )

    # ── Save ──────────────────────────────────────────────────────────────────
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
