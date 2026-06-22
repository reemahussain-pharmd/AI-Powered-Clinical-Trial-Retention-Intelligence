"""
Clinical Document Intake & Auto-Population Module — Version 2.0

Rule-based, deterministic extraction of participant data from clinical PDFs.
No LLMs. No external APIs. Human-in-the-loop validation before prediction.
"""

import io
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")


# ── Field result container ────────────────────────────────────────────────────

@dataclass
class FieldResult:
    value: Any
    confidence: str          # "High" | "Medium" | "Low"
    raw_match: str = ""
    method: str = ""
    is_fallback: bool = False

    @property
    def confidence_pct(self) -> int:
        """Numeric confidence score for display (0-99)."""
        if self.is_fallback:
            return 0
        if self.confidence == "High":
            return 95
        if self.confidence == "Medium":
            return 72
        return 45


# ── Defaults and labels ───────────────────────────────────────────────────────

FIELD_DEFAULTS: Dict[str, Any] = {
    "age":                          45,
    "gender":                       "M",
    "bmi":                          24.0,
    "disease_severity_score":       5.0,
    "number_of_comorbidities":      2,
    "concomitant_medications":      4,
    "distance_from_site_km":        30,
    "visit_frequency_per_month":    4,
    "side_effect_severity_at_week2": 2.0,
    "insurance_status":             "insured",
    "transportation_access":        "yes",
    "prior_trial_participation":    0,
    "trial_phase":                  2,
    "consent_complexity_score":     5,
    "visit_burden_index":           4,
    "logistic_friction_score":      3,
}

FIELD_LABELS: Dict[str, str] = {
    "age":                          "Participant Age",
    "gender":                       "Gender",
    "bmi":                          "Body Mass Index (BMI)",
    "disease_severity_score":       "Disease Severity Score",
    "number_of_comorbidities":      "Number of Comorbidities",
    "concomitant_medications":      "Concomitant Medications",
    "distance_from_site_km":        "Distance from Trial Site (km)",
    "visit_frequency_per_month":    "Visit Frequency per Month",
    "side_effect_severity_at_week2": "Side Effect Severity at Week 2",
    "insurance_status":             "Insurance Status",
    "transportation_access":        "Transportation Access",
    "prior_trial_participation":    "Prior Trial Participation",
    "trial_phase":                  "Trial Phase",
    "consent_complexity_score":     "Consent Complexity Score",
    "visit_burden_index":           "Visit Burden Index",
    "logistic_friction_score":      "Logistic Friction Score",
}

EXTRACTION_ORDER = list(FIELD_DEFAULTS.keys())


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, str]:
    """
    Extract text from PDF bytes.
    Tries pdfplumber first (table-aware), then PyMuPDF as fallback.
    Returns (extracted_text, method_name).
    """
    # pdfplumber — better for structured forms
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            text = "\n".join(pages)
        if text.strip():
            return text, "pdfplumber"
    except Exception:
        pass

    # PyMuPDF fallback
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [doc[i].get_text() for i in range(len(doc))]
        doc.close()
        text = "\n".join(pages)
        if text.strip():
            return text, "PyMuPDF"
    except Exception:
        pass

    return "", "none"


# ── Clinical document parser ──────────────────────────────────────────────────

class ClinicalDocumentParser:
    """
    Rule-based parser for clinical trial participant documents.

    Each field extractor tries patterns in confidence order (High → Medium → Low).
    Returns a FieldResult with confidence level and the raw match for audit trail.
    """

    def parse(self, text: str) -> Dict[str, FieldResult]:
        tl = text.lower()
        extractors = [
            ("age",                          self._age),
            ("gender",                       self._gender),
            ("bmi",                          self._bmi),
            ("disease_severity_score",       self._disease_severity),
            ("number_of_comorbidities",      self._comorbidities),
            ("concomitant_medications",      self._medications),
            ("distance_from_site_km",        self._distance),
            ("visit_frequency_per_month",    self._visit_frequency),
            ("side_effect_severity_at_week2", self._side_effects),
            ("insurance_status",             self._insurance),
            ("transportation_access",        self._transportation),
            ("prior_trial_participation",    self._prior_trial),
            ("trial_phase",                  self._trial_phase),
            ("consent_complexity_score",     self._consent_complexity),
            ("visit_burden_index",           self._visit_burden),
            ("logistic_friction_score",      self._logistic_friction),
        ]
        results = {}
        for key, fn in extractors:
            try:
                results[key] = fn(text, tl)
            except Exception:
                results[key] = FieldResult(
                    value=FIELD_DEFAULTS[key],
                    confidence="Low",
                    method="error_fallback",
                    is_fallback=True,
                )
        return results

    # ── Individual extractors ─────────────────────────────────────────────────

    def _age(self, text: str, tl: str) -> FieldResult:
        high = [
            r"(?:age|participant\s+age|age\s+at\s+screening)\s*[:\-=]\s*(\d{1,3})\s*(?:years?|yrs?)?",
            r"(\d{1,3})\s*-?\s*year\s*-?\s*old",
            r"(\d{1,3})\s*(?:y\.?o\.?|yo)\b",
        ]
        medium = [
            r"aged?\s+(\d{2,3})\b",
            r"age\s+of\s+(\d{2,3})",
        ]
        for conf, patterns in [("High", high), ("Medium", medium)]:
            for p in patterns:
                m = re.search(p, tl, re.IGNORECASE)
                if m:
                    try:
                        val = int(m.group(1))
                        if 1 < val < 120:
                            return FieldResult(val, conf, m.group(0), "regex")
                    except (IndexError, ValueError):
                        pass
        # DOB calculation
        m = re.search(r"(?:dob|date\s+of\s+birth)\s*[:\-]\s*(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})", tl)
        if m:
            try:
                year = int(m.group(3))
                age = datetime.now().year - year
                if 1 < age < 120:
                    return FieldResult(age, "Medium", m.group(0), "dob_calc")
            except (ValueError, IndexError):
                pass
        return FieldResult(FIELD_DEFAULTS["age"], "Low", "", "fallback", True)

    def _gender(self, text: str, tl: str) -> FieldResult:
        gmap = {"male": "M", "m": "M", "female": "F", "f": "F", "other": "Other"}
        high = [
            r"(?:gender|sex)\s*[:\-=]\s*(male|female|other|m|f)\b",
            r"sex\s*[:\-=]\s*([mf])\b",
        ]
        medium = [
            r"(?:participant|patient)\s+is\s+a?\s*(male|female)",
            r"\b(male|female)\b",
        ]
        for conf, patterns in [("High", high), ("Medium", medium)]:
            for p in patterns:
                m = re.search(p, tl, re.IGNORECASE)
                if m:
                    raw = m.group(1).lower().strip()
                    val = gmap.get(raw)
                    if val:
                        return FieldResult(val, conf, m.group(0), "regex")
        return FieldResult(FIELD_DEFAULTS["gender"], "Low", "", "fallback", True)

    def _bmi(self, text: str, tl: str) -> FieldResult:
        high = [
            r"(?:bmi|body\s*mass\s*index)\s*[:\-=]\s*(\d{1,2}(?:\.\d{1,2})?)",
            r"bmi\s+of\s+(\d{1,2}(?:\.\d{1,2})?)",
            r"(\d{1,2}\.\d)\s*kg\s*/\s*m[²2]",
        ]
        medium = [r"mass\s+index[:\s]+(\d{1,2}(?:\.\d)?)"]
        for conf, patterns in [("High", high), ("Medium", medium)]:
            for p in patterns:
                m = re.search(p, tl, re.IGNORECASE)
                if m:
                    try:
                        val = float(m.group(1))
                        if 10 < val < 70:
                            return FieldResult(round(val, 1), conf, m.group(0), "regex")
                    except (ValueError, IndexError):
                        pass
        return FieldResult(FIELD_DEFAULTS["bmi"], "Low", "", "fallback", True)

    def _disease_severity(self, text: str, tl: str) -> FieldResult:
        high = [
            r"(?:disease\s+severity\s+score|dss|severity\s+score)\s*[:\-=]\s*(\d{1,2}(?:\.\d)?)",
            r"disease\s+severity\s*[:\-=]\s*(\d{1,2}(?:\.\d)?)\s*(?:/\s*10)?",
        ]
        medium = [
            r"severity\s*[:\-=]\s*(\d{1,2}(?:\.\d)?)\s*(?:/\s*10)?",
        ]
        for conf, patterns in [("High", high), ("Medium", medium)]:
            for p in patterns:
                m = re.search(p, tl, re.IGNORECASE)
                if m:
                    try:
                        val = float(m.group(1))
                        if 0 <= val <= 10:
                            return FieldResult(round(val, 1), conf, m.group(0), "regex")
                    except (ValueError, IndexError):
                        pass
        # Severity word mapping
        word_map = {"mild": 3.0, "moderate": 5.5, "moderate-severe": 7.0,
                    "severe": 8.5, "critical": 9.5}
        for word, score in word_map.items():
            if re.search(r"\b" + word.replace("-", r"[\s\-]") + r"\b", tl):
                return FieldResult(score, "Medium", word, "severity_word")
        return FieldResult(FIELD_DEFAULTS["disease_severity_score"], "Low", "", "fallback", True)

    def _comorbidities(self, text: str, tl: str) -> FieldResult:
        high = [
            r"(?:number\s+of\s+comorbidities|comorbidity\s+count)\s*[:\-=]\s*(\d+)",
            r"(\d+)\s+comorbidities",
            r"comorbidities\s*[:\-=]\s*(\d+)",
        ]
        medium = [r"(\d+)\s+co[\s-]?morbid"]
        for conf, patterns in [("High", high), ("Medium", medium)]:
            for p in patterns:
                m = re.search(p, tl, re.IGNORECASE)
                if m:
                    try:
                        val = int(m.group(1))
                        if 0 <= val <= 15:
                            return FieldResult(val, conf, m.group(0), "regex")
                    except (ValueError, IndexError):
                        pass
        # Count distinct diagnoses listed
        diag_hits = re.findall(
            r"(?:diagnosis|diagnosed\s+with|history\s+of)\s+[A-Za-z]", tl
        )
        if len(diag_hits) >= 2:
            val = min(len(diag_hits), 8)
            return FieldResult(val, "Low", f"{val} diagnoses inferred", "diagnosis_count")
        return FieldResult(FIELD_DEFAULTS["number_of_comorbidities"], "Low", "", "fallback", True)

    def _medications(self, text: str, tl: str) -> FieldResult:
        high = [
            r"(?:concomitant\s+medications?|concom\.?\s*meds?|concurrent\s+medications?|medication\s+count|total\s+medications?)\s*[:\-=]\s*(\d+)",
            r"(\d+)\s+concomitant\s+medications?",
        ]
        medium = [
            r"(?:currently\s+taking|on)\s+(\d+)\s+medications?",
            r"(\d+)\s+medications?\b",
        ]
        for conf, patterns in [("High", high), ("Medium", medium)]:
            for p in patterns:
                m = re.search(p, tl, re.IGNORECASE)
                if m:
                    try:
                        val = int(m.group(1))
                        if 0 <= val <= 25:
                            return FieldResult(val, conf, m.group(0), "regex")
                    except (ValueError, IndexError):
                        pass
        return FieldResult(FIELD_DEFAULTS["concomitant_medications"], "Low", "", "fallback", True)

    def _distance(self, text: str, tl: str) -> FieldResult:
        high = [
            r"(?:distance\s+from\s+(?:trial\s+)?site|travel\s+distance|distance\s+to\s+site)\s*[:\-=]\s*(\d+(?:\.\d+)?)\s*km",
            r"(\d+(?:\.\d+)?)\s*km\s+from\s+(?:the\s+)?site",
            r"distance\s*[:\-=]\s*(\d+(?:\.\d+)?)\s*km",
        ]
        medium = [
            r"(\d+(?:\.\d+)?)\s*km\b",
            r"(?:lives?|resides?)\s+(\d+)\s*km",
        ]
        # miles patterns
        miles_patterns = [
            r"distance\s*[:\-=]\s*(\d+(?:\.\d+)?)\s*miles?",
            r"(\d+(?:\.\d+)?)\s*miles?\s+from\s+site",
        ]
        for p in miles_patterns:
            m = re.search(p, tl, re.IGNORECASE)
            if m:
                try:
                    val = round(float(m.group(1)) * 1.609, 0)
                    if 0 < val <= 600:
                        return FieldResult(int(val), "High", m.group(0), "miles_converted")
                except (ValueError, IndexError):
                    pass
        for conf, patterns in [("High", high), ("Medium", medium)]:
            for p in patterns:
                m = re.search(p, tl, re.IGNORECASE)
                if m:
                    try:
                        val = float(m.group(1))
                        if 0 < val <= 600:
                            return FieldResult(int(val), conf, m.group(0), "regex")
                    except (ValueError, IndexError):
                        pass
        return FieldResult(FIELD_DEFAULTS["distance_from_site_km"], "Low", "", "fallback", True)

    def _visit_frequency(self, text: str, tl: str) -> FieldResult:
        high = [
            r"(?:visit\s+frequency(?:\s+per\s+month)?|visits?\s+per\s+month|monthly\s+visits?)\s*[:\-=]\s*(\d+)",
            r"(\d+)\s+visits?\s*(?:per|/)\s*month",
        ]
        medium = [
            r"visit\s+(?:schedule|freq(?:uency)?)\s*[:\-=]\s*(\d+)",
            r"(\d+)\s+monthly\s+visits?",
        ]
        for conf, patterns in [("High", high), ("Medium", medium)]:
            for p in patterns:
                m = re.search(p, tl, re.IGNORECASE)
                if m:
                    try:
                        val = int(m.group(1))
                        if 0 < val <= 20:
                            return FieldResult(val, conf, m.group(0), "regex")
                    except (ValueError, IndexError):
                        pass
        return FieldResult(FIELD_DEFAULTS["visit_frequency_per_month"], "Low", "", "fallback", True)

    def _side_effects(self, text: str, tl: str) -> FieldResult:
        high = [
            r"(?:side\s*effect\s*severity\s*(?:at\s*)?week\s*2|week\s*2\s*(?:ae|adverse\s*event|side\s*effect)\s*severity|ae\s*severity\s*week\s*2)\s*[:\-=]\s*(\d(?:\.\d)?)",
            r"week\s*2\s*severity\s*[:\-=]\s*(\d(?:\.\d)?)",
            r"week\s*2\s*side\s*effect[s]?\s*[:\-=]\s*(\d(?:\.\d)?)",
        ]
        medium = [
            r"adverse\s*event\s*severity\s*[:\-=]\s*(\d(?:\.\d)?)",
            r"ae\s*severity\s*[:\-=]\s*(\d(?:\.\d)?)",
            r"side\s*effect\s*severity\s*[:\-=]\s*(\d(?:\.\d)?)",
        ]
        # CTCAE grade
        ctcae = re.search(r"(?:ctcae|nci[\s-]ctcae)\s+grade\s+([1-5])", tl)
        if ctcae:
            try:
                val = min(float(ctcae.group(1)), 5.0)
                return FieldResult(round(val, 1), "Medium", ctcae.group(0), "ctcae_grade")
            except (ValueError, IndexError):
                pass
        for conf, patterns in [("High", high), ("Medium", medium)]:
            for p in patterns:
                m = re.search(p, tl, re.IGNORECASE)
                if m:
                    try:
                        val = float(m.group(1))
                        if 0 <= val <= 5:
                            return FieldResult(round(val, 1), conf, m.group(0), "regex")
                    except (ValueError, IndexError):
                        pass
        return FieldResult(FIELD_DEFAULTS["side_effect_severity_at_week2"], "Low", "", "fallback", True)

    def _insurance(self, text: str, tl: str) -> FieldResult:
        imap = {
            "insured": "insured", "uninsured": "uninsured",
            "partial": "partial", "partially insured": "partial",
            "no insurance": "uninsured", "none": "uninsured",
        }
        high = [
            r"insurance\s*(?:status)?\s*[:\-=]\s*(insured|uninsured|partial(?:ly\s+insured)?|none)",
        ]
        medium = [r"\b(insured|uninsured|partial(?:ly\s+insured)?)\b"]
        for conf, patterns in [("High", high), ("Medium", medium)]:
            for p in patterns:
                m = re.search(p, tl, re.IGNORECASE)
                if m:
                    raw = m.group(1).lower().strip()
                    val = imap.get(raw, "insured")
                    return FieldResult(val, conf, m.group(0), "regex")
        if re.search(r"(?:has|with)\s+(?:health\s+)?insurance", tl):
            return FieldResult("insured", "Medium", "has insurance", "keyword")
        if re.search(r"no\s+(?:health\s+)?insurance", tl):
            return FieldResult("uninsured", "Medium", "no insurance", "keyword")
        return FieldResult(FIELD_DEFAULTS["insurance_status"], "Low", "", "fallback", True)

    def _transportation(self, text: str, tl: str) -> FieldResult:
        high = [
            r"transportation\s*(?:access)?\s*[:\-=]\s*(yes|no|available|unavailable)",
            r"transport(?:ation)?\s*[:\-=]\s*(yes|no)",
        ]
        val_map = {"yes": "yes", "available": "yes", "no": "no", "unavailable": "no"}
        for p in high:
            m = re.search(p, tl, re.IGNORECASE)
            if m:
                raw = m.group(1).lower().strip()
                val = val_map.get(raw, "yes")
                return FieldResult(val, "High", m.group(0), "regex")
        if re.search(r"(?:no|without|lacks?|no\s+reliable)\s+transportation(?:\s+access)?", tl):
            return FieldResult("no", "High", "no transportation", "keyword")
        if re.search(r"(?:has|with|has\s+access\s+to)\s+(?:reliable\s+)?transportation", tl):
            return FieldResult("yes", "High", "has transportation", "keyword")
        return FieldResult(FIELD_DEFAULTS["transportation_access"], "Low", "", "fallback", True)

    def _prior_trial(self, text: str, tl: str) -> FieldResult:
        high = [
            r"(?:prior\s+trial\s+participation|prior\s+trial(?:s)?|previous\s+(?:clinical\s+)?trials?)\s*[:\-=]\s*(\d+)",
            r"(?:participated|enrolled)\s+in\s+(\d+)\s+(?:previous|prior|clinical)\s+trial",
        ]
        medium = [r"prior\s+(?:clinical\s+)?trial\s*[:\-=]\s*(yes|no)"]
        for p in high:
            m = re.search(p, tl, re.IGNORECASE)
            if m:
                try:
                    val = int(m.group(1))
                    if 0 <= val <= 10:
                        return FieldResult(val, "High", m.group(0), "regex")
                except (ValueError, IndexError):
                    pass
        for p in medium:
            m = re.search(p, tl, re.IGNORECASE)
            if m:
                raw = m.group(1).lower().strip()
                return FieldResult(1 if raw == "yes" else 0, "Medium", m.group(0), "regex")
        if re.search(r"(?:first[\s-]time|naive|treatment[\s-]naive)\s+participant", tl):
            return FieldResult(0, "Medium", "first-time participant", "keyword")
        if re.search(r"(?:no|zero|0)\s+prior\s+(?:clinical\s+)?trial", tl):
            return FieldResult(0, "High", "no prior trial", "keyword")
        return FieldResult(FIELD_DEFAULTS["prior_trial_participation"], "Low", "", "fallback", True)

    def _trial_phase(self, text: str, tl: str) -> FieldResult:
        roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
        high = [
            r"(?:trial\s+)?phase\s*[:\-=]\s*([1234]|i{1,3}v?)\b",
            r"phase\s+([1234]|i{1,3}v?)\s+(?:clinical\s+)?trial",
            r"phase\s+([1234]|i{1,3}v?)\b",
        ]
        for p in high:
            m = re.search(p, tl, re.IGNORECASE)
            if m:
                raw = m.group(1).lower().strip()
                val = int(raw) if raw.isdigit() else roman.get(raw, 2)
                if 1 <= val <= 4:
                    return FieldResult(val, "High", m.group(0), "regex")
        return FieldResult(FIELD_DEFAULTS["trial_phase"], "Low", "", "fallback", True)

    def _consent_complexity(self, text: str, tl: str) -> FieldResult:
        high = [
            r"consent\s*complexity(?:\s*score)?\s*[:\-=]\s*(\d+(?:\.\d)?)",
            r"icf\s*complexity\s*(?:score)?\s*[:\-=]\s*(\d+(?:\.\d)?)",
        ]
        medium = [r"consent\s*score\s*[:\-=]\s*(\d+(?:\.\d)?)"]
        for conf, patterns in [("High", high), ("Medium", medium)]:
            for p in patterns:
                m = re.search(p, tl, re.IGNORECASE)
                if m:
                    try:
                        val = float(m.group(1))
                        if 1 <= val <= 10:
                            return FieldResult(round(val, 1), conf, m.group(0), "regex")
                    except (ValueError, IndexError):
                        pass
        return FieldResult(FIELD_DEFAULTS["consent_complexity_score"], "Low", "", "fallback", True)

    def _visit_burden(self, text: str, tl: str) -> FieldResult:
        high = [
            r"visit\s*burden(?:\s*index)?\s*[:\-=]\s*(\d+(?:\.\d)?)",
            r"\bvbi\s*[:\-=]\s*(\d+(?:\.\d)?)",
        ]
        for p in high:
            m = re.search(p, tl, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    if 0 <= val <= 20:
                        return FieldResult(round(val, 1), "High", m.group(0), "regex")
                except (ValueError, IndexError):
                    pass
        return FieldResult(FIELD_DEFAULTS["visit_burden_index"], "Low", "", "fallback", True)

    def _logistic_friction(self, text: str, tl: str) -> FieldResult:
        high = [
            r"logistic(?:al)?\s*friction(?:\s*score)?\s*[:\-=]\s*(\d+(?:\.\d)?)",
            r"\blfs\s*[:\-=]\s*(\d+(?:\.\d)?)",
        ]
        for p in high:
            m = re.search(p, tl, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    if 0 <= val <= 10:
                        return FieldResult(round(val, 1), "High", m.group(0), "regex")
                except (ValueError, IndexError):
                    pass
        return FieldResult(FIELD_DEFAULTS["logistic_friction_score"], "Low", "", "fallback", True)


# ── Audit log builder ─────────────────────────────────────────────────────────

def build_audit_log(
    filename: str,
    extraction_method: str,
    results: Dict[str, FieldResult],
    edited_fields: List[str],
) -> List[dict]:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    high_n  = sum(1 for r in results.values() if r.confidence == "High"   and not r.is_fallback)
    med_n   = sum(1 for r in results.values() if r.confidence == "Medium" and not r.is_fallback)
    fall_n  = sum(1 for r in results.values() if r.is_fallback)
    return [
        {"Event": "Document Uploaded",        "Detail": filename,                               "Timestamp": ts},
        {"Event": "Text Extraction Method",   "Detail": extraction_method,                      "Timestamp": ts},
        {"Event": "High Confidence Extractions", "Detail": f"{high_n} / {len(results)} fields", "Timestamp": ts},
        {"Event": "Medium Confidence Extractions","Detail": f"{med_n} / {len(results)} fields", "Timestamp": ts},
        {"Event": "Fallback / Missing Fields", "Detail": f"{fall_n} / {len(results)} fields",   "Timestamp": ts},
        {"Event": "Fields Edited by User",    "Detail": ", ".join(edited_fields) or "None",     "Timestamp": ts},
        {"Event": "Confirmed for Analysis",   "Detail": "Pending user confirmation",            "Timestamp": ts},
    ]


# ── Sample CRF generator ──────────────────────────────────────────────────────

def generate_sample_crf() -> bytes:
    """
    Generate a synthetic CRF PDF that exercises all extraction patterns.
    Requires fpdf2 (already in requirements). Returns PDF bytes or b"" on error.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return b""

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(18, 18, 18)

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(13, 27, 42)
    pdf.rect(0, 0, 210, 24, "F")
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 6)
    pdf.cell(0, 7, "CLINICAL TRIAL SCREENING & INTAKE FORM  |  CONFIDENTIAL", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(10, 14)
    pdf.cell(0, 5, "Study: CT-2026-ONK-004  |  Sponsor: Demo Pharma  |  Protocol v3.2", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(18, 30)

    TEAL = (29, 158, 117)

    def section(title: str):
        pdf.ln(3)
        pdf.set_fill_color(*TEAL)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 7, f"  {title}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 9)
        pdf.ln(1)

    def row(label: str, value: str):
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(88, 5.5, f"{label}:", ln=False)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5.5, str(value), ln=True)

    section("PARTICIPANT DEMOGRAPHICS")
    row("Participant Age",    "62 years")
    row("Gender",             "Female")
    row("Date of Birth",      "15/03/1962")
    row("BMI",                "28.3 kg/m²")
    row("Employment Status",  "Retired")
    row("Education Level",    "Graduate")
    row("Insurance Status",   "Insured")
    row("Transportation Access", "No")

    section("CLINICAL PROFILE")
    row("Disease Severity Score",          "7.5 / 10")
    row("Number of Comorbidities",         "4")
    row("Baseline Pain Score",             "6.0")
    row("Prior Adverse Event History",     "Yes")
    row("Concomitant Medications",         "9")
    row("Side Effect Severity at Week 2",  "3.5")

    section("LOGISTICAL PROFILE")
    row("Distance from Trial Site",        "72 km")
    row("Visit Frequency per Month",       "5")
    row("Prior Trial Participation",       "1")
    row("Visit Burden Index",              "7")
    row("Logistic Friction Score",         "6")

    section("TRIAL CHARACTERISTICS")
    row("Trial Phase",                     "Phase 2")
    row("Protocol Complexity Score",       "7")
    row("Trial Duration",                  "18 months")
    row("Consent Complexity Score",        "6")
    row("Investigator Experience",         "6 years")
    row("Trial Site",                      "SITE_04")

    section("INVESTIGATOR NOTES")
    pdf.set_font("Helvetica", "", 9)
    notes = (
        "Participant is a 62-year-old retired female with moderate-severe disease. "
        "She has 4 known comorbidities including hypertension and Type 2 diabetes. "
        "Currently taking 9 concomitant medications. No reliable transportation access - "
        "lives 72 km from site. Prior participation in 1 previous clinical trial. "
        "Insurance: Insured. Consent complexity assessed as 6/10. "
        "Week 2 adverse event severity assessed at 3.5 by site investigator."
    )
    pdf.multi_cell(0, 5, notes)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 5,
             "Synthetic document - for demonstration purposes only.  "
             "TrialGuard  |  AI-Powered Clinical Trial Intelligence Platform  |  v2.0",
             align="C")

    return bytes(pdf.output())
