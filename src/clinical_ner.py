"""
Clinical Named Entity Recognition — Version 3.0

Dictionary-based entity recognition for clinical trial documents.
Identifies diseases, symptoms, adverse events, medications,
trial burden indicators, and patient engagement signals.
No LLMs. No external APIs. Deterministic pattern matching.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ── Entity dictionaries ────────────────────────────────────────────────────────

DISEASE_PATTERNS: Dict[str, str] = {
    r"\btype\s+2\s+diabetes\b":               "Type 2 Diabetes Mellitus",
    r"\btype\s+1\s+diabetes\b":               "Type 1 Diabetes Mellitus",
    r"\bdiabetes\s+mellitus\b":               "Diabetes Mellitus",
    r"\bdiabetes\b":                          "Diabetes",
    r"\bhypertension\b":                      "Hypertension",
    r"\bhigh\s+blood\s+pressure\b":           "Hypertension",
    r"\bcoronary\s+artery\s+disease\b":       "Coronary Artery Disease",
    r"\bcoronary\s+heart\s+disease\b":        "Coronary Heart Disease",
    r"\bheart\s+(?:disease|failure)\b":       "Cardiac Disease",
    r"\batrial\s+fibrillation\b":             "Atrial Fibrillation",
    r"\bafib\b":                              "Atrial Fibrillation",
    r"\bchronic\s+obstructive\s+pulmonary\b": "COPD",
    r"\bcopd\b":                              "COPD",
    r"\basthma\b":                            "Asthma",
    r"\bchronic\s+kidney\s+disease\b":        "Chronic Kidney Disease",
    r"\bckd\b":                               "Chronic Kidney Disease",
    r"\brenal\s+(?:disease|failure|impairment)\b": "Renal Disease",
    r"\bliver\s+disease\b":                   "Hepatic Disease",
    r"\bhepatic\s+(?:disease|impairment|failure)\b": "Hepatic Disease",
    r"\bdepression\b":                        "Depression",
    r"\banxiety\s+disorder\b":                "Anxiety Disorder",
    r"\banxiety\b":                           "Anxiety",
    r"\bobesity\b":                           "Obesity",
    r"\bosteoarthritis\b":                    "Osteoarthritis",
    r"\brheumatoid\s+arthritis\b":            "Rheumatoid Arthritis",
    r"\barthritis\b":                         "Arthritis",
    r"\bosteoporosis\b":                      "Osteoporosis",
    r"\bhypothyroidism\b":                    "Hypothyroidism",
    r"\bhyperthyroidism\b":                   "Hyperthyroidism",
    r"\bthyroid\s+disease\b":                 "Thyroid Disease",
    r"\blung\s+cancer\b":                     "Lung Cancer",
    r"\bbreast\s+cancer\b":                   "Breast Cancer",
    r"\bcancer\b":                            "Oncological Condition",
    r"\bmalignancy\b":                        "Malignancy",
    r"\bcerebrovascular\s+disease\b":         "Cerebrovascular Disease",
    r"\bstroke\b":                            "Cerebrovascular Disease",
    r"\balzheimer\b":                         "Alzheimer's Disease",
    r"\bdementia\b":                          "Dementia",
    r"\bparkinson\b":                         "Parkinson's Disease",
    r"\bepileps[yi]\b":                       "Epilepsy",
    r"\bseizure\s+disorder\b":                "Seizure Disorder",
    r"\bgerd\b":                              "GERD",
    r"\bgastroesophageal\s+reflux\b":         "GERD",
    r"\birritable\s+bowel\b":                 "IBS",
    r"\bibs\b":                               "IBS",
    r"\banaemia\b":                           "Anaemia",
    r"\banemia\b":                            "Anaemia",
    r"\bfibromyalgia\b":                      "Fibromyalgia",
    r"\bmigraine\b":                          "Migraine",
    r"\bchronic\s+pain\b":                    "Chronic Pain",
    r"\bmultiple\s+sclerosis\b":              "Multiple Sclerosis",
    r"\bsle\b":                               "Systemic Lupus Erythematosus",
    r"\blupus\b":                             "Systemic Lupus Erythematosus",
    r"\bpsoriasis\b":                         "Psoriasis",
    r"\beczema\b":                            "Eczema",
}

SYMPTOM_PATTERNS: Dict[str, str] = {
    r"\bnausea\b":                            "Nausea",
    r"\bvomiting\b":                          "Vomiting",
    r"\bfatigue\b":                           "Fatigue",
    r"\bdizziness\b":                         "Dizziness",
    r"\bvertigo\b":                           "Vertigo",
    r"\bheadache\b":                          "Headache",
    r"\binsomnia\b":                          "Insomnia",
    r"\bsleep\s+(?:disturbance|disorder|problem)\b": "Sleep Disturbance",
    r"\bshortness\s+of\s+breath\b":           "Dyspnoea",
    r"\bdyspnoea\b":                          "Dyspnoea",
    r"\bbreathing\s+difficulty\b":            "Dyspnoea",
    r"\bchest\s+pain\b":                      "Chest Pain",
    r"\babdominal\s+pain\b":                  "Abdominal Pain",
    r"\bback\s+pain\b":                       "Back Pain",
    r"\bjoint\s+pain\b":                      "Joint Pain",
    r"\bappetite\s+loss\b":                   "Loss of Appetite",
    r"\bweight\s+loss\b":                     "Weight Loss",
    r"\bdiarrhea\b":                          "Diarrhoea",
    r"\bdiarrhoea\b":                         "Diarrhoea",
    r"\bconstipation\b":                      "Constipation",
    r"\bedema\b":                             "Oedema",
    r"\boedema\b":                            "Oedema",
    r"\bswelling\b":                          "Oedema",
    r"\bpalpitation\b":                       "Cardiac Palpitations",
    r"\btremor\b":                            "Tremor",
    r"\bweakness\b":                          "Weakness",
    r"\bnumbness\b":                          "Paraesthesia",
    r"\btingling\b":                          "Paraesthesia",
    r"\bblurred\s+vision\b":                  "Visual Disturbance",
    r"\bmemory\s+(?:loss|problem)\b":         "Memory Impairment",
    r"\bconfusion\b":                         "Cognitive Impairment",
    r"\brash\b":                              "Dermatological Reaction",
    r"\bskin\s+reaction\b":                   "Dermatological Reaction",
}

AE_PATTERNS: Dict[str, str] = {
    r"\bserious\s+adverse\s+event\b":         "Serious Adverse Event (SAE)",
    r"\bsae\b":                               "Serious Adverse Event (SAE)",
    r"\badverse\s+event\b":                   "Adverse Event (AE)",
    r"\badverse\s+reaction\b":                "Adverse Drug Reaction",
    r"\badverse\s+drug\s+reaction\b":         "Adverse Drug Reaction",
    r"\badr\b":                               "Adverse Drug Reaction",
    r"\bctcae\s+grade\s+[3-5]\b":             "High-Grade Toxicity (CTCAE G3-5)",
    r"\bgrade\s+[3-5]\s+(?:ae|adverse|toxicity)\b": "High-Grade Toxicity",
    r"\bdiscontinued\b":                      "Prior Discontinuation",
    r"\bwithdr[ea]w\b":                       "Prior Withdrawal",
    r"\bdose\s+reduction\b":                  "Dose Reduction Required",
    r"\bdose\s+interruption\b":               "Dose Interruption",
    r"\bhospitali[sz]ation\b":                "Hospitalisation Event",
    r"\bhospitali[sz]ed\b":                   "Hospitalisation Event",
    r"\bside\s+effect\b":                     "Side Effect Reported",
}

MEDICATION_PATTERNS: List[str] = [
    r"\bmetformin\b", r"\binsulin\b", r"\blisinopril\b", r"\bamlodipine\b",
    r"\batorvastatin\b", r"\brosuvastatin\b", r"\bsimvastatin\b", r"\bstatin\b",
    r"\bwarfarin\b", r"\bapixaban\b", r"\brivaroxaban\b", r"\baspirin\b",
    r"\bibuprofen\b", r"\bnaproxen\b", r"\bcelecoxib\b",
    r"\bomeprazole\b", r"\bpantoprazole\b", r"\besomeprazole\b",
    r"\blevothyroxine\b", r"\bmetoprolol\b", r"\bcarvedilol\b",
    r"\bfurosemide\b", r"\bspironolactone\b", r"\bamiodarone\b",
    r"\bsertraline\b", r"\bfluoxetine\b", r"\bcitalopram\b", r"\bescitalopram\b",
    r"\bvenlafaxine\b", r"\bduloxetine\b", r"\balprazolam\b", r"\blorazepam\b",
    r"\bgabapentin\b", r"\bpregabalin\b", r"\blamotrigine\b",
    r"\bprednisone\b", r"\bprednisolone\b", r"\bdexamethasone\b",
    r"\bmethotrexate\b", r"\bhydroxychloroquine\b",
    r"\btamoxifen\b", r"\bletrozole\b", r"\bbevacizumab\b",
    r"\bimatinib\b", r"\bosimertinib\b", r"\bpembrolizumab\b",
    r"\bnivolumab\b", r"\batezolizumab\b",
    r"\betoposide\b", r"\bpaclitaxel\b", r"\bcarboplatin\b", r"\bcisplatin\b",
    r"\bglipizide\b", r"\bsitagliptin\b", r"\bempagliflozin\b",
    r"\bdapagliflozin\b", r"\bsemaglutide\b", r"\bliraglutide\b",
]

BURDEN_PATTERNS: Dict[str, str] = {
    r"\bmissed\s+(?:visit|appointment|dose)\b":      "Missed Visit/Dose",
    r"\btransportation\s+(?:difficulty|challenge|barrier|issue|problem)\b": "Transportation Barrier",
    r"\bno\s+(?:reliable\s+)?transportation\b":      "No Transportation Access",
    r"\bunable\s+to\s+(?:travel|commute|drive)\b":   "Travel Barrier",
    r"\blong\s+(?:travel|commute|distance)\b":       "Long Travel Distance",
    r"\bhigh\s+visit\s+(?:burden|frequency)\b":      "High Visit Burden",
    r"\bfrequent\s+visit\b":                         "Frequent Visit Demand",
    r"\bprotocol\s+(?:burden|complexity)\b":         "Protocol Complexity Burden",
    r"\bmedication\s+non[\s-]?adherence\b":          "Medication Non-Adherence",
    r"\bnon[\s-]?adher[ae]nt\b":                     "Non-Adherence Signal",
    r"\bpolypharmacy\b":                             "Polypharmacy Risk",
    r"\bprotocol\s+deviation\b":                     "Protocol Deviation",
    r"\bwork\s+(?:conflict|schedule|barrier)\b":     "Work Schedule Conflict",
    r"\bcaregiver\s+(?:burden|responsibility)\b":    "Caregiver Responsibility",
    r"\bfar\s+from\s+(?:the\s+)?site\b":             "Distance Barrier",
    r"\bfinancial\s+(?:barrier|difficulty|concern)\b": "Financial Barrier",
    r"\binsurance\s+(?:issue|problem|concern)\b":    "Insurance Issue",
}

ENGAGEMENT_PATTERNS: Dict[str, str] = {
    r"\bmotivated\b":                         "Positive Engagement Signal",
    r"\bcommitted\b":                         "Strong Commitment",
    r"\bcompliant\b":                         "Adherent Profile",
    r"\badherent\b":                          "Adherent Profile",
    r"\bwilling\s+to\s+participate\b":        "Willingness Confirmed",
    r"\bfirst[\s-]time\s+participant\b":      "First-Time Participant",
    r"\btrial[\s-]naive\b":                   "Trial-Naive Participant",
    r"\bhas\s+(?:reliable\s+)?transportation\b": "Transport Access Confirmed",
    r"\binsured\b":                           "Insurance Coverage Confirmed",
    r"\bprior\s+(?:trial\s+)?(?:experience|participation)\b": "Experienced Participant",
    r"\bno\s+significant\s+(?:ae|adverse\s+event)\b": "Favourable AE Profile",
    r"\bgood\s+(?:tolerability|tolerance)\b": "Good Tolerability",
    r"\bsupport(?:ive)?\s+(?:family|caregiver|support\s+system)\b": "Family Support",
}


@dataclass
class NERResult:
    diseases:          List[Tuple[str, str]] = field(default_factory=list)  # (raw, canonical)
    symptoms:          List[Tuple[str, str]] = field(default_factory=list)
    adverse_events:    List[Tuple[str, str]] = field(default_factory=list)
    medications:       List[str]             = field(default_factory=list)
    burden_flags:      List[Tuple[str, str]] = field(default_factory=list)
    engagement_flags:  List[Tuple[str, str]] = field(default_factory=list)
    inferred_comorbidity_count: int          = 0

    @property
    def total_entities(self) -> int:
        return (len(self.diseases) + len(self.symptoms) + len(self.adverse_events)
                + len(self.medications) + len(self.burden_flags) + len(self.engagement_flags))

    @property
    def has_high_ae_burden(self) -> bool:
        return any("SAE" in c or "High-Grade" in c or "Discontinuation" in c
                   for _, c in self.adverse_events)

    @property
    def has_transport_barrier(self) -> bool:
        return any("Transportation" in c or "Travel" in c or "Distance" in c
                   for _, c in self.burden_flags)


class ClinicalNERExtractor:
    """Dictionary-based clinical entity recognition — deterministic, no LLMs."""

    def extract(self, text: str) -> NERResult:
        tl = text.lower()
        result = NERResult()
        seen = {cat: set() for cat in
                ("diseases", "symptoms", "ae", "meds", "burden", "engagement")}

        for pattern, canonical in DISEASE_PATTERNS.items():
            if canonical not in seen["diseases"]:
                m = re.search(pattern, tl, re.IGNORECASE)
                if m:
                    result.diseases.append((m.group(0), canonical))
                    seen["diseases"].add(canonical)

        result.inferred_comorbidity_count = len(result.diseases)

        for pattern, canonical in SYMPTOM_PATTERNS.items():
            if canonical not in seen["symptoms"]:
                m = re.search(pattern, tl, re.IGNORECASE)
                if m:
                    result.symptoms.append((m.group(0), canonical))
                    seen["symptoms"].add(canonical)

        for pattern, canonical in AE_PATTERNS.items():
            if canonical not in seen["ae"]:
                m = re.search(pattern, tl, re.IGNORECASE)
                if m:
                    result.adverse_events.append((m.group(0), canonical))
                    seen["ae"].add(canonical)

        for pattern in MEDICATION_PATTERNS:
            m = re.search(pattern, tl, re.IGNORECASE)
            if m:
                name = m.group(0).strip()
                if name not in seen["meds"]:
                    result.medications.append(name)
                    seen["meds"].add(name)

        for pattern, canonical in BURDEN_PATTERNS.items():
            if canonical not in seen["burden"]:
                m = re.search(pattern, tl, re.IGNORECASE)
                if m:
                    result.burden_flags.append((m.group(0), canonical))
                    seen["burden"].add(canonical)

        for pattern, canonical in ENGAGEMENT_PATTERNS.items():
            if canonical not in seen["engagement"]:
                m = re.search(pattern, tl, re.IGNORECASE)
                if m:
                    result.engagement_flags.append((m.group(0), canonical))
                    seen["engagement"].add(canonical)

        return result
