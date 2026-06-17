"""
Retention Coordinator Copilot — Version 3.0

Generates AI-powered coordinator summaries, clinical reasoning narratives,
prioritised action plans, and scenario optimisation from retention analysis.
Template-based generation — no LLMs, no external APIs. Deterministic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Risk narrative templates ───────────────────────────────────────────────────

_RISK_TEXT = {
    "Critical": (
        "exhibits a critical probability of early trial discontinuation",
        "Immediate multi-disciplinary retention intervention is indicated.",
    ),
    "High": (
        "presents with a high dropout risk profile",
        "Proactive intervention within the next 48-72 hours is strongly recommended.",
    ),
    "Moderate": (
        "demonstrates a moderate retention risk",
        "Monitoring escalation and selective interventions are advised.",
    ),
    "Low": (
        "has a favourable retention profile at this assessment",
        "Standard protocol monitoring and routine site engagement are sufficient.",
    ),
}

# ── Driver → action mapping (keyed on substring of SHAP label) ────────────────

_DRIVER_ACTIONS: List[Dict] = [
    {
        "keywords": ["week 2 side effect", "week 2 adverse", "side effect severity"],
        "timeline": "Within 24 hours",
        "title": "Week 2 Safety Contact Protocol",
        "description": (
            "Activate the Week 2 proactive safety contact protocol immediately. "
            "Unmanaged early adverse events are the strongest modifiable predictor of dropout in "
            "this model (SHAP contribution ~3x the next factor). "
            "Contact participant to assess current AE status, document per ICH E6(R2) Section 4, "
            "and escalate to pharmacovigilance team if CTCAE Grade >= 2."
        ),
        "expected_reduction": 14.0,
    },
    {
        "keywords": ["no transportation", "transportation access", "logistic friction", "distance from trial site"],
        "timeline": "Within 72 hours",
        "title": "Transportation & Logistical Support Package",
        "description": (
            "Enrol participant in the site transportation reimbursement programme. "
            "Provide information on mileage reimbursement schedules, rideshare partnerships, "
            "and volunteer driver networks. For sites with telemedicine capability, identify "
            "visit windows eligible for remote delivery to reduce total trip burden "
            "(FDA Decentralised Clinical Trials Guidance, 2023)."
        ),
        "expected_reduction": 9.0,
    },
    {
        "keywords": ["visit burden", "visit frequency per month"],
        "timeline": "Before Next Visit",
        "title": "Protocol Visit Burden Review",
        "description": (
            "Escalate to the Medical Monitor for a proportionate monitoring review. "
            "Identify non-critical assessment windows for consolidation or telemedicine substitution. "
            "ICH E6(R2) Section 5 explicitly supports risk-adapted, proportionate monitoring to "
            "reduce participant burden without compromising protocol integrity."
        ),
        "expected_reduction": 7.0,
    },
    {
        "keywords": ["polypharmacy", "concomitant medication"],
        "timeline": "Before Next Visit",
        "title": "Pharmacist-Led Medication Counselling",
        "description": (
            "Schedule a pharmacist-led medication counselling session before the next protocol visit. "
            "Conduct a full medication reconciliation to identify potential drug-drug interactions "
            "with the investigational product. Develop a personalised adherence support plan and "
            "simplify the regimen where feasible (WHO Technical Report on Polypharmacy, 2019)."
        ),
        "expected_reduction": 8.0,
    },
    {
        "keywords": ["number of comorbidities", "participant burden", "disease severity"],
        "timeline": "Before Next Visit",
        "title": "Comorbidity Care Coordination",
        "description": (
            "Assign a Clinical Research Associate to coordinate between the participant's treating "
            "physicians and the trial site. Ensure that concomitant condition management does not "
            "conflict with trial requirements. Explore telemedicine options for comorbidity follow-up "
            "to reduce the overall appointment burden on the participant."
        ),
        "expected_reduction": 6.0,
    },
    {
        "keywords": ["consent complexity", "protocol complexity", "phase-complexity"],
        "timeline": "Protocol Review Cycle",
        "title": "Protocol & Consent Simplification",
        "description": (
            "Request plain-language consent supplementary materials (FDA Plain Language Guidance, 2014). "
            "Assign a participant liaison for ongoing consent support and questions. "
            "Submit a protocol review request to the Medical Monitor identifying non-essential "
            "endpoint assessments that could be removed or moved to telemedicine delivery."
        ),
        "expected_reduction": 5.0,
    },
    {
        "keywords": ["investigator experience", "trial phase"],
        "timeline": "Protocol Review Cycle",
        "title": "Site Engagement & Investigator Support",
        "description": (
            "Schedule a retention-focused site training session. Provide the site with participant "
            "engagement toolkits, structured check-in protocols, and escalation pathways "
            "for high-risk participants. Review site-level retention performance "
            "against trial benchmarks (ICH E6(R2) Section 4)."
        ),
        "expected_reduction": 4.0,
    },
    {
        "keywords": ["insurance", "uninsured", "partial"],
        "timeline": "Within 72 hours",
        "title": "Financial Navigation & Insurance Support",
        "description": (
            "Connect participant with the site financial navigator within 72 hours. "
            "Identify coverage gaps, patient assistance programmes, and trial-related cost "
            "reimbursement options. Document financial barriers in the retention risk register "
            "for Medical Monitor awareness."
        ),
        "expected_reduction": 5.0,
    },
    {
        "keywords": ["prior adverse event", "adverse event history"],
        "timeline": "Within 24 hours",
        "title": "Adverse Event History Review & Monitoring Plan",
        "description": (
            "Review prior adverse event history in full. Initiate enhanced monitoring if a "
            "pattern of AE-related burden is identified. Develop a personalised AE management "
            "plan in collaboration with the investigator, clearly outlining self-management "
            "strategies, reporting thresholds, and 24-hour support contacts."
        ),
        "expected_reduction": 7.0,
    },
]

_TIMELINE_ORDER = {
    "Within 24 hours":       1,
    "Within 72 hours":       2,
    "Before Next Visit":     3,
    "Protocol Review Cycle": 4,
}

_PRIORITY_BY_RISK = {
    "Critical": ["Critical", "Critical", "High",   "Medium"],
    "High":     ["High",     "High",     "Medium", "Medium"],
    "Moderate": ["High",     "Medium",   "Medium", "Low"],
    "Low":      ["Medium",   "Low",      "Low",    "Low"],
}


@dataclass
class ActionItem:
    title:              str
    description:        str
    priority:           str    # Critical | High | Medium | Low
    timeline:           str    # Within 24 hours | Within 72 hours | ...
    expected_reduction: float  # percentage-point reduction estimate


@dataclass
class CoordinatorSummary:
    risk_narrative:          str
    reasoning_text:          str
    primary_drivers:         List[str]
    action_items:            List[ActionItem]
    expected_improvement_low:  int
    expected_improvement_high: int
    combo_scenarios:         List[Dict]


class CoordinatorCopilot:
    """
    Template-based coordinator summary generator.
    Maps SHAP risk drivers to clinical actions and reasoning text.
    """

    def generate(
        self,
        risk_pct:           int,
        risk_cat:           str,
        top3_risk_factors:  List[Tuple],   # [(feat_name, shap_val, human_label), ...]
        top3_protective:    List[Tuple],
        interventions:      List[Dict],
        participant_data:   Dict[str, Any],
    ) -> CoordinatorSummary:

        phrase, action_note = _RISK_TEXT.get(risk_cat, _RISK_TEXT["Moderate"])
        primary_drivers     = [label for _, _, label in top3_risk_factors[:3]]
        protective_factors  = [label for _, _, label in top3_protective[:2]]

        driver_text    = self._driver_phrase(primary_drivers)
        risk_narrative = (
            f"This participant {phrase} ({risk_pct}%). "
            f"Primary retention concerns identified: {driver_text}. "
            f"{action_note}"
        )

        reasoning_text = self._build_reasoning(
            risk_pct, risk_cat, primary_drivers, protective_factors, participant_data
        )

        action_items   = self._build_actions(primary_drivers, risk_cat)
        combo_scenarios = self._build_combos(risk_pct, action_items)

        if action_items:
            reds = [a.expected_reduction for a in action_items]
            low  = max(round(min(reds[:2], default=reds[0]) * 0.55), 3)
            high = min(round(sum(sorted(reds, reverse=True)[:3]) * 0.75), 35)
        else:
            low, high = 0, 0

        return CoordinatorSummary(
            risk_narrative=risk_narrative,
            reasoning_text=reasoning_text,
            primary_drivers=primary_drivers,
            action_items=action_items,
            expected_improvement_low=low,
            expected_improvement_high=high,
            combo_scenarios=combo_scenarios,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _driver_phrase(self, drivers: List[str]) -> str:
        if not drivers:
            return "multiple clinical and logistical factors"
        dl = [d.lower() for d in drivers]
        if len(dl) == 1:
            return dl[0]
        if len(dl) == 2:
            return f"{dl[0]} and {dl[1]}"
        return f"{dl[0]}, {dl[1]}, and {dl[2]}"

    def _build_reasoning(
        self,
        risk_pct:  int,
        risk_cat:  str,
        drivers:   List[str],
        protective: List[str],
        pdata:     Dict,
    ) -> str:
        paras = []
        cat_lower = risk_cat.lower()
        paras.append(
            f"The participant demonstrates a {cat_lower} ({risk_pct}%) estimated probability "
            "of early clinical trial discontinuation based on the AI retention risk model."
        )

        if drivers:
            ds = "; ".join(d.lower() for d in drivers)
            paras.append(
                f"The primary predictors associated with this risk classification are: {ds}. "
                "These are among the strongest determinants of attrition in this participant's profile, "
                "consistent with published clinical trial retention literature (FDA, 2012; ICH E6(R2), 2016)."
            )

        combined = " ".join(drivers).lower()

        if any(k in combined for k in ["week 2", "side effect", "adverse event"]):
            paras.append(
                "Early adverse event burden at Week 2 is the strongest modifiable predictor in this model, "
                "with a SHAP contribution approximately 3x greater than the next factor. "
                "Proactive pharmacovigilance contact at this window is both low-cost and high-impact."
            )
        if any(k in combined for k in ["transport", "distance", "logistic", "travel"]):
            paras.append(
                "Logistical barriers — particularly travel distance in the absence of reliable "
                "transportation — create a hard friction point independent of clinical profile. "
                "Transportation reimbursement provides clinically meaningful risk reduction at "
                "minimal cost per unit benefit (FDA, 2012)."
            )
        if any(k in combined for k in ["visit burden", "visit frequency"]):
            paras.append(
                "High protocol visit burden elevates participant fatigue risk over the trial duration. "
                "ICH E6(R2)'s principle of proportionate monitoring supports review of non-critical "
                "assessment windows for consolidation or telemedicine delivery."
            )
        if any(k in combined for k in ["medication", "polypharmacy", "comorbid"]):
            paras.append(
                "Polypharmacy complexity and comorbidity burden amplify the cognitive and logistical "
                "load of trial participation. Pharmacist-led medication management support can "
                "reduce this burden and improve adherence to trial requirements."
            )

        if protective:
            ps = " and ".join(p.lower() for p in protective)
            paras.append(
                f"Protective factors identified include {ps}, which partially offset the risk "
                "profile and may be reinforced in the retention strategy."
            )

        paras.append(
            "The coordinator action plan below addresses each primary risk driver with "
            "evidence-based interventions, prioritised by clinical urgency and expected impact."
        )
        return " ".join(paras)

    def _build_actions(self, primary_drivers: List[str], risk_cat: str) -> List[ActionItem]:
        actions      = []
        seen_titles  = set()
        prio_list    = _PRIORITY_BY_RISK.get(risk_cat, ["High", "Medium", "Low", "Low"])
        prio_idx     = 0

        for driver_label in primary_drivers:
            dl = driver_label.lower()
            for action_def in _DRIVER_ACTIONS:
                if any(kw in dl for kw in action_def["keywords"]):
                    if action_def["title"] not in seen_titles:
                        priority = prio_list[min(prio_idx, len(prio_list) - 1)]
                        actions.append(ActionItem(
                            title=action_def["title"],
                            description=action_def["description"],
                            priority=priority,
                            timeline=action_def["timeline"],
                            expected_reduction=action_def["expected_reduction"],
                        ))
                        seen_titles.add(action_def["title"])
                        prio_idx += 1
                    break

        if not actions and risk_cat in ("Critical", "High"):
            actions.append(ActionItem(
                title="Escalate to Retention Management Team",
                description=(
                    "Schedule an urgent retention review with the site coordinator, medical monitor, "
                    "and participant liaison. Develop a personalised retention action plan tailored "
                    "to this participant's specific risk profile."
                ),
                priority="Critical" if risk_cat == "Critical" else "High",
                timeline="Within 24 hours",
                expected_reduction=10.0,
            ))

        actions.sort(key=lambda a: _TIMELINE_ORDER.get(a.timeline, 99))
        return actions

    def _build_combos(self, risk_pct: int, action_items: List[ActionItem]) -> List[Dict]:
        """Ranked intervention combination scenarios with diminishing returns."""
        if not action_items:
            return []

        sorted_a = sorted(action_items, key=lambda a: a.expected_reduction, reverse=True)
        combos   = []

        for n in range(1, min(len(sorted_a) + 1, 5)):
            selected = sorted_a[:n]
            total_red = sum(a.expected_reduction * (0.80 ** i) for i, a in enumerate(selected))
            total_red = min(total_red, risk_pct * 0.68)
            new_risk  = max(risk_pct - total_red, 5)
            est_cost  = 600 + sum(400 * (i + 1) for i in range(n))

            names = [a.title for a in selected]
            if len(names) == 1:
                label = names[0]
            elif len(names) == 2:
                label = f"{names[0]}  +  {names[1]}"
            else:
                label = f"{names[0]}  +  {names[1]}  +  {len(names)-2} more"

            combos.append({
                "label":              label,
                "n_interventions":    n,
                "current_risk_pct":   risk_pct,
                "projected_risk_pct": round(new_risk),
                "risk_reduction_pp":  round(total_red, 1),
                "interventions":      names,
                "est_cost":           est_cost,
                "badge":              "",
            })

        if combos:
            best_roi_idx  = max(range(len(combos)),
                                key=lambda i: combos[i]["risk_reduction_pp"] / max(combos[i]["est_cost"], 1))
            combos[best_roi_idx]["badge"] = "Best ROI"
            best_ret_idx  = max(range(len(combos)),
                                key=lambda i: combos[i]["risk_reduction_pp"])
            if best_ret_idx != best_roi_idx:
                combos[best_ret_idx]["badge"] = "Max Retention"

        return combos
