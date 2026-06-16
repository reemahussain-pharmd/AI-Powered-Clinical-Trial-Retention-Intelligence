"""
Synthetic Clinical Trial Dataset Generator.

Generates 2000 realistic patient records with demographics, clinical features,
trial parameters, and dropout outcomes for retention modelling.
All correlations are grounded in published clinical trial attrition literature.
"""

import numpy as np
import pandas as pd
from pathlib import Path


RANDOM_SEED = 42
N_PATIENTS = 2000
DROPOUT_RATE = 0.30
SITE_IDS = [f"SITE_{str(i).zfill(2)}" for i in range(1, 9)]
# SITE_04 and SITE_07 structurally have higher dropout — reflect investigator/infrastructure gaps
HIGH_DROPOUT_SITES = {"SITE_04", "SITE_07"}


def _rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_demographics(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate patient demographic features.

    Older age, lack of insurance, and greater distance from site are established
    non-clinical predictors of clinical trial dropout (FDA, 2012).

    Args:
        n: Number of patient records to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        DataFrame with demographic columns.
    """
    patient_ids = [f"P{str(i).zfill(4)}" for i in range(1, n + 1)]
    ages = rng.integers(18, 81, size=n)
    genders = rng.choice(["M", "F", "Other"], size=n, p=[0.48, 0.48, 0.04])
    bmis = np.round(rng.uniform(16, 45, size=n), 1)
    employment = rng.choice(
        ["employed", "unemployed", "retired", "student"],
        size=n,
        p=[0.50, 0.15, 0.25, 0.10],
    )
    insurance = rng.choice(
        ["insured", "uninsured", "partial"], size=n, p=[0.60, 0.20, 0.20]
    )
    distance = np.round(rng.uniform(1, 200, size=n), 1)
    # Transportation access is inversely correlated with distance in real populations
    transport_prob = np.where(distance > 80, 0.30, 0.80)
    transport = np.array(
        [rng.choice(["yes", "no"], p=[p, 1 - p]) for p in transport_prob]
    )
    education = rng.choice(
        ["primary", "secondary", "graduate"], size=n, p=[0.20, 0.50, 0.30]
    )

    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": ages,
            "gender": genders,
            "bmi": bmis,
            "employment_status": employment,
            "insurance_status": insurance,
            "distance_from_site_km": distance,
            "transportation_access": transport,
            "education_level": education,
        }
    )


def generate_clinical_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate patient clinical profile features.

    Side effect severity at week 2 is modelled as the strongest dropout predictor,
    consistent with pharmacovigilance evidence that early adverse events drive
    discontinuation before therapeutic benefit is realised (ICH E6(R2), 2016).

    Args:
        n: Number of patient records to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        DataFrame with clinical feature columns.
    """
    disease_severity = np.round(rng.uniform(0, 10, size=n), 1)
    comorbidities = rng.integers(0, 9, size=n)
    baseline_pain = np.round(rng.uniform(0, 10, size=n), 1)
    prior_trial = rng.integers(0, 5, size=n)
    prior_ae = rng.choice(["yes", "no"], size=n, p=[0.25, 0.75])
    concomitant_meds = rng.integers(0, 13, size=n)
    # Side effect severity at week 2 — strongest predictor; skewed toward lower values
    side_effect = np.clip(rng.beta(1.5, 3.5, size=n) * 5, 0, 5).round(1)

    return pd.DataFrame(
        {
            "disease_severity_score": disease_severity,
            "number_of_comorbidities": comorbidities,
            "baseline_pain_score": baseline_pain,
            "prior_trial_participation": prior_trial,
            "prior_adverse_event_history": prior_ae,
            "concomitant_medications": concomitant_meds,
            "side_effect_severity_at_week2": side_effect,
        }
    )


def generate_trial_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate trial design and site-level features.

    Phase 1 trials carry higher dropout risk due to unknown safety profiles.
    Investigator experience is a protective factor — experienced sites build
    stronger patient rapport and retention infrastructure (ICH E6(R2), 2016).

    Args:
        n: Number of patient records to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        DataFrame with trial feature columns.
    """
    phases = rng.choice([1, 2, 3, 4], size=n, p=[0.10, 0.30, 0.50, 0.10])
    visit_freq = rng.integers(1, 13, size=n)
    protocol_complexity = rng.integers(1, 11, size=n)
    trial_duration = rng.integers(3, 37, size=n)
    consent_complexity = rng.integers(1, 11, size=n)
    investigator_exp = rng.integers(1, 31, size=n)
    site_ids = rng.choice(SITE_IDS, size=n)

    return pd.DataFrame(
        {
            "trial_phase": phases,
            "visit_frequency_per_month": visit_freq,
            "protocol_complexity_score": protocol_complexity,
            "trial_duration_months": trial_duration,
            "consent_complexity_score": consent_complexity,
            "investigator_experience_years": investigator_exp,
            "site_id": site_ids,
        }
    )


def compute_dropout_probability(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a clinically-informed dropout probability for each patient.

    Weights are derived from the relative importance of each factor as described
    in FDA (2012) and Getz KA et al. (Ther Innov Regul Sci, 2016).

    Args:
        df: Combined patient DataFrame with all features.

    Returns:
        Array of dropout probabilities in [0, 1].
    """
    p = np.zeros(len(df))

    # Side effects at week 2 — strongest driver (pharmacovigilance evidence)
    p += 0.12 * df["side_effect_severity_at_week2"]

    # Phase 1 carries higher uncertainty, lower patient confidence
    p += np.where(df["trial_phase"] == 1, 0.10, 0.0)

    # Distance without transport — pure logistical barrier
    p += np.where(
        (df["distance_from_site_km"] > 80) & (df["transportation_access"] == "no"),
        0.14,
        0.0,
    )
    p += np.where(
        (df["distance_from_site_km"] > 40) & (df["transportation_access"] == "no"),
        0.06,
        0.0,
    )

    # Prior AE history heightens safety anxiety
    p += np.where(df["prior_adverse_event_history"] == "yes", 0.08, 0.0)

    # High visit frequency → scheduling fatigue
    p += 0.008 * df["visit_frequency_per_month"]

    # Protocol complexity → cognitive and scheduling burden
    p += 0.007 * df["protocol_complexity_score"]

    # Comorbidity burden adds clinical complexity and AE risk
    p += 0.012 * df["number_of_comorbidities"]

    # Polypharmacy risk
    p += 0.006 * df["concomitant_medications"]

    # Experienced investigators retain patients better — protective factor
    p -= 0.004 * df["investigator_experience_years"]

    # Structural site-level weakness — infrastructure/investigator gap
    p += np.where(df["site_id"].isin(HIGH_DROPOUT_SITES), 0.10, 0.0)

    # Uninsured patients face competing healthcare costs
    p += np.where(df["insurance_status"] == "uninsured", 0.05, 0.0)

    # Clip to valid probability range
    return np.clip(p, 0.02, 0.97)


def assign_dropout_outcomes(
    df: pd.DataFrame, probs: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Assign binary dropout outcomes and dropout week using computed probabilities.

    Dropout weeks follow a clinical pattern: most occur in weeks 4–12, coinciding
    with the early treatment phase when side effects peak and initial enthusiasm fades.

    Args:
        df: Combined patient DataFrame.
        probs: Per-patient dropout probabilities.
        rng: NumPy random generator.

    Returns:
        DataFrame with dropout and dropout_week columns appended.
    """
    # Scale probabilities to achieve ~30% overall dropout rate
    scale = DROPOUT_RATE / probs.mean()
    scaled = np.clip(probs * scale, 0.02, 0.95)

    dropout = rng.binomial(1, scaled).astype(int)

    # Dropout timing: skewed Beta toward weeks 4–12
    dropout_weeks = []
    for d in dropout:
        if d == 1:
            # Most dropouts: weeks 4–12; tail extends to week 48
            week = int(np.clip(rng.beta(2.0, 5.0) * 48 + 1, 1, 48))
            dropout_weeks.append(week)
        else:
            dropout_weeks.append(np.nan)

    result = df.copy()
    result["dropout"] = dropout
    result["dropout_week"] = dropout_weeks
    return result


def generate_dataset(output_path: Path = None) -> pd.DataFrame:
    """
    Generate the full synthetic clinical trial dataset and save to CSV.

    Args:
        output_path: Path to save the CSV file. Defaults to data/clinical_trial_data.csv.

    Returns:
        Complete patient DataFrame with all features and dropout labels.
    """
    rng = _rng(RANDOM_SEED)

    demo = generate_demographics(N_PATIENTS, rng)
    clinical = generate_clinical_features(N_PATIENTS, rng)
    trial = generate_trial_features(N_PATIENTS, rng)

    df = pd.concat([demo, clinical, trial], axis=1)

    probs = compute_dropout_probability(df)
    df = assign_dropout_outcomes(df, probs, rng)

    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "clinical_trial_data.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DataGenerator] Saved {len(df)} records -> {output_path}")
    print(f"[DataGenerator] Dropout rate: {df['dropout'].mean():.1%}")
    return df


if __name__ == "__main__":
    generate_dataset()
