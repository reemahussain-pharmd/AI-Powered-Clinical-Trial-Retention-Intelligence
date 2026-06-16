"""
Clinical Trial Dataset Quality Validator.

Validates the synthetic dataset before model training to ensure data integrity.
Mirrors the kind of data quality checks a CRO data manager would run before
database lock in a real trial setting.
"""

import pandas as pd
from pathlib import Path


VALID_SITE_IDS = {f"SITE_{str(i).zfill(2)}" for i in range(1, 9)}
AGE_MIN, AGE_MAX = 18, 80
DROPOUT_MIN, DROPOUT_MAX = 0.20, 0.40


def validate_dataset(df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Run data quality checks on the clinical trial dataset.

    Checks performed mirror pre-database-lock QC procedures in clinical data
    management: completeness, range validity, uniqueness, and domain-level
    plausibility.

    Args:
        df: Patient DataFrame to validate.
        verbose: If True, prints a formatted quality report to stdout.

    Returns:
        True if all checks pass, False if any check fails.
    """
    checks = []

    def _check(name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})

    # 1. No missing values in required columns
    required_cols = [
        "patient_id", "age", "gender", "bmi", "employment_status",
        "insurance_status", "distance_from_site_km", "transportation_access",
        "education_level", "disease_severity_score", "number_of_comorbidities",
        "baseline_pain_score", "prior_trial_participation",
        "prior_adverse_event_history", "concomitant_medications",
        "side_effect_severity_at_week2", "trial_phase",
        "visit_frequency_per_month", "protocol_complexity_score",
        "trial_duration_months", "consent_complexity_score",
        "investigator_experience_years", "site_id", "dropout",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    _check(
        "All required columns present",
        len(missing_cols) == 0,
        f"Missing: {missing_cols}" if missing_cols else "All present",
    )

    if missing_cols:
        # Cannot continue meaningful validation if columns are missing
        _print_report(checks, verbose)
        return False

    null_counts = df[required_cols].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    _check(
        "No missing values in required columns",
        len(null_cols) == 0,
        f"Nulls found: {null_cols.to_dict()}" if len(null_cols) > 0 else "None",
    )

    # 2. Age within valid trial eligibility range (18–80)
    age_violations = df[(df["age"] < AGE_MIN) | (df["age"] > AGE_MAX)]
    _check(
        f"Age within [{AGE_MIN}, {AGE_MAX}]",
        len(age_violations) == 0,
        f"{len(age_violations)} violations" if len(age_violations) > 0 else "Clean",
    )

    # 3. Dropout rate within expected 20–40% band
    dropout_rate = df["dropout"].mean()
    _check(
        f"Dropout rate in [{DROPOUT_MIN:.0%}, {DROPOUT_MAX:.0%}]",
        DROPOUT_MIN <= dropout_rate <= DROPOUT_MAX,
        f"Observed: {dropout_rate:.1%}",
    )

    # 4. No duplicate patient IDs — primary key integrity
    dup_count = df["patient_id"].duplicated().sum()
    _check(
        "No duplicate patient_ids",
        dup_count == 0,
        f"{dup_count} duplicates found" if dup_count > 0 else "Unique",
    )

    # 5. All site_ids are valid
    invalid_sites = set(df["site_id"].unique()) - VALID_SITE_IDS
    _check(
        "All site_ids valid",
        len(invalid_sites) == 0,
        f"Invalid: {invalid_sites}" if invalid_sites else "All valid",
    )

    # 6. BMI in physiologically plausible range
    bmi_violations = df[(df["bmi"] < 10) | (df["bmi"] > 60)]
    _check(
        "BMI in [10, 60]",
        len(bmi_violations) == 0,
        f"{len(bmi_violations)} violations" if len(bmi_violations) > 0 else "Clean",
    )

    # 7. Dropout binary (0/1 only)
    dropout_vals = set(df["dropout"].unique())
    _check(
        "Dropout is binary {0, 1}",
        dropout_vals.issubset({0, 1}),
        f"Values: {dropout_vals}",
    )

    _print_report(checks, verbose)
    return all(c["status"] == "PASS" for c in checks)


def _print_report(checks: list, verbose: bool) -> None:
    """Print a formatted quality report to stdout."""
    if not verbose:
        return

    passed = sum(1 for c in checks if c["status"] == "PASS")
    total = len(checks)

    print("\n" + "=" * 60)
    print("  CLINICAL TRIAL DATA QUALITY REPORT")
    print("=" * 60)
    for c in checks:
        icon = "+" if c["status"] == "PASS" else "x"
        print(f"  [{c['status']}] {icon} {c['check']}")
        if c["detail"]:
            print(f"         -> {c['detail']}")
    print("-" * 60)
    print(f"  Result: {passed}/{total} checks passed")
    overall = "DATASET VALID — READY FOR MODELLING" if passed == total else "DATASET FAILED QC — REVIEW REQUIRED"
    print(f"  {overall}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    from data_generator import generate_dataset

    df = generate_dataset()
    result = validate_dataset(df)
    print(f"Validation result: {'PASS' if result else 'FAIL'}")
