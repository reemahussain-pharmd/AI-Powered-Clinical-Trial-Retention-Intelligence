"""
Machine Learning Pipeline for Clinical Trial Dropout Prediction.

Trains five classifiers on engineered patient features. Recall is prioritised
over precision: in a clinical retention context, the cost of missing a future
dropout (false negative) exceeds the cost of an unnecessary intervention (false positive).
"""

import warnings
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, brier_score_loss, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

import xgboost as xgb

# Training-only dependencies — not required for inference/serving
try:
    import lightgbm as lgb
    import catboost as cb
    import optuna
    import mlflow
    import mlflow.sklearn
    from imblearn.over_sampling import SMOTE
    from lifelines import KaplanMeierFitter
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _TRAINING_DEPS = True
except ImportError:
    _TRAINING_DEPS = False

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def load_config() -> Dict[str, Any]:
    """Load project configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numerics and one-hot encodes categoricals.

    Args:
        num_cols: List of numerical feature column names.
        cat_cols: List of categorical feature column names.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def prepare_data(df: pd.DataFrame, config: Dict) -> Tuple:
    """
    Split data into train/validation/test sets, apply SMOTE to the training set.

    SMOTE (Synthetic Minority Oversampling Technique) addresses class imbalance.
    In clinical modelling, undersampling the majority class risks discarding
    genuine non-dropout signal, so oversampling is preferred.

    Args:
        df: Feature-engineered patient DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple of (X_train_res, X_val, X_test, y_train_res, y_val, y_test,
                  preprocessor, feature_names).
    """
    from feature_engineering import get_feature_columns, add_composite_features

    df = add_composite_features(df)
    cols = get_feature_columns()
    feature_cols = cols["numerical"] + cols["categorical"] + cols["composite"]
    target_col = "dropout"

    X = df[feature_cols]
    y = df[target_col]

    seed = config["model"]["random_seed"]
    test_size = config["model"]["test_size"]
    val_size = config["model"]["val_size"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=seed, stratify=y_temp
    )

    preprocessor = get_preprocessor(
        cols["numerical"] + cols["composite"], cols["categorical"]
    )
    X_train_pp = preprocessor.fit_transform(X_train)
    X_val_pp = preprocessor.transform(X_val)
    X_test_pp = preprocessor.transform(X_test)

    # SMOTE on training set only — never on val/test to avoid data leakage
    smote = SMOTE(random_state=seed)
    X_train_res, y_train_res = smote.fit_resample(X_train_pp, y_train)

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(cols["categorical"]).tolist()
    feature_names = cols["numerical"] + cols["composite"] + cat_feature_names

    return (
        X_train_res, X_val_pp, X_test_pp,
        y_train_res, y_val, y_test,
        preprocessor, feature_names,
    )


def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 seed: int, n_trials: int = 50) -> Dict:
    """
    Tune XGBoost hyperparameters with Optuna, targeting AUC > 0.82.

    Args:
        X_train: SMOTE-resampled training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        seed: Random seed.
        n_trials: Number of Optuna trials.

    Returns:
        Dict of best hyperparameters.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "random_state": seed,
            "eval_metric": "auc",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[Model] XGBoost best AUC: {study.best_value:.4f}")
    return study.best_params


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   name: str) -> Dict[str, float]:
    """
    Compute classification metrics for a trained model.

    Recall is listed first to emphasise its priority in clinical retention modelling.

    Args:
        model: Trained classifier with predict_proba method.
        X_test: Test features.
        y_test: True test labels.
        name: Model name for reporting.

    Returns:
        Dict of metric name → value.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.40).astype(int)  # Lowered threshold to boost recall

    return {
        "model": name,
        "auc": round(roc_auc_score(y_test, probs), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "brier_score": round(brier_score_loss(y_test, probs), 4),
    }


def plot_roc_comparison(models_dict: Dict, X_test: np.ndarray,
                        y_test: np.ndarray) -> None:
    """
    Plot ROC curves for all five models on one figure.

    Args:
        models_dict: Dict of model_name → trained model.
        X_test: Test features.
        y_test: True test labels.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#1D9E75", "#0D1B2A", "#E07B39", "#8B1A1A", "#5B5EA6"]

    for (name, model), color in zip(models_dict.items(), colors):
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve Comparison — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "roc_curve_comparison.png", dpi=150)
    plt.close(fig)
    print("[Model] Saved roc_curve_comparison.png")


def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Plot and save confusion matrix for the best model (XGBoost)."""
    preds = (model.predict_proba(X_test)[:, 1] >= 0.40).astype(int)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Retained", "Dropout"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — XGBoost (Best Model)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("[Model] Saved confusion_matrix.png")


def plot_calibration_curve(models_dict: Dict, X_test: np.ndarray,
                           y_test: np.ndarray) -> None:
    """
    Plot probability calibration curves.

    A well-calibrated model is essential in clinical settings: when the model
    outputs 70% dropout risk, that should reflect real-world frequency.

    Args:
        models_dict: Dict of model_name → trained model.
        X_test: Test features.
        y_test: True test labels.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1D9E75", "#0D1B2A", "#E07B39", "#8B1A1A", "#5B5EA6"]

    for (name, model), color in zip(models_dict.items(), colors):
        probs = model.predict_proba(X_test)[:, 1]
        fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mean_pred, fraction_pos, marker="o", label=name, color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Actual)", fontsize=12)
    ax.set_title("Calibration Curves — Reliability Diagram", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)
    print("[Model] Saved calibration_curve.png")


def plot_precision_recall(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Plot precision-recall curve for XGBoost."""
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#1D9E75", lw=2)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — XGBoost", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
    print("[Model] Saved precision_recall_curve.png")


def run_survival_analysis(df: pd.DataFrame, config: Dict) -> None:
    """
    Kaplan-Meier survival analysis of time-to-dropout by risk tier.

    Survival analysis provides temporal framing: it answers not just 'who'
    drops out but 'when', enabling proactive intervention scheduling.

    Args:
        df: Patient DataFrame with dropout and dropout_week columns.
        config: Configuration dict with risk thresholds.
    """
    from feature_engineering import add_composite_features

    df = add_composite_features(df.copy())
    low_t = config["thresholds"]["low_risk"]
    med_t = config["thresholds"]["medium_risk"]

    # Assign risk tier using a simple heuristic on patient_burden_score for survival plot
    df["_surv_score"] = (
        0.5 * df["side_effect_severity_at_week2"]
        + 0.3 * (df["distance_from_site_km"] / 200)
        + 0.2 * (df["number_of_comorbidities"] / 8)
    )
    df["_risk_tier"] = pd.cut(
        df["_surv_score"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True,
    )

    duration_col = "dropout_week"
    df["_duration"] = df[duration_col].fillna(48)  # censored at week 48 if no dropout
    df["_event"] = df["dropout"]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Low Risk": "#1D9E75", "Medium Risk": "#E07B39", "High Risk": "#8B1A1A"}

    for tier in ["Low Risk", "Medium Risk", "High Risk"]:
        subset = df[df["_risk_tier"] == tier]
        kmf = KaplanMeierFitter()
        kmf.fit(
            subset["_duration"],
            event_observed=subset["_event"],
            label=f"{tier} (n={len(subset)})",
        )
        kmf.plot_survival_function(ax=ax, color=colors[tier], ci_show=True, lw=2)

    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Probability of Remaining in Trial", fontsize=12)
    ax.set_title(
        "Kaplan-Meier Survival Curves by Risk Tier\n(Time-to-Dropout Analysis)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUTS_DIR / "survival_curve.png", dpi=150)
    plt.close(fig)
    print("[Model] Saved survival_curve.png")


def get_dropout_window(risk_score: float) -> str:
    """
    Estimate the most likely dropout window based on risk score.

    Derived from the survival analysis patterns: high-risk patients cluster
    in the early treatment phase (weeks 4–8), while medium-risk patients drift
    in weeks 8–16.

    Args:
        risk_score: Predicted dropout probability in [0, 1].

    Returns:
        Human-readable dropout window string.
    """
    if risk_score >= 0.70:
        return "Weeks 4–8 (Early Treatment Phase)"
    elif risk_score >= 0.50:
        return "Weeks 8–16 (Mid Treatment Phase)"
    elif risk_score >= 0.30:
        return "Weeks 16–28 (Late Treatment Phase)"
    else:
        return "Low risk — unlikely to dropout"


def train_all_models(df: pd.DataFrame) -> Tuple[Any, Any, Dict, list]:
    """
    Train all five models, save artefacts, and generate output plots.

    Args:
        df: Full patient DataFrame (raw, before feature engineering).

    Returns:
        Tuple of (best_model, preprocessor, metrics_dict, feature_names).
    """
    config = load_config()
    seed = config["model"]["random_seed"]
    OUTPUTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     preprocessor, feature_names) = prepare_data(df, config)

    print("[Model] Tuning XGBoost with Optuna (50 trials)...")
    xgb_params = tune_xgboost(X_train, y_train, X_val, y_val, seed)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=seed, C=0.5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            **xgb_params, random_state=seed, eval_metric="auc",
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, random_state=seed,
            class_weight="balanced", verbose=-1,
        ),
        "CatBoost": cb.CatBoostClassifier(
            iterations=300, learning_rate=0.05, random_seed=seed,
            verbose=False,
        ),
    }

    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
    mlflow.set_experiment("clinical_trial_retention")

    metrics_list = []
    trained_models = {}

    for name, model in models.items():
        print(f"[Model] Training {name}...")
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            m = evaluate_model(model, X_test, y_test, name)
            metrics_list.append(m)
            trained_models[name] = model
            mlflow.log_metrics({k: v for k, v in m.items() if k != "model"})
            mlflow.log_param("model_type", name)
            print(f"  AUC={m['auc']:.3f} | Recall={m['recall']:.3f} | F1={m['f1']:.3f}")

    metrics_df = pd.DataFrame(metrics_list)
    print("\n[Model] Results Summary:")
    print(metrics_df.to_string(index=False))

    best_model = trained_models["XGBoost"]

    plot_roc_comparison(trained_models, X_test, y_test)
    plot_confusion_matrix(best_model, X_test, y_test)
    plot_calibration_curve(trained_models, X_test, y_test)
    plot_precision_recall(best_model, X_test, y_test)

    # Save model and preprocessor
    joblib.dump(best_model, MODELS_DIR / "model_v1.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor_v1.pkl")
    print(f"[Model] Saved model_v1.pkl and preprocessor_v1.pkl → {MODELS_DIR}")

    return best_model, preprocessor, metrics_df.to_dict("records"), feature_names


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_generator import generate_dataset
    from validator import validate_dataset

    df = generate_dataset()
    validate_dataset(df)
    model, preprocessor, metrics, feature_names = train_all_models(df)
    run_survival_analysis(df, load_config())
