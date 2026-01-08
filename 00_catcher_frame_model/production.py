#!/usr/bin/env python
"""
production.py

Loads a pre-trained framing model (framing_model.pkl),
applies it to new_data.csv, and produces:

1) pitch_level_predictions.csv
   - PITCH_ID
   - IS_STRIKE (1 = called strike, 0 = called ball)
   - CS_PROB (model's called-strike probability)

2) new_output.csv
   Aggregated by catcher and year:
   - CatcherID
   - Year
   - Opportunities
   - ActualCalledStrikes
   - ExpectedCalledStrikes
   - CalledStrikesAdded
   - CalledStrikesAdded_per_100
"""

#!/usr/bin/env python
"""
production.py
"""

# ---------------------------
# Install required libraries
# ---------------------------
import sys
import subprocess
import importlib

def ensure_package(module_name: str, pip_spec: str | None = None) -> None:
    """
    If `module_name` cannot be imported, install it with pip using `pip_spec`.
    If `pip_spec` is None, use `module_name` as the pip package name.
    """
    if pip_spec is None:
        pip_spec = module_name

    try:
        importlib.import_module(module_name)
        # print(f"[INFO] {module_name} already installed.")
    except ImportError:
        print(f"[INFO] {module_name} not found. Installing {pip_spec} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_spec],
            check=True,
        )

# non-base libraries used in this script
ensure_package("lightgbm", "lightgbm")
ensure_package("pandas",   "pandas")
ensure_package("numpy",    "numpy")
ensure_package("pymc",     "pymc")
ensure_package("arviz",    "arviz")
ensure_package("scikit-learn", "scikit-learn")

# ---------------------------
# normal imports below
# ---------------------------
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az



# ----------------------------------------------------------------------
# Optional: feature engineering helper
# (Currently not required by the trained model, but kept for extension.)
# ----------------------------------------------------------------------
def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add optional engineered features for framing analysis.

    This function does NOT affect which columns are used for prediction;
    the model will still only consume the feature_cols stored in the
    saved model. Extra columns are safe but unused unless they are in feature_cols.
    """
    df = df.copy()

    # Basic count-related features
    df["PITCH_COUNT"] = df["BALLS"] + df["STRIKES"]
    df["IS_TWO_STRIKES"] = (df["STRIKES"] == 2).astype(int)
    df["IS_FULL_COUNT"] = ((df["BALLS"] == 3) & (df["STRIKES"] == 2)).astype(int)

    df["IS_HIGH_LEVERAGE"] = (
        (df["RUNNERS_ON"] > 0) |
        (df["IS_FULL_COUNT"] == 1) |
        (df["OUTS"] == 2)
    ).astype(int)

    # Base runner count from BASE_CONFIGURATION if available
    if "BASE_CONFIGURATION" in df.columns:
        if df["BASE_CONFIGURATION"].dtype == object:
            df["BASE_RUNNER_COUNT"] = (
                df["BASE_CONFIGURATION"].fillna("").astype(str).str.len()
            )
        else:
            df["BASE_RUNNER_COUNT"] = df["BASE_CONFIGURATION"]

    # Zone geometry normalization
    df["TOP_ZONE"] = df["TOP_ZONE"].replace(0, np.nan)
    df["BOT_ZONE"] = df["BOT_ZONE"].replace(0, np.nan)

    zone_mid = (df["TOP_ZONE"] + df["BOT_ZONE"]) / 2.0
    zone_half_height = (df["TOP_ZONE"] - df["BOT_ZONE"]) / 2.0

    df["PLATE_Z_FROM_MID"] = df["PLATELOCHEIGHT"] - zone_mid
    df["PLATE_Z_NORM"] = (df["PLATE_Z_FROM_MID"] / zone_half_height).fillna(0.0).clip(-5, 5)

    PLATE_HALF_WIDTH = 0.7083  # ~8.5 inches in feet
    df["PLATE_X_NORM"] = (df["PLATELOCSIDE"] / PLATE_HALF_WIDTH).fillna(0.0).clip(-5, 5)

    df["PLATE_Z_NORM_SQ"] = df["PLATE_Z_NORM"] ** 2
    df["PLATE_X_NORM_SQ"] = df["PLATE_X_NORM"] ** 2

    # Edge / shadow zone flags
    EDGE_LOWER = 0.8
    EDGE_UPPER = 1.2

    df["IN_ZONE_VERT"] = (df["PLATE_Z_NORM"].abs() <= 1.0)
    df["IN_ZONE_HORZ"] = (df["PLATE_X_NORM"].abs() <= 1.0)
    df["IN_ZONE"] = (df["IN_ZONE_VERT"] & df["IN_ZONE_HORZ"]).astype(int)

    df["IS_EDGE_VERT"] = (
        (df["PLATE_Z_NORM"].abs() >= EDGE_LOWER) &
        (df["PLATE_Z_NORM"].abs() <= EDGE_UPPER)
    )
    df["IS_EDGE_HORZ"] = (
        (df["PLATE_X_NORM"].abs() >= EDGE_LOWER) &
        (df["PLATE_X_NORM"].abs() <= EDGE_UPPER)
    )

    df["IS_SHADOW_PITCH"] = (df["IS_EDGE_VERT"] | df["IS_EDGE_HORZ"]).astype(int)

    # High / low flags
    df["IS_LOW_PITCH"] = (df["PLATE_Z_NORM"] <= -0.5).astype(int)
    df["IS_HIGH_PITCH"] = (df["PLATE_Z_NORM"] >= 0.5).astype(int)

    # Inside / outside relative to batter
    is_rhb = (df["BATTERSIDE"] == "R")
    is_lhb = (df["BATTERSIDE"] == "L")

    inside_mask = (is_rhb & (df["PLATELOCSIDE"] < 0)) | (is_lhb & (df["PLATELOCSIDE"] > 0))
    df["IS_INSIDE"] = inside_mask.astype(int)
    df["IS_OUTSIDE"] = 1 - df["IS_INSIDE"]

    # Glove side / arm side relative to catcher
    df["IS_GLOVE_SIDE"] = (df["PLATELOCSIDE"] > 0).astype(int)
    df["IS_ARM_SIDE"] = 1 - df["IS_GLOVE_SIDE"]

    # Interaction examples
    df["EDGE_TWO_STRIKES"] = df["IS_SHADOW_PITCH"] * df["IS_TWO_STRIKES"]
    df["EDGE_HIGH_LEVERAGE"] = df["IS_SHADOW_PITCH"] * df["IS_HIGH_LEVERAGE"]
    df["VERT_BREAK_LOW"] = df["INDUCEDVERTBREAK"] * df["IS_LOW_PITCH"]
    df["VERT_BREAK_HIGH"] = df["INDUCEDVERTBREAK"] * df["IS_HIGH_PITCH"]

    # Pitch-type interaction buckets (string-based)
    if "AUTOPITCHTYPE" in df.columns:
        rounded_z = df["PLATE_Z_NORM"].round(1).astype(str)
        rounded_x = df["PLATE_X_NORM"].round(1).astype(str)
        pitch_str = df["AUTOPITCHTYPE"].astype(str)

        df["PITCHTYPE_Z_NORM"] = pitch_str + "_" + rounded_z
        df["PITCHTYPE_X_NORM"] = pitch_str + "_" + rounded_x

    return df


# ----------------------------------------------------------------------
# Model and data loading helpers
# ----------------------------------------------------------------------
def load_model(model_path: Path):
    """Load the trained LightGBM model and metadata."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    feature_cols = saved["feature_cols"]
    cat_cols = saved["cat_cols"]

    print(f"Loaded model from: {model_path}")
    print(f"Feature columns expected by model ({len(feature_cols)}): {feature_cols}")
    print(f"Categorical columns expected by model: {cat_cols}")

    return model, feature_cols, cat_cols


def load_new_data(data_path: Path) -> pd.DataFrame:
    """Load the new pitch-level dataset for scoring."""
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    print(f"Loading new data from: {data_path}")
    df = pd.read_csv(data_path)

    required_cols = ["PITCHCALL", "PITCH_ID", "CATCHER_ID"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in new_data.csv")

    # Build binary target: 1 = called strike, 0 = called ball
    df["IS_STRIKE"] = (df["PITCHCALL"] == "StrikeCalled").astype(int)

    return df


def detect_year_column(df: pd.DataFrame) -> str:
    """Detect which column to use as year for aggregation."""
    if "GAME_YEAR" in df.columns:
        return "GAME_YEAR"
    if "YEAR" in df.columns:
        return "YEAR"
    raise ValueError("Need a YEAR or GAME_YEAR column in new_data.csv for aggregation.")


# ----------------------------------------------------------------------
# Prediction and aggregation
# ----------------------------------------------------------------------
def prepare_for_prediction(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str]
) -> pd.DataFrame:
    """
    Ensure categorical types and feature presence before prediction.
    This function may optionally add engineered features, but the model
    will only consume the columns in feature_cols.
    """
    df = df.copy()

    # Optional: add engineered features (safe because we still subset by feature_cols)
    # df = add_feature_engineering(df)

    # Ensure categorical dtypes match training
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Make sure all expected feature columns are present
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(
            "The following feature columns expected by the model "
            f"are missing in new_data.csv: {missing_feats}"
        )

    return df


def predict_called_strike_probabilities(
    df: pd.DataFrame,
    model,
    feature_cols: list[str]
) -> pd.DataFrame:
    """Add CS_PROB column with called-strike probability from the model."""
    X_new = df[feature_cols]
    print("Predicting called-strike probabilities on new data...")
    df = df.copy()
    df["CS_PROB"] = model.predict_proba(X_new)[:, 1]
    return df


def save_pitch_level_predictions(df: pd.DataFrame, out_path: Path) -> None:
    """Save pitch-level predictions to CSV."""
    cols = ["PITCH_ID", "IS_STRIKE", "CS_PROB"]
    df[cols].to_csv(out_path, index=False)
    print(f"Saved pitch-level predictions to: {out_path}")

def aggregate_catcher_year_metrics(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    """
    Aggregate pitch-level data to catcher-year framing metrics and fit a
    Bayesian hierarchical Normal-Normal model using PyMC to obtain
    partially-pooled (shrunk) Called Strikes Added for each catcher-year.

    Final output columns match the spec:
        CatcherID
        Year
        Opportunities
        ActualCalledStrikes
        CalledStrikesAdded
        CalledStrikesAdded_per_100
    """

    df = df.copy()

    # Per-pitch residual: actual - baseline probability
    df["residual"] = df["IS_STRIKE"] - df["CS_PROB"]

    # Catcher-year aggregation
    group_cols = ["CATCHER_ID", year_col]
    grouped = df.groupby(group_cols).agg(
        OPPORTUNITIES=("residual", "size"),
        ACTUAL_CALLED_STRIKES=("IS_STRIKE", "sum"),
        ExpectedCalledStrikes=("CS_PROB", "sum"),
        mean_resid=("residual", "mean"),
    ).reset_index()

    # Data for the hierarchical model
    n = grouped["OPPORTUNITIES"].to_numpy().astype(float)
    r_bar = grouped["mean_resid"].to_numpy().astype(float)
    G = len(grouped)

    # If only one group, no pooling is possible
    if G == 1:
        theta_post_mean = r_bar.copy()
    else:
        with pm.Model() as model:
            # Hyperpriors (you can tweak these if needed)
            tau = pm.HalfNormal("tau", sigma=0.2)
            sigma = pm.HalfNormal("sigma", sigma=1.0)

            # True per-pitch effect for each catcher-year
            theta = pm.Normal("theta", mu=0.0, sigma=tau, shape=G)

            # Observed mean residuals r_bar_i ~ N(theta_i, sigma / sqrt(n_i))
            mu_obs = theta
            sigma_obs = sigma / pm.math.sqrt(n)

            r_bar_obs = pm.Normal(
                "r_bar_obs",
                mu=mu_obs,
                sigma=sigma_obs,
                observed=r_bar,
            )

            idata = pm.sample(
                draws=1500,
                tune=1000,
                target_accept=0.95,
                chains=2,
                cores=1,
                progressbar=False,
            )

        # idata.posterior["theta"]: shape (chains, draws, G)
        theta_arr = idata.posterior["theta"].values
        chains, draws, G2 = theta_arr.shape
        assert G2 == G

        # Flatten chains and draws -> (samples, G)
        theta_samples = theta_arr.reshape(chains * draws, G)

        # Posterior mean per catcher-year: shape (G,)
        theta_post_mean = theta_samples.mean(axis=0)

    # Convert per-pitch effect back to counts and per-100 rates
    grouped["CALLED_STRIKES_ADDED"] = theta_post_mean * n
    grouped["CALLED_STRIKES_ADDED_PER_100"] = theta_post_mean * 100.0

    # Rounding for readability
    grouped["ACTUAL_CALLED_STRIKES"] = grouped["ACTUAL_CALLED_STRIKES"].round(3)
    grouped["CALLED_STRIKES_ADDED"] = grouped["CALLED_STRIKES_ADDED"].round(3)
    grouped["CALLED_STRIKES_ADDED_PER_100"] = grouped["CALLED_STRIKES_ADDED_PER_100"].round(3)

    # Rename ID/year columns
    grouped.rename(
        columns={
            "CATCHER_ID": "CATCHER_ID",
            year_col: "YEAR",
        },
        inplace=True,
    )

    # Keep only the required columns
    grouped = grouped[
        [
            "CATCHER_ID",
            "YEAR",
            "OPPORTUNITIES",
            "ACTUAL_CALLED_STRIKES",
            "CALLED_STRIKES_ADDED",
            "CALLED_STRIKES_ADDED_PER_100",
        ]
    ]

    return grouped



def save_catcher_metrics(agg: pd.DataFrame, out_path: Path) -> None:
    """Save catcher-year framing metrics to CSV."""
    agg.to_csv(out_path, index=False)
    print(f"Saved catcher-year framing output to: {out_path}")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    base_dir = Path(__file__).resolve().parent

    # You can parameterize these or keep them hard-coded
    model_path = base_dir / "framing_model_20251203_055049.pkl"
    data_path = base_dir / "new_data.csv"

    # Load model and data
    model, feature_cols, cat_cols = load_model(model_path)
    df = load_new_data(data_path)
    year_col = detect_year_column(df)

    # Prepare dataframe for prediction (ensure columns / dtypes)
    df = prepare_for_prediction(df, feature_cols, cat_cols)

    # Predict CS_PROB
    df = predict_called_strike_probabilities(df, model, feature_cols)

    # 1) Pitch-level CSV
    pitch_out_path = base_dir / "pitch_level_predictions.csv"
    save_pitch_level_predictions(df, pitch_out_path)

    # 2) Catcher-year framing metrics
    catcher_metrics = aggregate_catcher_year_metrics(df, year_col)
    catcher_out_path = base_dir / "new_output.csv"
    save_catcher_metrics(catcher_metrics, catcher_out_path)


if __name__ == "__main__":
    main()
