#!/usr/bin/env python
"""
build_model.py  — baseline demo with K-fold CV, modularized

Trains a LightGBM model to predict IS_STRIKE using raw columns
(no feature engineering) and evaluates with Stratified K-fold CV.

Usage:
    python build_model.py \
        --train_csv ML_TAKES_ENCODED.csv \
        --model_out framing_model.pkl \
        --n_splits 5
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from datetime import datetime
import logging
import os
import csv


import sys

class Tee(object):
    def __init__(self, logfile):
        self.log = open(logfile, "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.stdout.write(data)
        self.log.write(data)

    def flush(self):
        self.stdout.flush()
        self.log.flush()

    def close(self):
        self.log.close()



# -------------------------
# Argument parsing
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        type=str,
        default="train_data.csv",
        help="Path to training CSV file."
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="framing_model.pkl",
        help="Output path for the trained model pickle."
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds for StratifiedKFold."
    )
    return parser.parse_args()


# -------------------------
# Step 1: Load data
# -------------------------

def load_data(train_path: Path) -> pd.DataFrame:
    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found at: {train_path}")

    print(f"Loading training data from: {train_path}")
    df = pd.read_csv(train_path)

    if "PITCHCALL" not in df.columns:
        raise ValueError("Expected column 'PITCHCALL' not found in data.")
    if "GAME_YEAR" not in df.columns:
        # You don't strictly need GAME_YEAR for K-fold, but it's useful later.
        raise ValueError("Expected column 'GAME_YEAR' not found in data.")

    return df


# -------------------------
# Step 2: Prepare target (filter + IS_STRIKE)
# -------------------------

def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to called pitches and create IS_STRIKE target.
    """
    df = df.copy()

    valid_calls = ["BallCalled", "StrikeCalled"]
    df = df[df["PITCHCALL"].isin(valid_calls)].copy()

    df["IS_STRIKE"] = (df["PITCHCALL"] == "StrikeCalled").astype(int)

    return df


# -------------------------
# Step 3: Select features (no feature engineering)
# -------------------------

def select_features(df: pd.DataFrame):
    """
    Drop obvious ID and target columns, use the rest as features.
    Split into numeric and categorical based on dtype.
    """
    id_like_cols = [
        "PITCH_ID",
        "GAMEID",
        "CATCHER_ID",          # exclude so catcher effect is not baked into model
        "PITCHER_ID",
        "UMPIRE_ID",
        "RUNNER_ON_FIRST_ID",
        "RUNNER_ON_SECOND_ID",
        "RUNNER_ON_THIRD_ID",
        "HOME_TEAM_ID",
        "PITCHER_TEAM_ID",
        "STADIUM_ID",
    ]
    target_cols = ["PITCHCALL", "IS_STRIKE", "GAME_YEAR"]

    drop_cols = set(id_like_cols + target_cols)

    candidate_cols = [c for c in df.columns if c not in drop_cols]

    if not candidate_cols:
        raise ValueError("No candidate feature columns found after exclusions.")

    cat_cols = [
        c for c in candidate_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    num_cols = [c for c in candidate_cols if c not in cat_cols]

    # Cast categoricals
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype("category")

    feature_cols = num_cols + cat_cols

    print(f"Using {len(feature_cols)} features (no feature engineering).")
    print("Numeric features:", num_cols)
    print("Categorical features:", cat_cols)

    return df, feature_cols, cat_cols, num_cols

# -------------------------
# Step 3.1: Feature Engineer
# -------------------------

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["IS_STRIKE"] = (df["PITCHCALL"] == "StrikeCalled").astype(int)

    df["PITCH_COUNT"] = df["BALLS"] + df["STRIKES"]
    df["IS_TWO_STRIKES"] = (df["STRIKES"] == 2).astype(int)
    df["IS_FULL_COUNT"] = ((df["BALLS"] == 3) & (df["STRIKES"] == 2)).astype(int)

    df["IS_HIGH_LEVERAGE"] = (
        (df["RUNNERS_ON"] > 0) | 
        (df["IS_FULL_COUNT"] == 1) |
        (df["OUTS"] == 2)
    ).astype(int)

    if "BASE_CONFIGURATION" in df.columns:
        if df["BASE_CONFIGURATION"].dtype == object:
            df["BASE_RUNNER_COUNT"] = df["BASE_CONFIGURATION"].fillna("").astype(str).str.len()
        else:
            df["BASE_RUNNER_COUNT"] = df["BASE_CONFIGURATION"]

    
    df['TOP_ZONE'] = df['TOP_ZONE'].replace(0, np.nan) 
    df['BOT_ZONE'] = df['BOT_ZONE'].replace(0, np.nan)
    
    zone_mid = (df["TOP_ZONE"] + df["BOT_ZONE"]) / 2.0
    zone_half_height = (df["TOP_ZONE"] - df["BOT_ZONE"]) / 2.0
    
    df["PLATE_Z_FROM_MID"] = df["PLATELOCHEIGHT"] - zone_mid
    
    df["PLATE_Z_NORM"] = df["PLATE_Z_FROM_MID"] / zone_half_height
    df["PLATE_Z_NORM"] = df["PLATE_Z_NORM"].fillna(0.0).clip(-5, 5) 
    
    PLATE_HALF_WIDTH = 0.7083 
    df["PLATE_X_NORM"] = df["PLATELOCSIDE"] / PLATE_HALF_WIDTH
    df["PLATE_X_NORM"] = df["PLATE_X_NORM"].fillna(0.0).clip(-5, 5) 
    
    df["PLATE_Z_NORM_SQ"] = df["PLATE_Z_NORM"] ** 2
    df["PLATE_X_NORM_SQ"] = df["PLATE_X_NORM"] ** 2
    
    
    EDGE_LOWER = 0.8
    EDGE_UPPER = 1.2
    
    df["IN_ZONE_VERT"] = df["PLATE_Z_NORM"].abs() <= 1.0
    df["IN_ZONE_HORZ"] = df["PLATE_X_NORM"].abs() <= 1.0
    
    df["IN_ZONE"] = (df["IN_ZONE_VERT"] & df["IN_ZONE_HORZ"]).astype(int)

    df["IS_EDGE_VERT"] = (df["PLATE_Z_NORM"].abs() >= EDGE_LOWER) & (df["PLATE_Z_NORM"].abs() <= EDGE_UPPER)

    df["IS_EDGE_HORZ"] = (df["PLATE_X_NORM"].abs() >= EDGE_LOWER) & (df["PLATE_X_NORM"].abs() <= EDGE_UPPER)

    df["IS_SHADOW_PITCH"] = (df["IS_EDGE_VERT"] | df["IS_EDGE_HORZ"]).astype(int)

    
    df["IS_LOW_PITCH"] = (df["PLATE_Z_NORM"] <= -0.5).astype(int) # 相对中位偏低
    df["IS_HIGH_PITCH"] = (df["PLATE_Z_NORM"] >= 0.5).astype(int) # 相对中位偏高

    is_rhb = (df["BATTERSIDE"] == "R")
    is_lhb = (df["BATTERSIDE"] == "L")
    
    df["IS_INSIDE"] = (is_rhb & (df["PLATELOCSIDE"] < 0)) | (is_lhb & (df["PLATELOCSIDE"] > 0)).astype(int)
    df["IS_OUTSIDE"] = 1 - df["IS_INSIDE"]
    
    df["IS_GLOVE_SIDE"] = (df["PLATELOCSIDE"] > 0).astype(int)
    df["IS_ARM_SIDE"] = 1 - df["IS_GLOVE_SIDE"]


    df["EDGE_TWO_STRIKES"] = df["IS_SHADOW_PITCH"] * df["IS_TWO_STRIKES"]

    df["EDGE_HIGH_LEVERAGE"] = df["IS_SHADOW_PITCH"] * df["IS_HIGH_LEVERAGE"]

    df["VERT_BREAK_LOW"] = df["INDUCEDVERTBREAK"] * df["IS_LOW_PITCH"]

    df["VERT_BREAK_HIGH"] = df["INDUCEDVERTBREAK"] * df["IS_HIGH_PITCH"]
    
    if "AUTOPITCHTYPE" in df.columns:
        df["PITCHTYPE_Z_NORM"] = df["AUTOPITCHTYPE"].astype(str) + "_" + df["PLATE_Z_NORM"].round(1).astype(str)
        df["PITCHTYPE_X_NORM"] = df["AUTOPITCHTYPE"].astype(str) + "_" + df["PLATE_X_NORM"].round(1).astype(str)


    return df


# -------------------------
# Step 4: K-fold CV
# -------------------------
def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list,
    cat_cols: list,
    n_splits: int = 5,
):
    """
    Run Stratified K-fold CV, print fold metrics,
    and return suggested n_estimators for final model.
    Tracks AUC, LogLoss, and RMSE across folds.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    cat_idx = [feature_cols.index(c) for c in cat_cols]

    fold_aucs = []
    fold_lls = []
    fold_rmses = []
    best_iterations = []

    print(f"Running {n_splits}-fold StratifiedKFold CV...")

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42 + fold,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=["auc", "binary_logloss"],
            categorical_feature=cat_idx,
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(period=50),
            ],
        )

        y_valid_pred = model.predict_proba(X_valid)[:, 1]

        # Metrics
        auc = roc_auc_score(y_valid, y_valid_pred)
        ll = log_loss(y_valid, y_valid_pred)
        rmse = np.sqrt(np.mean((y_valid_pred - y_valid) ** 2))

        fold_aucs.append(auc)
        fold_lls.append(ll)
        fold_rmses.append(rmse)

        best_iter = getattr(model, "best_iteration_", None)
        # Some versions can return 0 or None; ignore those
        if best_iter is not None and best_iter > 0:
            best_iterations.append(best_iter)

        print(f"Fold {fold} AUC:      {auc:.4f}")
        print(f"Fold {fold} LogLoss: {ll:.4f}")
        print(f"Fold {fold} RMSE:    {rmse:.4f}")
        print(f"Fold {fold} best_iteration_: {best_iter}")

    print("\n=== CV Summary ===")
    print(f"Mean AUC:      {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Mean LogLoss:  {np.mean(fold_lls):.4f} ± {np.std(fold_lls):.4f}")
    print(f"Mean RMSE:     {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")

    # Decide final n_estimators
    default_n_estimators = 500
    if best_iterations:
        avg_best_iter = int(np.mean(best_iterations))
        # Extra safety: require at least, say, 50 trees
        n_estimators_final = max(avg_best_iter, 50)
        print(f"Average best_iteration_ across folds (filtered): {avg_best_iter}")
    else:
        print("No valid best_iteration_ recorded, using default "
              f"n_estimators={default_n_estimators}.")
        n_estimators_final = default_n_estimators

    return n_estimators_final


# -------------------------
# Step 5: Train final model on full data
# -------------------------

def train_full_model(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list,
    cat_cols: list,
    n_estimators: int,
):
    cat_idx = [feature_cols.index(c) for c in cat_cols]

    # Safety: ensure positive n_estimators
    if n_estimators is None or n_estimators <= 0:
        n_estimators = 500

    print(f"\nTraining final model on full data with n_estimators={n_estimators}...")
    final_model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(
        X,
        y,
        categorical_feature=cat_idx,
    )

    return final_model


# -------------------------
# Step 6: Save model
# -------------------------

def save_model(model, feature_cols, cat_cols, out_path: Path):
    to_save = {
        "model": model,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
    }

    with open(out_path, "wb") as f:
        pickle.dump(to_save, f)

    print(f"\nSaved final model to: {out_path.resolve()}")


# -------------------------
# Timestamp
# -------------------------

def get_timestamp():
    # Example: 20251201_163045
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# -------------------------
# Orchestrator
# -------------------------

def main():
    args = parse_args()

    # ---- 1. Timestamp ----
    timestamp = get_timestamp()

    # ---- 2. Prepare log paths ----
    model_out_path = Path(args.model_out)
    base_dir = model_out_path.parent if model_out_path.parent != Path('') else Path.cwd()
    model_stem = model_out_path.stem
    model_suffix = model_out_path.suffix or ".pkl"

    final_model_path = base_dir / f"{model_stem}_{timestamp}{model_suffix}"
    log_path = base_dir / f"{model_stem}_{timestamp}.log"

    # ---- 3. Begin capturing ALL terminal output ----
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print("=== Training run started ===")
    print(f"Args: {args}")
    print(f"Model will be saved to: {final_model_path}")
    print(f"Log file: {log_path}")

    # ---- 4. Load & preprocess ----
    print(f"Loading training data from: {args.train_csv}")
    df_raw = load_data(Path(args.train_csv))

    print("Preparing target...")
    df = prepare_target(df_raw)

    # print("Applying feature engineering...")
    # df = add_feature_engineering(df)

    print("Selecting features...")
    df, feature_cols, cat_cols, num_cols = select_features(df)

    X = df[feature_cols]
    y = df["IS_STRIKE"].values

    # ---- 5. CV ----
    print("Running cross-validation...")
    n_estimators_final = run_cv(
        X=X,
        y=y,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        n_splits=args.n_splits,
    )
    print(f"Best n_estimators from CV: {n_estimators_final}")

    # ---- 6. Final training ----
    print("Training final model...")
    final_model = train_full_model(
        X=X,
        y=y,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        n_estimators=n_estimators_final,
    )
    print("Final model trained.")

    # ---- 7. Save model ----
    print(f"Saving model to: {final_model_path}")
    save_model(final_model, feature_cols, cat_cols, final_model_path)
    print("Model saved successfully.")

    print("=== Training run finished ===")

    # ---- 8. Append run record ----
    runs_log = base_dir / "training_runs_log.csv"
    try:
        file_exists = runs_log.exists()
        with open(runs_log, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "model_path", "train_csv", "n_estimators"])
            writer.writerow([timestamp, str(final_model_path), str(args.train_csv), n_estimators_final])
    except Exception as e:
        print(f"[Warning] Failed to update training_runs_log.csv: {e}")

    # ---- 9. Close Tee ----
    tee.close()



if __name__ == "__main__":
    main()
