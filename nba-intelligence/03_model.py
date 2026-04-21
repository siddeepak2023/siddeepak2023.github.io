"""
03_model.py
Train HistGradientBoostingClassifier (sklearn native, no OpenMP dependency)
on 2022-23 + 2023-24; evaluate on 2024-25 holdout.
Outputs data/model.pkl and data/model_metrics.json.

Run:  python3 03_model.py
"""

import json
import logging
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, brier_score_loss,
                              log_loss, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES_CSV   = "data/features.csv"
MODEL_PKL      = "data/model.pkl"
METRICS_JSON   = "data/model_metrics.json"
MODEL_VERSION  = "hgbm_v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "home_roll_win_pct", "home_roll_pts", "home_roll_off_rtg", "home_roll_ts",
    "home_roll_pace", "home_roll_tov", "home_roll_oreb", "home_roll_reb",
    "home_roll_ast", "home_roll_fg3_rate", "home_rest_days", "home_season_win_pct",
    "away_roll_win_pct", "away_roll_pts", "away_roll_off_rtg", "away_roll_ts",
    "away_roll_pace", "away_roll_tov", "away_roll_oreb", "away_roll_reb",
    "away_roll_ast", "away_roll_fg3_rate", "away_rest_days", "away_season_win_pct",
    "diff_roll_win_pct", "diff_roll_pts", "diff_roll_off_rtg", "diff_roll_ts",
    "diff_roll_pace", "diff_roll_tov", "diff_roll_oreb", "diff_roll_reb",
    "diff_roll_ast", "diff_roll_fg3_rate", "diff_rest_days", "diff_season_win_pct",
]
LABEL_COL = "home_win"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def make_estimator(n_estimators=300):
    return HistGradientBoostingClassifier(
        max_iter=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42,
    )


def evaluate(model, X, y, label: str) -> dict:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy":  round(float(accuracy_score(y, pred)), 4),
        "auc":       round(float(roc_auc_score(y, prob)), 4),
        "log_loss":  round(float(log_loss(y, prob)), 4),
        "brier":     round(float(brier_score_loss(y, prob)), 4),
        "n":         int(len(y)),
    }
    log.info("  %s  acc=%.3f  AUC=%.3f  log_loss=%.4f  brier=%.4f  (n=%d)",
             label, metrics["accuracy"], metrics["auc"],
             metrics["log_loss"], metrics["brier"], metrics["n"])
    return metrics


def main():
    df = load_data()
    log.info("Loaded %d game rows", len(df))

    train_df = df[df["season"].isin(["2022-23", "2023-24"])].sort_values("game_date")
    test_df  = df[df["season"] == "2024-25"].sort_values("game_date")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[LABEL_COL].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[LABEL_COL].values

    log.info("Train: %d  |  Test (2024-25): %d", len(X_train), len(X_test))

    # ── TimeSeriesSplit cross-validation ─────────────────────────────────────
    log.info("Cross-validating with TimeSeriesSplit (5 folds) …")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_aucs = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        m = make_estimator(300)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        prob = m.predict_proba(X_train[val_idx])[:, 1]
        auc  = roc_auc_score(y_train[val_idx], prob)
        cv_aucs.append(auc)
        log.info("  Fold %d  AUC=%.4f", fold, auc)

    log.info("CV AUC: %.4f ± %.4f", np.mean(cv_aucs), np.std(cv_aucs))

    # ── Final model: fit on full train + calibrate ────────────────────────────
    log.info("Fitting final model …")
    base = make_estimator(400)
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")
    model.fit(X_train, y_train)
    log.info("Calibrated model fitted.")

    train_metrics = evaluate(model, X_train, y_train, "TRAIN")
    test_metrics  = evaluate(model, X_test,  y_test,  "TEST ")

    # ── Permutation importance on test set ───────────────────────────────────
    log.info("Computing permutation importance (10 repeats) …")
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=1,
    )
    importance_pairs = sorted(
        zip(FEATURE_COLS, perm.importances_mean.tolist()),
        key=lambda x: x[1], reverse=True,
    )
    log.info("Top 10 features by permutation importance:")
    for feat, val in importance_pairs[:10]:
        log.info("  %-35s %.4f", feat, val)

    # ── Persist ──────────────────────────────────────────────────────────────
    with open(MODEL_PKL, "wb") as f:
        pickle.dump({
            "model":         model,
            "feature_cols":  FEATURE_COLS,
            "model_version": MODEL_VERSION,
        }, f)
    log.info("Model saved → %s", MODEL_PKL)

    metrics_out = {
        "model_version":    MODEL_VERSION,
        "feature_cols":     FEATURE_COLS,
        "cv_auc_mean":      round(float(np.mean(cv_aucs)), 4),
        "cv_auc_std":       round(float(np.std(cv_aucs)), 4),
        "train":            train_metrics,
        "test":             test_metrics,
        "feature_importance": [
            {"feature": f, "importance": round(v, 5)}
            for f, v in importance_pairs
        ],
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_out, f, indent=2)
    log.info("Metrics saved → %s", METRICS_JSON)
    log.info("✓  Model training complete.")


if __name__ == "__main__":
    main()
