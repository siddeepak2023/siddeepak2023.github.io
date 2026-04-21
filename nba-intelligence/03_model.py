"""
03_model.py
Train on 2016-17 → 2023-24 (8 seasons), evaluate on 2024-25 holdout.
Improvements over v1:
  - Expanded feature set: Elo, def_rtg, net_rtg, B2B, multi-window, EWMA
  - Optuna hyperparameter optimisation (50 trials, AUC objective)
  - TimeSeriesSplit(7) with more folds for better CV stability
  - Bubble-season sample weight (0.6) to down-weight anomalous 2020 games

Run:  python3 03_model.py
"""

import json
import logging
import pickle
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, brier_score_loss,
                              log_loss, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_CSV  = "data/features.csv"
MODEL_PKL     = "data/model.pkl"
METRICS_JSON  = "data/model_metrics.json"
MODEL_VERSION = "hgbm_v2_elo_multiwindow"
TRAIN_SEASONS = ["2016-17","2017-18","2018-19","2019-20",
                 "2020-21","2021-22","2022-23","2023-24"]
TEST_SEASON   = "2024-25"
OPTUNA_TRIALS = 60
BUBBLE_WEIGHT = 0.6   # down-weight 2019-20 bubble restart games

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    # ── 10-game rolling ──
    "home_roll_win_pct", "home_roll_pts", "home_roll_off_rtg", "home_roll_def_rtg",
    "home_roll_net_rtg", "home_roll_ts", "home_roll_pace", "home_roll_tov",
    "home_roll_oreb", "home_roll_reb", "home_roll_ast", "home_roll_fg3_rate",
    "away_roll_win_pct", "away_roll_pts", "away_roll_off_rtg", "away_roll_def_rtg",
    "away_roll_net_rtg", "away_roll_ts", "away_roll_pace", "away_roll_tov",
    "away_roll_oreb", "away_roll_reb", "away_roll_ast", "away_roll_fg3_rate",
    # ── 5-game hot streak ──
    "home_roll5_win_pct", "home_roll5_net_rtg", "home_roll5_pts",
    "away_roll5_win_pct", "away_roll5_net_rtg", "away_roll5_pts",
    # ── 20-game trend ──
    "home_roll20_win_pct", "home_roll20_net_rtg",
    "away_roll20_win_pct", "away_roll20_net_rtg",
    # ── EWMA ──
    "home_ewm_win_pct", "home_ewm_net_rtg", "home_ewm_pts",
    "away_ewm_win_pct", "away_ewm_net_rtg", "away_ewm_pts",
    # ── Elo ──
    "home_pre_elo", "away_pre_elo", "elo_diff",
    # ── Context ──
    "home_rest_days", "away_rest_days", "home_is_b2b", "away_is_b2b",
    "home_season_win_pct", "away_season_win_pct",
    # ── Differentials ──
    "diff_roll_win_pct", "diff_roll_pts", "diff_roll_off_rtg", "diff_roll_def_rtg",
    "diff_roll_net_rtg", "diff_roll_ts", "diff_roll_pace", "diff_roll_tov",
    "diff_roll_oreb", "diff_roll_reb", "diff_roll_ast", "diff_roll_fg3_rate",
    "diff_roll5_win_pct", "diff_roll5_net_rtg",
    "diff_roll20_win_pct", "diff_roll20_net_rtg",
    "diff_ewm_win_pct", "diff_ewm_net_rtg",
    "diff_rest_days", "diff_season_win_pct", "diff_is_b2b", "diff_elo",
]
LABEL_COL = "home_win"


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Keep only columns that exist in the CSV
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Missing %d features (will be skipped): %s", len(missing), missing[:5])
    return df, available


def make_estimator(params):
    return HistGradientBoostingClassifier(
        max_iter        = params.get("max_iter", 400),
        max_depth       = params.get("max_depth", 4),
        learning_rate   = params.get("learning_rate", 0.05),
        min_samples_leaf= params.get("min_samples_leaf", 20),
        l2_regularization= params.get("l2_reg", 0.0),
        random_state    = 42,
    )


def evaluate(model, X, y, label):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    m = {
        "accuracy":  round(float(accuracy_score(y, pred)), 4),
        "auc":       round(float(roc_auc_score(y, prob)), 4),
        "log_loss":  round(float(log_loss(y, prob)), 4),
        "brier":     round(float(brier_score_loss(y, prob)), 4),
        "n":         int(len(y)),
    }
    log.info("  %-6s  acc=%.3f  AUC=%.3f  brier=%.4f  (n=%d)",
             label, m["accuracy"], m["auc"], m["brier"], m["n"])
    return m


def build_sample_weights(df_train):
    """Down-weight 2019-20 bubble restart games."""
    weights = np.ones(len(df_train))
    bubble_mask = (
        (df_train["season"] == "2019-20") &
        (df_train["game_date"] >= pd.Timestamp("2020-07-30"))
    ).values
    weights[bubble_mask] = BUBBLE_WEIGHT
    return weights


def main():
    df, feat_cols = load_data()
    log.info("Loaded %d rows, %d features", len(df), len(feat_cols))

    train_df = df[df["season"].isin(TRAIN_SEASONS)].sort_values("game_date")
    test_df  = df[df["season"] == TEST_SEASON].sort_values("game_date")

    X_train = train_df[feat_cols].values
    y_train = train_df[LABEL_COL].values
    X_test  = test_df[feat_cols].values
    y_test  = test_df[LABEL_COL].values
    sw_train = build_sample_weights(train_df)

    log.info("Train: %d games (%s → %s)  |  Test: %d games (%s)",
             len(X_train), TRAIN_SEASONS[0], TRAIN_SEASONS[-1],
             len(X_test), TEST_SEASON)

    # ── Optuna hyperparameter search ─────────────────────────────────────────
    log.info("Optuna search (%d trials) …", OPTUNA_TRIALS)
    tscv = TimeSeriesSplit(n_splits=7)

    def objective(trial):
        params = {
            "max_iter":        trial.suggest_int("max_iter", 200, 800),
            "max_depth":       trial.suggest_int("max_depth", 3, 7),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf", 10, 60),
            "l2_reg":          trial.suggest_float("l2_reg", 0.0, 2.0),
        }
        est = make_estimator(params)
        scores = cross_val_score(est, X_train, y_train, cv=tscv,
                                 scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    log.info("Best params: %s  →  CV AUC=%.4f", best, study.best_value)

    # ── CV with best params ───────────────────────────────────────────────────
    log.info("Cross-validating best estimator (7 folds) …")
    cv_aucs = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        m = make_estimator(best)
        m.fit(X_train[tr_idx], y_train[tr_idx],
              sample_weight=sw_train[tr_idx])
        prob = m.predict_proba(X_train[val_idx])[:, 1]
        auc  = roc_auc_score(y_train[val_idx], prob)
        cv_aucs.append(auc)
        log.info("  Fold %d  AUC=%.4f", fold, auc)
    log.info("CV AUC: %.4f ± %.4f", np.mean(cv_aucs), np.std(cv_aucs))

    # ── Final model ──────────────────────────────────────────────────────────
    log.info("Fitting final calibrated model on all train data …")
    base  = make_estimator(best)
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")
    model.fit(X_train, y_train)

    train_m = evaluate(model, X_train, y_train, "TRAIN")
    test_m  = evaluate(model, X_test,  y_test,  "TEST ")

    # ── Permutation importance ────────────────────────────────────────────────
    log.info("Computing permutation importance (10 repeats) …")
    perm = permutation_importance(model, X_test, y_test,
                                  n_repeats=10, random_state=42,
                                  scoring="roc_auc", n_jobs=1)
    imp_pairs = sorted(zip(feat_cols, perm.importances_mean.tolist()),
                       key=lambda x: x[1], reverse=True)
    log.info("Top 10 features:")
    for f, v in imp_pairs[:10]:
        log.info("  %-40s %.4f", f, v)

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(MODEL_PKL, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feat_cols,
                     "model_version": MODEL_VERSION}, f)
    log.info("Model saved → %s", MODEL_PKL)

    metrics_out = {
        "model_version":  MODEL_VERSION,
        "feature_cols":   feat_cols,
        "best_params":    best,
        "cv_auc_mean":    round(float(np.mean(cv_aucs)), 4),
        "cv_auc_std":     round(float(np.std(cv_aucs)), 4),
        "train":          train_m,
        "test":           test_m,
        "feature_importance": [{"feature": f, "importance": round(v, 5)}
                                for f, v in imp_pairs],
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_out, f, indent=2)
    log.info("Metrics saved → %s", METRICS_JSON)
    log.info("✓  Model v2 training complete.")


if __name__ == "__main__":
    main()
