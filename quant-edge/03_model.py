"""
03_model.py
Walk-forward backtested HistGradientBoosting model.
Train on rolling 3-year window, predict next quarter, repeat.
Outputs model.pkl + metrics.json with backtest P&L curve.

Run:  python3 03_model.py
"""

import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta

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

FEATURES_CSV   = "data/features.csv"
MODEL_PKL      = "data/model.pkl"
METRICS_JSON   = "data/model_metrics.json"
MODEL_VERSION  = "quant_edge_v1_hgbm"
OPTUNA_TRIALS  = 50
TRAIN_YEARS    = 3       # rolling training window
EMBARGO_DAYS   = 10      # gap between train end and test start (prevent leakage)
MIN_TEST_ROWS  = 500

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "ret_1d","ret_5d","ret_10d","ret_20d","ret_60d","mom_accel",
    "price_52w_pct","rsi_14","rsi_7","macd_hist","macd_cross",
    "bb_pct_b","bb_width","atr_14","vol_ratio","vol_trend",
    "vol_20d","vol_60d","vol_ratio_rv",
    "pct_above_sma20","pct_above_sma50","pct_above_sma200","sma_20_50_cross",
    "spy_ret_5d","spy_ret_20d","vix_level","vix_ret_5d","sector_ret_5d",
]
LABEL_COL = "target"


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Missing %d features (skipped): %s", len(missing), missing[:5])
    return df.sort_values("date").reset_index(drop=True), available


def make_estimator(params):
    return HistGradientBoostingClassifier(
        max_iter          = params.get("max_iter", 300),
        max_depth         = params.get("max_depth", 4),
        learning_rate     = params.get("learning_rate", 0.05),
        min_samples_leaf  = params.get("min_samples_leaf", 30),
        l2_regularization = params.get("l2_reg", 0.1),
        random_state      = 42,
    )


def evaluate(model, X, y, label):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    m = {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "auc":      round(float(roc_auc_score(y, prob)), 4),
        "log_loss": round(float(log_loss(y, prob)), 4),
        "brier":    round(float(brier_score_loss(y, prob)), 4),
        "n":        int(len(y)),
    }
    log.info("  %-6s  acc=%.3f  AUC=%.3f  brier=%.4f  (n=%d)",
             label, m["accuracy"], m["auc"], m["brier"], m["n"])
    return m


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def walk_forward_backtest(df, feat_cols, best_params):
    """
    Roll a 3-year training window forward in 3-month steps.
    Returns df with columns: date, ticker, prob, target, fwd_ret_5d
    """
    log.info("Walk-forward backtest …")
    dates     = df["date"].sort_values().unique()
    all_preds = []

    # Date windows
    d_min = dates[0]
    d_max = dates[-1]
    train_delta   = timedelta(days=TRAIN_YEARS * 365)
    embargo_delta = timedelta(days=EMBARGO_DAYS)
    step_delta    = timedelta(days=90)

    test_start = d_min + train_delta
    fold = 0

    while test_start < d_max:
        test_end   = test_start + step_delta
        train_end  = test_start - embargo_delta
        train_start = train_end - train_delta

        train = df[(df["date"] >= train_start) & (df["date"] < train_end)]
        test  = df[(df["date"] >= test_start)  & (df["date"] <  test_end)]

        if len(train) < 5000 or len(test) < MIN_TEST_ROWS:
            test_start += step_delta
            continue

        X_tr = train[feat_cols].values
        y_tr = train[LABEL_COL].values
        X_te = test[feat_cols].values
        y_te = test[LABEL_COL].values

        model = make_estimator(best_params)
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        auc  = roc_auc_score(y_te, prob)

        fold += 1
        log.info("  Fold %2d  [%s → %s]  test=%d  AUC=%.4f",
                 fold,
                 train_start.strftime("%Y-%m"),
                 train_end.strftime("%Y-%m"),
                 len(test), auc)

        fold_df = test[["date","ticker","sector","fwd_ret_5d",LABEL_COL]].copy()
        fold_df["prob"]      = prob
        fold_df["pred"]      = (prob >= 0.5).astype(int)
        fold_df["fold"]      = fold
        fold_df["fold_auc"]  = round(auc, 4)
        all_preds.append(fold_df)
        test_start += step_delta

    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


# ── Backtest analytics ────────────────────────────────────────────────────────

def compute_backtest_stats(preds, conf_threshold=0.60):
    """Compute strategy P&L, Sharpe, win rate, drawdown."""
    if preds.empty:
        return {}

    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"])

    # Strategy A: all predictions with conf > threshold (long high prob, skip rest)
    strategy = preds[preds["prob"] >= conf_threshold].copy()
    strategy["ret"] = strategy.apply(
        lambda r: r["fwd_ret_5d"] if r["pred"] == 1 else -r["fwd_ret_5d"], axis=1)

    # Group by date → average return per day signal fired
    daily = strategy.groupby("date")["ret"].mean()
    total_trades = len(strategy)
    win_rate = (strategy["ret"] > 0).mean() if not strategy.empty else 0

    # Annualised Sharpe (5-day returns → annualize by 252/5 periods/year)
    periods_per_year = 252 / 5
    sharpe = (daily.mean() * periods_per_year) / (daily.std() * np.sqrt(periods_per_year) + 1e-9)

    # Cumulative P&L curve (normalised to 100 start)
    cum  = (1 + daily).cumprod()
    cum_monthly = cum.resample("ME").last().dropna()
    monthly_rets = cum.resample("ME").last().pct_change().dropna()

    # Max drawdown
    roll_max  = cum.cummax()
    drawdown  = (cum - roll_max) / roll_max
    max_dd    = float(drawdown.min())

    # Strategy B: top-N decile by prob per day (long top 10%)
    preds["prob_decile"] = preds.groupby("date")["prob"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
    top_decile = preds[preds["prob_decile"] == 9].copy()
    top_decile["ret"] = top_decile["fwd_ret_5d"]
    daily_b = top_decile.groupby("date")["ret"].mean()
    cum_b   = (1 + daily_b).cumprod()
    sharpe_b = (daily_b.mean() * periods_per_year) / \
               (daily_b.std() * np.sqrt(periods_per_year) + 1e-9)

    # Monthly curve for dashboard
    def to_monthly_curve(cum_series):
        monthly = cum_series.resample("ME").last().dropna()
        return [{"date": d.strftime("%Y-%m"), "value": round(float(v * 100), 2)}
                for d, v in monthly.items()]

    # Per-fold AUC
    fold_aucs = preds.groupby("fold")["fold_auc"].first().tolist() if "fold" in preds else []

    return {
        "strategy_a": {
            "conf_threshold": conf_threshold,
            "total_trades":   int(total_trades),
            "win_rate":       round(float(win_rate), 4),
            "sharpe":         round(float(sharpe), 3),
            "max_drawdown":   round(max_dd, 4),
            "total_return":   round(float(cum.dropna().iloc[-1] - 1) if not cum.dropna().empty else 0, 4),
        },
        "strategy_b": {
            "description":  "Top-decile daily long",
            "sharpe":       round(float(sharpe_b), 3),
            "total_return": round(float(cum_b.dropna().iloc[-1] - 1) if not cum_b.dropna().empty else 0, 4),
        },
        "curve_a":       to_monthly_curve(cum),
        "curve_b":       to_monthly_curve(cum_b),
        "fold_aucs":     [round(float(a), 4) for a in fold_aucs],
        "monthly_rets":  [{"date": d.strftime("%Y-%m"), "ret": round(float(r), 4)}
                          for d, r in monthly_rets.items()],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df, feat_cols = load_data()
    log.info("Loaded %d rows, %d features, %d tickers",
             len(df), len(feat_cols), df["ticker"].nunique())

    # Use last 6 months as final holdout; rest for Optuna search + walk-forward
    cutoff = df["date"].max() - pd.DateOffset(months=6)
    train_pool = df[df["date"] < cutoff]
    holdout    = df[df["date"] >= cutoff]

    X_pool = train_pool[feat_cols].values
    y_pool = train_pool[LABEL_COL].values

    # ── Optuna search on pool ────────────────────────────────────────────────
    log.info("Optuna search (%d trials) …", OPTUNA_TRIALS)
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "max_iter":        trial.suggest_int("max_iter", 150, 600),
            "max_depth":       trial.suggest_int("max_depth", 3, 6),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf", 20, 80),
            "l2_reg":          trial.suggest_float("l2_reg", 0.0, 2.0),
        }
        scores = cross_val_score(make_estimator(params), X_pool, y_pool,
                                 cv=tscv, scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    log.info("Best params: %s  →  CV AUC=%.4f", best, study.best_value)

    # ── Walk-forward backtest ─────────────────────────────────────────────────
    preds_df = walk_forward_backtest(df, feat_cols, best)
    backtest_stats = compute_backtest_stats(preds_df, conf_threshold=0.60)

    # ── Final model on all non-holdout data ───────────────────────────────────
    log.info("Fitting final calibrated model …")
    X_all = train_pool[feat_cols].values
    y_all = train_pool[LABEL_COL].values
    base  = make_estimator(best)
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")
    model.fit(X_all, y_all)

    train_m   = evaluate(model, X_all,                  y_all,                  "TRAIN")
    holdout_m = evaluate(model, holdout[feat_cols].values, holdout[LABEL_COL].values, "HOLD ")

    # ── Permutation importance ────────────────────────────────────────────────
    log.info("Computing permutation importance …")
    perm = permutation_importance(
        model, holdout[feat_cols].values, holdout[LABEL_COL].values,
        n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=1)
    imp_pairs = sorted(zip(feat_cols, perm.importances_mean.tolist()),
                       key=lambda x: x[1], reverse=True)
    log.info("Top 10 features:")
    for f, v in imp_pairs[:10]:
        log.info("  %-40s %.5f", f, v)

    # ── Save model ────────────────────────────────────────────────────────────
    with open(MODEL_PKL, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feat_cols,
                     "model_version": MODEL_VERSION}, f)
    log.info("Model saved → %s", MODEL_PKL)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_out = {
        "model_version":  MODEL_VERSION,
        "feature_cols":   feat_cols,
        "best_params":    best,
        "cv_auc":         round(study.best_value, 4),
        "train":          train_m,
        "holdout":        holdout_m,
        "backtest":       backtest_stats,
        "feature_importance": [{"feature": f, "importance": round(v, 6)}
                                for f, v in imp_pairs],
        "n_tickers":      int(df["ticker"].nunique()),
        "date_range":     [str(df["date"].min())[:10], str(df["date"].max())[:10]],
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_out, f, indent=2)
    log.info("Metrics saved → %s", METRICS_JSON)
    log.info("✓  Model training complete.")


if __name__ == "__main__":
    main()
