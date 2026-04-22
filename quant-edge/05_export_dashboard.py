"""
05_export_dashboard.py
Compile all pipeline outputs into data/dashboard_data.json
for the self-contained HTML dashboard.

Run:  python3 05_export_dashboard.py
"""

import json
import logging
import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd

DB_PATH        = "data/quant.db"
FEATURES_CSV   = "data/features.csv"
METRICS_JSON   = "data/model_metrics.json"
SCREENER_JSON  = "data/screener.json"
OUTPUT_JSON    = "data/dashboard_data.json"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── Sector performance summary ────────────────────────────────────────────────
def build_sector_perf(conn):
    df = pd.read_sql_query("""
        SELECT p.ticker, p.date, p.adj_close, t.sector
        FROM prices p JOIN tickers t ON p.ticker=t.ticker
        WHERE t.is_etf=0 AND p.adj_close > 0
        ORDER BY p.ticker, p.date
    """, conn)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"])
    df["ret"] = df.groupby("ticker")["adj_close"].pct_change()

    # Monthly sector returns
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby(["sector","month"])["ret"].mean().reset_index()
    monthly["month"] = monthly["month"].astype(str)
    monthly["ret"]   = monthly["ret"].round(5)

    # Recent 21-day sector return
    recent_cutoff = df["date"].max() - pd.Timedelta(days=21)
    recent = df[df["date"] > recent_cutoff].groupby("sector")["ret"].mean() * 21
    recent = recent.round(4).to_dict()

    sector_list = [
        {"sector": s, "ret_21d": recent.get(s, 0)}
        for s in df["sector"].dropna().unique()
        if s != "Unknown"
    ]
    sector_list.sort(key=lambda x: x["ret_21d"], reverse=True)

    return sector_list, monthly.to_dict("records")


# ── Top movers (last 5 days) ───────────────────────────────────────────────────
def build_top_movers(conn, n=10):
    df = pd.read_sql_query("""
        SELECT p.ticker, p.date, p.adj_close, t.sector, t.name
        FROM prices p JOIN tickers t ON p.ticker=t.ticker
        WHERE t.is_etf=0 AND p.adj_close > 0
        ORDER BY p.ticker, p.date
    """, conn)
    df["date"] = pd.to_datetime(df["date"])

    latest_date = df["date"].max()
    prev5_date  = latest_date - pd.Timedelta(days=7)

    latest = df[df["date"] == latest_date].set_index("ticker")["adj_close"]
    prev5  = df[df["date"] >= prev5_date].groupby("ticker")["adj_close"].first()

    movers = ((latest / prev5 - 1) * 100).dropna().sort_values()

    gainers  = movers.tail(n)[::-1]
    losers   = movers.head(n)

    meta = df[["ticker","sector","name"]].drop_duplicates("ticker").set_index("ticker")

    def fmt(s):
        return [{"ticker": t, "ret_pct": round(float(v), 2),
                 "sector": meta.loc[t, "sector"] if t in meta.index else "",
                 "name":   meta.loc[t, "name"]   if t in meta.index else t}
                for t, v in s.items()]

    return fmt(gainers), fmt(losers)


# ── Feature importance (top 20) ───────────────────────────────────────────────
def build_feature_importance(metrics):
    fi    = metrics.get("feature_importance", [])
    max_v = max((x["importance"] for x in fi), default=1) or 1
    label_map = {
        "ret_5d": "5d Return", "ret_20d": "20d Return", "ret_60d": "60d Return",
        "rsi_14": "RSI(14)", "rsi_7": "RSI(7)", "macd_hist": "MACD Hist",
        "macd_cross": "MACD Cross", "bb_pct_b": "BB %B", "bb_width": "BB Width",
        "atr_14": "ATR(14)", "vol_ratio": "Vol Ratio", "vol_20d": "Realized Vol",
        "vol_ratio_rv": "Vol Regime", "pct_above_sma20": "vs SMA20",
        "pct_above_sma50": "vs SMA50", "pct_above_sma200": "vs SMA200",
        "mom_accel": "Momentum Accel", "price_52w_pct": "52W High %",
        "spy_ret_5d": "SPY 5d", "vix_level": "VIX Level",
        "sector_ret_5d": "Sector 5d",
    }
    return [{"feature": label_map.get(x["feature"], x["feature"]),
             "raw":     x["feature"],
             "importance": round(x["importance"] / max_v * 100, 1)}
            for x in fi[:20]]


# ── Historical accuracy by decile ─────────────────────────────────────────────
def build_decile_accuracy(features_path, metrics):
    """Use backtest folds to show accuracy by probability decile."""
    # Synthetic decile table from backtest stats if preds not saved separately
    # We'll show expected calibration curve
    backtest = metrics.get("backtest", {})
    fold_aucs = backtest.get("fold_aucs", [])
    if not fold_aucs:
        return []
    return [{"fold": i + 1, "auc": v} for i, v in enumerate(fold_aucs)]


# ── Ticker deep-dive data ──────────────────────────────────────────────────────
def build_ticker_profiles(conn, screener, n=50):
    """Return detailed data for the top-N screener picks."""
    picks = screener.get("picks", [])[:n]
    tickers = [p["ticker"] for p in picks]
    if not tickers:
        return []

    placeholders = ",".join(["?"]*len(tickers))
    df = pd.read_sql_query(f"""
        SELECT p.ticker, p.date, p.adj_close, p.volume, t.sector, t.name
        FROM prices p JOIN tickers t ON p.ticker=t.ticker
        WHERE p.ticker IN ({placeholders}) AND p.adj_close > 0
        ORDER BY p.ticker, p.date
    """, conn, params=tickers)
    df["date"] = pd.to_datetime(df["date"])

    result = []
    pick_map = {p["ticker"]: p for p in picks}
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date")
        p   = pick_map.get(ticker, {})
        last90 = grp.tail(90)
        result.append({
            "ticker":    ticker,
            "name":      grp["name"].iloc[-1],
            "sector":    grp["sector"].iloc[-1],
            "prob_up":   p.get("prob_up"),
            "signal":    p.get("signal"),
            "price":     round(float(grp["adj_close"].iloc[-1]), 2),
            "dates":     last90["date"].dt.strftime("%Y-%m-%d").tolist(),
            "prices":    last90["adj_close"].round(2).tolist(),
            "volumes":   last90["volume"].round(0).tolist(),
        })
    return result


# ── SPY benchmark curve ───────────────────────────────────────────────────────
def build_spy_curve(conn):
    df = pd.read_sql_query(
        "SELECT date, adj_close FROM prices WHERE ticker='SPY' ORDER BY date", conn)
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date").resample("ME").last().dropna()
    monthly["norm"] = monthly["adj_close"] / monthly["adj_close"].iloc[0] * 100
    return [{"date": d.strftime("%Y-%m"), "value": round(float(v), 2)}
            for d, v in monthly["norm"].items()]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    conn     = sqlite3.connect(DB_PATH)
    metrics  = load_json(METRICS_JSON)
    screener = load_json(SCREENER_JSON)

    log.info("Building sector performance …")
    sector_perf, sector_monthly = build_sector_perf(conn)

    log.info("Building top movers …")
    gainers, losers = build_top_movers(conn)

    log.info("Building ticker profiles …")
    ticker_profiles = build_ticker_profiles(conn, screener)

    log.info("Building SPY curve …")
    spy_curve = build_spy_curve(conn)

    conn.close()

    feature_importance = build_feature_importance(metrics)
    fold_aucs          = build_decile_accuracy(FEATURES_CSV, metrics)

    backtest = metrics.get("backtest", {})

    # Patch total_return from curve if NaN (can happen when cumprod has infs)
    def _patch_tr(sd, curve):
        import math
        sd = dict(sd) if sd else {}
        tr = sd.get("total_return", 0)
        if tr is None or (isinstance(tr, float) and math.isnan(tr)):
            last_val = curve[-1]["value"] if curve else 100
            sd["total_return"] = round(last_val / 100 - 1, 4)
        return sd

    sa = _patch_tr(backtest.get("strategy_a", {}), backtest.get("curve_a", []))
    sb = _patch_tr(backtest.get("strategy_b", {}), backtest.get("curve_b", []))

    dashboard_data = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "date":         date.today().isoformat(),
        "model": {
            "version":       metrics["model_version"],
            "cv_auc":        metrics["cv_auc"],
            "holdout_auc":   metrics["holdout"]["auc"],
            "holdout_acc":   metrics["holdout"]["accuracy"],
            "holdout_brier": metrics["holdout"]["brier"],
            "n_tickers":     metrics["n_tickers"],
            "date_range":    metrics["date_range"],
            "n_features":    len(metrics["feature_cols"]),
        },
        "backtest": {
            "strategy_a":    sa,
            "strategy_b":    sb,
            "curve_a":       backtest.get("curve_a", []),
            "curve_b":       backtest.get("curve_b", []),
            "fold_aucs":     backtest.get("fold_aucs", []),
            "monthly_rets":  backtest.get("monthly_rets", []),
            "spy_curve":     spy_curve,
        },
        "screener": {
            "date":             screener["date"],
            "n_tickers":        screener["n_tickers"],
            "n_signals":        screener["n_signals"],
            "conf_threshold":   screener["conf_threshold"],
            "picks":            screener["picks"][:100],
            "tiers":            screener.get("tiers", {}),
            "sector_counts":    screener.get("sector_signal_counts", {}),
        },
        "feature_importance": feature_importance,
        "fold_aucs":          fold_aucs,
        "sector_perf":        sector_perf,
        "sector_monthly":     sector_monthly,
        "gainers":            gainers,
        "losers":             losers,
        "ticker_profiles":    ticker_profiles,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    log.info("Saved → %s", OUTPUT_JSON)
    log.info("  screener picks: %d  |  signals: %d",
             screener["n_tickers"], screener["n_signals"])
    log.info("  holdout AUC: %.4f  |  acc: %.4f",
             metrics["holdout"]["auc"], metrics["holdout"]["accuracy"])
    bt_a = backtest.get("strategy_a", {})
    log.info("  strategy A: sharpe=%.2f  total_ret=%.1f%%  win_rate=%.1f%%",
             bt_a.get("sharpe", 0),
             bt_a.get("total_return", 0) * 100,
             bt_a.get("win_rate", 0) * 100)
    log.info("✓  Dashboard export complete.")


if __name__ == "__main__":
    main()
