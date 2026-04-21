"""
06_backtest.py
Backtest the model on the full 2024-25 holdout season.
Simulates flat-bet strategies and computes ROI, win rate, P&L.
Outputs data/backtest.json.

Run:  python3 06_backtest.py
"""

import json
import logging
import pickle
import numpy as np
import pandas as pd

FEATURES_CSV  = "data/features.csv"
MODEL_PKL     = "data/model.pkl"
OUTPUT_JSON   = "data/backtest.json"
FLAT_BET      = 100        # dollars per bet
VIG_ODDS      = -110       # standard American vig
CONF_THRESH   = 0.55       # bet when model prob >= this
EDGE_THRESH   = 0.04       # edge vs ~52.4% break-even

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def vig_payout(bet: float, odds: int = VIG_ODDS) -> float:
    """Return profit on a winning -110 bet."""
    if odds < 0:
        return bet * (100 / abs(odds))
    return bet * (odds / 100)


def breakeven_pct(odds: int = VIG_ODDS) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def simulate(df_test: pd.DataFrame, model, feature_cols: list) -> dict:
    X = df_test[feature_cols].values
    probs = model.predict_proba(X)[:, 1]   # P(home wins)

    rows = []
    for i, (_, row) in enumerate(df_test.iterrows()):
        home_prob = float(probs[i])
        away_prob = 1.0 - home_prob
        actual    = int(row["home_win"])

        # Determine best bet side and edge vs break-even
        if home_prob >= away_prob:
            bet_side = "home"
            bet_prob = home_prob
        else:
            bet_side = "away"
            bet_prob = away_prob

        edge = bet_prob - breakeven_pct()

        # Strategy A: bet if confident
        strat_a = bet_prob >= CONF_THRESH

        # Strategy B: bet if edge meaningful
        strat_b = edge >= EDGE_THRESH

        # Did we win?
        correct = (bet_side == "home" and actual == 1) or \
                  (bet_side == "away" and actual == 0)

        pnl_a = (vig_payout(FLAT_BET) if correct else -FLAT_BET) if strat_a else 0
        pnl_b = (vig_payout(FLAT_BET) if correct else -FLAT_BET) if strat_b else 0

        rows.append({
            "game_date":   str(row["game_date"])[:10],
            "home":        row["home_team_abbr"],
            "away":        row["away_team_abbr"],
            "home_prob":   round(home_prob, 4),
            "away_prob":   round(away_prob, 4),
            "bet_side":    bet_side,
            "bet_prob":    round(bet_prob, 4),
            "edge":        round(edge, 4),
            "actual_home_win": actual,
            "correct":     correct,
            "strat_a_bet": strat_a,
            "strat_a_pnl": round(pnl_a, 2),
            "strat_b_bet": strat_b,
            "strat_b_pnl": round(pnl_b, 2),
        })

    return rows


def summarise(rows: list, strat: str) -> dict:
    bets   = [r for r in rows if r[f"{strat}_bet"]]
    if not bets:
        return {"n_bets": 0, "wins": 0, "win_pct": 0, "roi": 0, "total_pnl": 0}
    wins   = sum(1 for r in bets if r["correct"])
    total  = sum(r[f"{strat}_pnl"] for r in bets)
    risk   = len(bets) * FLAT_BET
    return {
        "n_bets":   len(bets),
        "wins":     wins,
        "win_pct":  round(wins / len(bets), 4),
        "roi":      round(total / risk, 4),
        "total_pnl": round(total, 2),
        "breakeven_pct": round(breakeven_pct(), 4),
    }


def monthly_pnl(rows: list, strat: str) -> list:
    df = pd.DataFrame(rows)
    df["month"] = df["game_date"].str[:7]
    out = []
    for month, grp in df.groupby("month"):
        bets = grp[grp[f"{strat}_bet"]]
        pnl  = bets[f"{strat}_pnl"].sum()
        out.append({"month": month, "pnl": round(float(pnl), 2), "n": int(len(bets))})
    return out


def cumulative_pnl(rows: list, strat: str) -> list:
    cum = 0
    result = []
    for r in rows:
        if r[f"{strat}_bet"]:
            cum += r[f"{strat}_pnl"]
            result.append({"date": r["game_date"], "cum_pnl": round(cum, 2)})
    return result


def recent_picks(rows: list, n: int = 20) -> list:
    """Return last N games where model was most confident."""
    high_conf = sorted(rows, key=lambda r: r["bet_prob"], reverse=True)[:50]
    # Sort those by date descending for recency
    high_conf = sorted(high_conf, key=lambda r: r["game_date"], reverse=True)[:n]
    return high_conf


def build_all_matchups(df_feats: pd.DataFrame, model, feature_cols: list) -> dict:
    """
    Pre-compute home win probability for all 30×29 team pairings
    using each team's most recent 2024-25 feature row.
    """
    season_df = df_feats[df_feats["season"] == "2024-25"].copy()
    season_df["game_date"] = pd.to_datetime(season_df["game_date"])

    # Latest home row per team
    home_latest = (season_df.sort_values("game_date")
                   .drop_duplicates("home_team_abbr", keep="last")
                   .set_index("home_team_abbr"))

    # Latest away row per team
    away_latest = (season_df.sort_values("game_date")
                   .drop_duplicates("away_team_abbr", keep="last")
                   .set_index("away_team_abbr"))

    teams = sorted(set(home_latest.index) & set(away_latest.index))

    matchups = {}
    for home in teams:
        matchups[home] = {}
        h = home_latest.loc[home]
        for away in teams:
            if home == away:
                continue
            a = away_latest.loc[away]

            row = []
            for col in feature_cols:
                if col.startswith("home_"):
                    row.append(float(h.get(col, 0) or 0))
                elif col.startswith("away_"):
                    row.append(float(a.get(col, 0) or 0))
                elif col.startswith("diff_"):
                    base = col[5:]
                    hval = float(h.get(f"home_{base}", 0) or 0)
                    aval = float(a.get(f"away_{base}", 0) or 0)
                    row.append(hval - aval)
                else:
                    row.append(0.0)

            prob = float(model.predict_proba([row])[0][1])
            matchups[home][away] = round(prob, 4)

    return matchups


def team_ratings(df_feats: pd.DataFrame) -> list:
    """Current team ratings based on most recent 2024-25 features."""
    season_df = df_feats[df_feats["season"] == "2024-25"].copy()
    season_df["game_date"] = pd.to_datetime(season_df["game_date"])

    home_latest = (season_df.sort_values("game_date")
                   .drop_duplicates("home_team_abbr", keep="last"))

    ratings = []
    for _, row in home_latest.iterrows():
        ratings.append({
            "team": row["home_team_abbr"],
            "off_rtg":   round(float(row.get("home_roll_off_rtg", 0) or 0), 1),
            "ts_pct":    round(float(row.get("home_roll_ts", 0) or 0), 3),
            "win_pct":   round(float(row.get("home_season_win_pct", 0) or 0), 3),
            "pace":      round(float(row.get("home_roll_pace", 0) or 0), 1),
            "reb":       round(float(row.get("home_roll_reb", 0) or 0), 1),
            "ast":       round(float(row.get("home_roll_ast", 0) or 0), 1),
            "tov":       round(float(row.get("home_roll_tov", 0) or 0), 1),
            "rest_days": round(float(row.get("home_rest_days", 0) or 0), 1),
        })

    return sorted(ratings, key=lambda r: r["win_pct"], reverse=True)


def main():
    df = pd.read_csv(FEATURES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])

    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    model_version = bundle["model_version"]

    test_df = df[df["season"] == "2024-25"].sort_values("game_date").reset_index(drop=True)
    log.info("Backtesting on %d games (2024-25)", len(test_df))

    rows = simulate(test_df, model, feature_cols)

    sum_a = summarise(rows, "strat_a")
    sum_b = summarise(rows, "strat_b")
    log.info("Strategy A (conf≥%.0f%%): %d bets | %.1f%% win | ROI %.1f%% | P&L $%.0f",
             CONF_THRESH*100, sum_a["n_bets"], sum_a["win_pct"]*100,
             sum_a["roi"]*100, sum_a["total_pnl"])
    log.info("Strategy B (edge≥%.0f%%): %d bets | %.1f%% win | ROI %.1f%% | P&L $%.0f",
             EDGE_THRESH*100, sum_b["n_bets"], sum_b["win_pct"]*100,
             sum_b["roi"]*100, sum_b["total_pnl"])

    # Overall model accuracy
    all_correct = sum(1 for r in rows if r["correct"])
    total_games = len(rows)
    log.info("Overall model accuracy: %d/%d = %.1f%%",
             all_correct, total_games, all_correct/total_games*100)

    log.info("Pre-computing all team matchup predictions …")
    matchups = build_all_matchups(df, model, feature_cols)
    log.info("Built %d × team matchup matrix", len(matchups))

    ratings = team_ratings(df)

    output = {
        "model_version":  model_version,
        "season":         "2024-25",
        "flat_bet":       FLAT_BET,
        "vig_odds":       VIG_ODDS,
        "breakeven_pct":  round(breakeven_pct(), 4),
        "conf_thresh":    CONF_THRESH,
        "edge_thresh":    EDGE_THRESH,
        "total_games":    total_games,
        "overall_accuracy": round(all_correct / total_games, 4),
        "strategy_a": sum_a,
        "strategy_b": sum_b,
        "strategy_a_monthly": monthly_pnl(rows, "strat_a"),
        "strategy_b_monthly": monthly_pnl(rows, "strat_b"),
        "strategy_a_cumulative": cumulative_pnl(rows, "strat_a"),
        "strategy_b_cumulative": cumulative_pnl(rows, "strat_b"),
        "recent_picks":   recent_picks(rows, 20),
        "all_games":      rows,
        "matchups":       matchups,
        "team_ratings":   ratings,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved → %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
