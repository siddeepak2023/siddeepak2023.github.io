"""
05_export_dashboard.py
Compile all pipeline outputs into a single data/dashboard_data.json
for the self-contained HTML dashboard.

Run:  python3 05_export_dashboard.py
"""

import json
import logging
import sqlite3
from datetime import date, datetime

import pandas as pd

DB_PATH        = "data/nba.db"
FEATURES_CSV   = "data/features.csv"
METRICS_JSON   = "data/model_metrics.json"
EDGES_JSON     = "data/todays_edges.json"
OUTPUT_JSON    = "data/dashboard_data.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Season standings (win % per team per season) ─────────────────────────────
def build_standings(conn) -> dict:
    df = pd.read_sql_query("""
        SELECT season, team_abbr, team_name,
               SUM(win) as wins,
               SUM(CASE WHEN win=0 THEN 1 ELSE 0 END) as losses
        FROM team_game_logs
        WHERE wl IN ('W','L')
        GROUP BY season, team_abbr, team_name
        ORDER BY season, wins DESC
    """, conn)
    df["win_pct"] = (df["wins"] / (df["wins"] + df["losses"])).round(3)

    standings = {}
    for season, grp in df.groupby("season"):
        standings[season] = grp[["team_abbr", "team_name", "wins", "losses", "win_pct"]].to_dict("records")
    return standings


# ── Rolling team form (last 20 games per team for current season) ─────────────
def build_team_form(conn) -> list:
    df = pd.read_sql_query("""
        SELECT team_abbr, game_date, win, pts, opponent_abbr, is_home
        FROM team_game_logs
        WHERE season='2024-25' AND wl IN ('W','L')
        ORDER BY team_abbr, game_date DESC
    """, conn)
    result = []
    for team, grp in df.groupby("team_abbr"):
        last20 = grp.head(20).sort_values("game_date")
        result.append({
            "team": team,
            "dates":  last20["game_date"].tolist(),
            "wins":   last20["win"].tolist(),
            "pts":    last20["pts"].tolist(),
        })
    return result


# ── Home vs away splits ───────────────────────────────────────────────────────
def build_home_away_splits(conn) -> list:
    df = pd.read_sql_query("""
        SELECT season, team_abbr,
               AVG(CASE WHEN is_home=1 THEN win END) as home_win_pct,
               AVG(CASE WHEN is_home=0 THEN win END) as away_win_pct,
               AVG(CASE WHEN is_home=1 THEN pts END) as home_pts,
               AVG(CASE WHEN is_home=0 THEN pts END) as away_pts
        FROM team_game_logs
        WHERE wl IN ('W','L')
        GROUP BY season, team_abbr
    """, conn)
    df = df.round(3).fillna(0)
    return df.to_dict("records")


# ── Season-level home win rate trend ─────────────────────────────────────────
def build_home_win_trend(conn) -> list:
    df = pd.read_sql_query("""
        SELECT season,
               AVG(CASE WHEN is_home=1 THEN win ELSE NULL END) as home_win_pct,
               AVG(CASE WHEN is_home=0 THEN win ELSE NULL END) as away_win_pct
        FROM team_game_logs
        WHERE wl IN ('W','L')
        GROUP BY season
        ORDER BY season
    """, conn)
    return df.round(3).to_dict("records")


# ── Monthly scoring trends (league avg pts per game by month) ─────────────────
def build_monthly_scoring(conn) -> list:
    df = pd.read_sql_query("""
        SELECT substr(game_date, 1, 7) as month,
               AVG(pts) as avg_pts,
               COUNT(*) as games
        FROM team_game_logs
        WHERE season='2024-25' AND pts > 0
        GROUP BY month
        ORDER BY month
    """, conn)
    return df.round(1).to_dict("records")


# ── Feature importance for model tab ─────────────────────────────────────────
def build_feature_importance(metrics: dict) -> list:
    fi = metrics.get("feature_importance", [])
    # Normalise to 0-100
    max_val = max((x["importance"] for x in fi), default=1)
    return [
        {
            "feature": x["feature"].replace("diff_", "Δ ").replace("home_", "H ").replace("away_", "A ").replace("roll_", "").replace("_", " "),
            "raw_feature": x["feature"],
            "importance": round(x["importance"] / max_val * 100, 1) if max_val > 0 else 0,
        }
        for x in fi[:20]
    ]


# ── Top predicted games for today ────────────────────────────────────────────
def build_today_predictions(edges: dict) -> list:
    games = edges.get("games", [])
    return sorted(games, key=lambda g: abs(g.get("home_edge") or 0), reverse=True)


# ── Head-to-head history ─────────────────────────────────────────────────────
def build_h2h(conn) -> list:
    df = pd.read_sql_query("""
        SELECT a.team_abbr as team1, a.opponent_abbr as team2,
               a.season,
               SUM(a.win) as wins,
               COUNT(*) as games
        FROM team_game_logs a
        WHERE a.is_home = 1
        GROUP BY a.season, a.team_abbr, a.opponent_abbr
        HAVING games >= 2
        ORDER BY a.team_abbr, a.season
    """, conn)
    return df.to_dict("records")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    conn    = sqlite3.connect(DB_PATH)
    metrics = load_json(METRICS_JSON)
    edges   = load_json(EDGES_JSON)

    log.info("Building standings …")
    standings = build_standings(conn)

    log.info("Building team form …")
    team_form = build_team_form(conn)

    log.info("Building home/away splits …")
    splits = build_home_away_splits(conn)

    log.info("Building home win trend …")
    hw_trend = build_home_win_trend(conn)

    log.info("Building monthly scoring …")
    monthly = build_monthly_scoring(conn)

    log.info("Building H2H …")
    h2h = build_h2h(conn)

    conn.close()

    feature_importance = build_feature_importance(metrics)
    today_games = build_today_predictions(edges)

    # Row counts by season
    features_df = pd.read_csv(FEATURES_CSV)
    season_counts = features_df.groupby("season").size().to_dict()

    dashboard_data = {
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "date":           date.today().isoformat(),
        "model": {
            "version":    metrics["model_version"],
            "cv_auc":     metrics["cv_auc_mean"],
            "cv_auc_std": metrics["cv_auc_std"],
            "test_auc":   metrics["test"]["auc"],
            "test_acc":   metrics["test"]["accuracy"],
            "test_brier": metrics["test"]["brier"],
            "train_seasons": ["2022-23", "2023-24"],
            "test_season":   "2024-25",
            "n_features":    len(metrics["feature_cols"]),
            "n_train":       metrics["train"]["n"],
            "n_test":        metrics["test"]["n"],
        },
        "feature_importance": feature_importance,
        "standings":          standings,
        "home_win_trend":     hw_trend,
        "monthly_scoring":    monthly,
        "home_away_splits":   splits,
        "team_form":          team_form,
        "today": {
            "date":       edges["date"],
            "games":      today_games,
            "value_bets": [g for g in today_games if g.get("value_bet")],
            "has_odds":   len(today_games) > 0,
            "edge_threshold": edges.get("edge_threshold", 0.05),
        },
        "h2h": h2h,
        "season_game_counts": season_counts,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    log.info("Saved → %s", OUTPUT_JSON)
    log.info("  standings seasons: %s", list(standings.keys()))
    log.info("  today games: %d  value bets: %d",
             len(today_games), len(dashboard_data["today"]["value_bets"]))
    log.info("  model AUC: %.4f  acc: %.4f", metrics["test"]["auc"], metrics["test"]["accuracy"])
    log.info("✓  Dashboard data export complete.")


if __name__ == "__main__":
    main()
