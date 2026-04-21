"""
04_edge_finder.py
Compare model predictions vs Vegas implied probabilities for today's games.
Flags bets where the model sees >5% edge.
Outputs data/todays_edges.json.

Run:  python3 04_edge_finder.py
"""

import json
import logging
import pickle
import sqlite3
from datetime import date

import pandas as pd

DB_PATH      = "data/nba.db"
FEATURES_CSV = "data/features.csv"
MODEL_PKL    = "data/model.pkl"
OUTPUT_JSON  = "data/todays_edges.json"
EDGE_THRESH  = 0.05   # flag if model prob exceeds implied by this much

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── American odds → implied probability ─────────────────────────────────────
def american_to_implied(odds: float) -> float:
    if odds is None:
        return None
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


# ── Build feature row for today's game ───────────────────────────────────────
def get_team_features(conn, team_abbr: str, as_of_date: str,
                      feature_cols: list) -> dict | None:
    """
    Pull the most recent rolling-feature row for a team before as_of_date
    from features.csv (which was built from all historical data).
    """
    df = pd.read_csv(FEATURES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    cutoff = pd.to_datetime(as_of_date)

    # home side
    home_rows = df[(df["home_team_abbr"] == team_abbr) & (df["game_date"] < cutoff)]
    away_rows = df[(df["away_team_abbr"] == team_abbr) & (df["game_date"] < cutoff)]

    all_rows = pd.concat([home_rows, away_rows]).sort_values("game_date")
    if all_rows.empty:
        return None
    last = all_rows.iloc[-1]
    return last


def build_matchup_features(home_abbr: str, away_abbr: str,
                            as_of_date: str, feature_cols: list,
                            df_feats: pd.DataFrame) -> list | None:
    """
    Approximate feature vector for a new matchup using each team's last game row.
    We pull home-side features from the last row where home_team_abbr == home_abbr,
    and similarly for away.
    """
    cutoff = pd.to_datetime(as_of_date)

    home_df = df_feats[
        (df_feats["home_team_abbr"] == home_abbr) & (df_feats["game_date"] < cutoff)
    ].sort_values("game_date")

    away_df = df_feats[
        (df_feats["away_team_abbr"] == away_abbr) & (df_feats["game_date"] < cutoff)
    ].sort_values("game_date")

    if home_df.empty or away_df.empty:
        log.warning("  No historical rows for %s (home=%d) or %s (away=%d)",
                    home_abbr, len(home_df), away_abbr, len(away_df))
        return None

    h = home_df.iloc[-1]
    a = away_df.iloc[-1]

    row = []
    for col in feature_cols:
        if col.startswith("home_"):
            row.append(h.get(col, 0.0))
        elif col.startswith("away_"):
            row.append(a.get(col, 0.0))
        elif col.startswith("diff_"):
            base = col[5:]   # strip "diff_"
            hval = h.get(f"home_{base}", 0.0)
            aval = a.get(f"away_{base}", 0.0)
            row.append(hval - aval if pd.notna(hval) and pd.notna(aval) else 0.0)
        else:
            row.append(0.0)
    return row


def match_team_abbr(full_name: str, abbr_map: dict) -> str | None:
    """Fuzzy match a full team name (from odds API) to a 3-letter abbreviation."""
    full_name = full_name.lower()
    for abbr, name in abbr_map.items():
        if abbr.lower() in full_name or name.lower() in full_name:
            return abbr
    # Try last word of city name
    parts = full_name.split()
    for p in parts:
        for abbr, name in abbr_map.items():
            if p in name.lower():
                return abbr
    return None


def main():
    today_str = date.today().isoformat()
    log.info("Edge finder — %s", today_str)

    # Load model
    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    model_version = bundle["model_version"]
    log.info("Loaded model: %s  (%d features)", model_version, len(feature_cols))

    # Load feature history
    df_feats = pd.read_csv(FEATURES_CSV)
    df_feats["game_date"] = pd.to_datetime(df_feats["game_date"])

    # Build abbreviation map from features CSV
    abbr_map = {}
    for _, row in df_feats.iterrows():
        abbr_map[row["home_team_abbr"]] = row["home_team_abbr"]
        abbr_map[row["away_team_abbr"]] = row["away_team_abbr"]

    # Load today's odds from DB
    conn = sqlite3.connect(DB_PATH)
    odds_df = pd.read_sql_query("""
        SELECT * FROM odds
        WHERE date(fetched_at) = date('now', 'localtime')
        ORDER BY fetched_at DESC
    """, conn)
    conn.close()

    if odds_df.empty:
        log.warning("No odds found for today. Run 01_data_pipeline.py with ODDS_API_KEY set.")
        # Still produce output with all games marked as no-odds
        output = {
            "date": today_str,
            "model_version": model_version,
            "games": [],
            "note": "No odds data available — set ODDS_API_KEY and re-run pipeline.",
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(output, f, indent=2)
        log.info("Saved → %s (empty edges)", OUTPUT_JSON)
        return

    # Deduplicate to one odds row per game (latest fetch)
    odds_df = odds_df.drop_duplicates(subset=["home_team", "away_team"], keep="first")
    log.info("Found %d games with odds", len(odds_df))

    games_out = []

    for _, odds_row in odds_df.iterrows():
        home_full = odds_row["home_team"]
        away_full = odds_row["away_team"]

        # Map full names to abbreviations
        home_abbr = match_team_abbr(home_full, abbr_map)
        away_abbr = match_team_abbr(away_full, abbr_map)

        if not home_abbr or not away_abbr:
            log.warning("  Could not map: '%s' or '%s' — skipping", home_full, away_full)
            continue

        log.info("  %s @ %s  [%s @ %s]", away_full, home_full, away_abbr, home_abbr)

        # Build feature vector
        feat_vec = build_matchup_features(
            home_abbr, away_abbr, today_str, feature_cols, df_feats
        )
        if feat_vec is None:
            continue

        feat_array = [[v if pd.notna(v) else 0.0 for v in feat_vec]]
        home_prob = float(model.predict_proba(feat_array)[0][1])
        away_prob = 1.0 - home_prob

        # Implied probs from moneyline
        home_ml = odds_row.get("home_ml")
        away_ml = odds_row.get("away_ml")
        home_implied = american_to_implied(home_ml)
        away_implied = american_to_implied(away_ml)

        # Normalise implied (vig removal)
        if home_implied and away_implied:
            total = home_implied + away_implied
            home_implied_norm = home_implied / total
            away_implied_norm = away_implied / total
        else:
            home_implied_norm = away_implied_norm = None

        home_edge = (home_prob - home_implied_norm) if home_implied_norm else None
        away_edge = (away_prob - away_implied_norm) if away_implied_norm else None

        # Recommendation
        rec = "NO EDGE"
        if home_edge and home_edge >= EDGE_THRESH:
            rec = f"BET HOME ({home_abbr})"
        elif away_edge and away_edge >= EDGE_THRESH:
            rec = f"BET AWAY ({away_abbr})"

        game = {
            "home_team":         home_abbr,
            "away_team":         away_abbr,
            "home_team_full":    home_full,
            "away_team_full":    away_full,
            "home_ml":           home_ml,
            "away_ml":           away_ml,
            "home_spread":       odds_row.get("home_spread"),
            "model_home_prob":   round(home_prob, 4),
            "model_away_prob":   round(away_prob, 4),
            "implied_home_prob": round(home_implied_norm, 4) if home_implied_norm else None,
            "implied_away_prob": round(away_implied_norm, 4) if away_implied_norm else None,
            "home_edge":         round(home_edge, 4) if home_edge else None,
            "away_edge":         round(away_edge, 4) if away_edge else None,
            "recommendation":    rec,
            "value_bet":         rec != "NO EDGE",
        }
        games_out.append(game)

        log.info("    model: home=%.1f%%  away=%.1f%%  |  implied: home=%.1f%%  away=%.1f%%  |  %s",
                 home_prob * 100, away_prob * 100,
                 (home_implied_norm or 0) * 100, (away_implied_norm or 0) * 100,
                 rec)

    value_bets = [g for g in games_out if g["value_bet"]]
    log.info("Value bets found: %d / %d games", len(value_bets), len(games_out))

    output = {
        "date":          today_str,
        "model_version": model_version,
        "edge_threshold": EDGE_THRESH,
        "games":         games_out,
        "value_bets":    value_bets,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved → %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
