"""
02_feature_engineering.py
Build rolling 10-game team features for every game in the DB.
Outputs data/features.csv with one row per (game_id, home_team, away_team).

Run:  python3 02_feature_engineering.py
"""

import sqlite3
import logging
import numpy as np
import pandas as pd

DB_PATH      = "data/nba.db"
FEATURES_CSV = "data/features.csv"
WINDOW       = 10   # rolling game window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Load raw logs ────────────────────────────────────────────────────────────
def load_logs(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("""
        SELECT season, team_id, team_abbr, game_id, game_date,
               is_home, opponent_abbr, win,
               pts, fga, fgm, fg3m, fg3a, ftm, fta,
               oreb, dreb, reb, ast, stl, blk, tov, pf, min
        FROM team_game_logs
        ORDER BY team_id, game_date
    """, conn)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# ── Per-team rolling features ────────────────────────────────────────────────
def _possessions(row) -> float:
    return row.fga - row.oreb + row.tov + 0.4 * row.fta


def _ts_pct(row) -> float:
    denom = 2 * (row.fga + 0.44 * row.fta)
    return row.pts / denom if denom > 0 else 0.0


def build_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute rolling WINDOW-game averages (excluding current game)
    so there's no data leakage into the label.
    """
    records = []

    for team_id, grp in df.groupby("team_id"):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        grp["poss"]   = grp.apply(_possessions, axis=1)
        grp["off_rtg"] = (grp["pts"]   / grp["poss"].replace(0, np.nan)) * 100
        grp["def_pts"] = np.nan           # filled via opponent join later
        grp["ts"]      = grp.apply(_ts_pct, axis=1)
        grp["pace"]    = grp["poss"]

        # Rolling means (shift 1 so current game is excluded)
        roll = grp[["win", "pts", "off_rtg", "ts", "pace",
                     "tov", "oreb", "dreb", "reb", "ast",
                     "fga", "fgm", "fg3m", "fg3a"]].shift(1).rolling(WINDOW, min_periods=3)

        roll_mean = roll.mean()

        # Rest days
        grp["rest_days"] = grp["game_date"].diff().dt.days.fillna(3)

        for i, row in grp.iterrows():
            if i < 3:
                continue  # skip until we have enough history
            rec = {
                "game_id":    row.game_id,
                "game_date":  row.game_date,
                "season":     row.season,
                "team_id":    team_id,
                "team_abbr":  row.team_abbr,
                "is_home":    int(row.is_home),
                "opponent_abbr": row.opponent_abbr,
                "win":        row.win,
                # rolling features
                "roll_win_pct":  roll_mean.loc[i, "win"],
                "roll_pts":      roll_mean.loc[i, "pts"],
                "roll_off_rtg":  roll_mean.loc[i, "off_rtg"],
                "roll_ts":       roll_mean.loc[i, "ts"],
                "roll_pace":     roll_mean.loc[i, "pace"],
                "roll_tov":      roll_mean.loc[i, "tov"],
                "roll_oreb":     roll_mean.loc[i, "oreb"],
                "roll_reb":      roll_mean.loc[i, "reb"],
                "roll_ast":      roll_mean.loc[i, "ast"],
                "roll_fg3_rate": (roll_mean.loc[i, "fg3a"] / roll_mean.loc[i, "fga"]
                                  if roll_mean.loc[i, "fga"] > 0 else np.nan),
                "rest_days":     row.rest_days,
                # season-to-date win%
                "season_win_pct": (grp.loc[:i-1, "win"].sum() / max(i, 1)),
            }
            records.append(rec)

    return pd.DataFrame(records)


# ── Pair home/away into one game row ─────────────────────────────────────────
def build_game_features(team_feats: pd.DataFrame) -> pd.DataFrame:
    home = team_feats[team_feats["is_home"] == 1].copy()
    away = team_feats[team_feats["is_home"] == 0].copy()

    home = home.rename(columns={c: f"home_{c}" for c in team_feats.columns
                                 if c not in ("game_id", "game_date", "season")})
    away = away.rename(columns={c: f"away_{c}" for c in team_feats.columns
                                 if c not in ("game_id", "game_date", "season")})

    merged = home.merge(away, on=["game_id", "game_date", "season"], how="inner")

    # Differential features (home minus away)
    diff_cols = ["roll_win_pct", "roll_pts", "roll_off_rtg", "roll_ts",
                 "roll_pace", "roll_tov", "roll_oreb", "roll_reb", "roll_ast",
                 "roll_fg3_rate", "rest_days", "season_win_pct"]
    for c in diff_cols:
        merged[f"diff_{c}"] = merged[f"home_{c}"] - merged[f"away_{c}"]

    # Label: 1 = home team won
    merged["home_win"] = merged["home_win"].astype(int)

    return merged


# ── Opponent-adjusted offensive rating ───────────────────────────────────────
def add_opp_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate opp-adjusted off rating: home_roll_off_rtg - away_roll_off_rtg
    is already captured in diff_roll_off_rtg. We add the opponent's defensive
    proxy = opponent's rolling points-allowed (pts scored against them).
    For simplicity we approximate as: opp roll_pts (which is the opponent
    team's own scoring, not their defence). A true defensive rating needs the
    opponent's pts-allowed which requires joining from the other side of the game.
    We mark this as a TODO upgrade and keep the diff features for now.
    """
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    conn = sqlite3.connect(DB_PATH)
    log.info("Loading game logs from %s …", DB_PATH)
    raw = load_logs(conn)
    conn.close()

    log.info("Loaded %d team-game rows across %d teams",
             len(raw), raw["team_id"].nunique())

    log.info("Computing rolling %d-game team features …", WINDOW)
    team_feats = build_team_features(raw)
    log.info("Team feature rows: %d", len(team_feats))

    log.info("Pairing home/away into game rows …")
    game_feats = build_game_features(team_feats)
    game_feats = add_opp_adjusted(game_feats)
    log.info("Game feature rows: %d", len(game_feats))

    # Drop rows with any NaN in feature columns
    feature_cols = [c for c in game_feats.columns if c.startswith(("home_roll", "away_roll", "diff_"))]
    before = len(game_feats)
    game_feats = game_feats.dropna(subset=feature_cols)
    log.info("Dropped %d rows with NaN features → %d remaining", before - len(game_feats), len(game_feats))

    game_feats["game_date"] = game_feats["game_date"].astype(str)
    game_feats.to_csv(FEATURES_CSV, index=False)
    log.info("Saved → %s  (%d rows × %d cols)", FEATURES_CSV, *game_feats.shape)

    # Season summary
    for season, grp in game_feats.groupby("season"):
        log.info("  %s: %d games, %.1f%% home-win rate",
                 season, len(grp), grp["home_win"].mean() * 100)


if __name__ == "__main__":
    main()
