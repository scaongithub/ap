"""
Data ingestion and feature engineering module for the Football Match Score Predictor.

Handles downloading historical international football results, preprocessing,
and computing rolling statistical features for each team.
"""

import os
import logging
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "Copa América": 0.8,
    "Copa America": 0.8,
    "UEFA Euro": 0.8,
    "UEFA Euro qualification": 0.7,
    "African Cup of Nations": 0.8,
    "AFC Asian Cup": 0.8,
    "Gold Cup": 0.8,
    "CONCACAF Nations League": 0.8,
    "UEFA Nations League": 0.8,
    "FIFA World Cup qualification": 0.7,
    "Friendly": 0.4,
}
DEFAULT_TOURNAMENT_WEIGHT = 0.6
TIME_DECAY_LAMBDA = 0.003
CACHE_MAX_AGE_DAYS = 7


# ===================================================================
# 1.  fetch_data
# ===================================================================
def fetch_data(cache_dir: str = "data") -> pd.DataFrame:
    """Download (or load from cache) international football results.

    Parameters
    ----------
    cache_dir : str
        Directory used for caching the downloaded CSV. Defaults to ``data``.

    Returns
    -------
    pd.DataFrame
        Raw results with columns: date, home_team, away_team, home_score,
        away_score, tournament, city, country, neutral.
    """
    cache_path = Path(cache_dir) / "results.csv"

    # Check cache freshness ---------------------------------------------------
    if cache_path.exists():
        file_age = datetime.now() - datetime.fromtimestamp(
            cache_path.stat().st_mtime
        )
        if file_age < timedelta(days=CACHE_MAX_AGE_DAYS):
            print(f"[cache] Using cached data ({file_age.days}d old): {cache_path}")
            logger.info("Loading cached data from %s", cache_path)
            return pd.read_csv(cache_path)
        else:
            print(f"[cache] Cache expired ({file_age.days}d old). Re-downloading …")

    # Download ----------------------------------------------------------------
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching data from {DATA_URL} …")
    try:
        urllib.request.urlretrieve(DATA_URL, cache_path)
        print(f"[download] Saved to {cache_path}")
    except Exception as exc:
        logger.error("Failed to download data: %s", exc)
        if cache_path.exists():
            print("[download] Download failed – falling back to stale cache.")
            return pd.read_csv(cache_path)
        raise RuntimeError(
            f"Could not download data and no cache available: {exc}"
        ) from exc

    return pd.read_csv(cache_path)


# ===================================================================
# 2.  preprocess
# ===================================================================
def preprocess(df: pd.DataFrame, start_year: int = 2018) -> pd.DataFrame:
    """Clean and enrich the raw results DataFrame.

    Steps
    -----
    1. Parse ``date`` to datetime and filter to ``start_year`` onwards.
    2. Strip whitespace / title-case team names.
    3. Drop rows with missing scores.
    4. Add ``days_ago`` and exponential ``time_weight``.
    5. Add ``tournament_weight`` based on competition importance.

    Parameters
    ----------
    df : pd.DataFrame
        Raw results from :func:`fetch_data`.
    start_year : int
        Keep only matches from this year onwards (inclusive).

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame sorted by date ascending.
    """
    df = df.copy()

    # Date parsing & filtering ------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_before = len(df)
    df = df[df["date"].dt.year >= start_year].reset_index(drop=True)
    print(
        f"[preprocess] Filtered to {start_year}+: "
        f"{n_before:,} → {len(df):,} matches"
    )

    # Team name normalisation --------------------------------------------------
    for col in ("home_team", "away_team"):
        df[col] = df[col].astype(str).str.strip().str.title()

    # Drop missing scores -----------------------------------------------------
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Time-based columns -------------------------------------------------------
    today = pd.Timestamp.now().normalize()
    df["days_ago"] = (today - df["date"]).dt.days
    df["time_weight"] = np.exp(-TIME_DECAY_LAMBDA * df["days_ago"])

    # Tournament weight --------------------------------------------------------
    df["tournament_weight"] = df["tournament"].map(
        lambda t: _match_tournament_weight(t)
    )

    # Combined sample weight ---------------------------------------------------
    # Multiply time decay by tournament importance so that recent *and*
    # competitive matches (World Cup, continental cups) dominate the fit, while
    # stale friendlies contribute little signal. Models that support weighting
    # (e.g. Dixon-Coles) consume this column directly.
    df["sample_weight"] = df["time_weight"] * df["tournament_weight"]

    # Sort chronologically -----------------------------------------------------
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[preprocess] Done – {len(df):,} matches ready")
    return df


def _match_tournament_weight(tournament: str) -> float:
    """Return the importance weight for a tournament name.

    Performs a case-insensitive substring match against the known tournament
    mapping so that slight naming variations (e.g.
    ``"FIFA World Cup qualification"`` vs ``"FIFA World Cup Qualifier"``) are
    still captured.
    """
    if not isinstance(tournament, str):
        return DEFAULT_TOURNAMENT_WEIGHT

    t_lower = tournament.strip().lower()

    # Exact / direct lookup first
    for key, weight in TOURNAMENT_WEIGHTS.items():
        if key.lower() == t_lower:
            return weight

    # Substring / fuzzy fallback
    if "world cup" in t_lower and "qualif" in t_lower:
        return 0.7
    if "world cup" in t_lower:
        return 1.0
    if any(
        kw in t_lower
        for kw in [
            "euro",
            "copa am",
            "african cup",
            "asian cup",
            "gold cup",
            "nations league",
        ]
    ):
        # Qualifiers for continental tournaments
        if "qualif" in t_lower:
            return 0.7
        return 0.8
    if "friendly" in t_lower:
        return 0.4

    return DEFAULT_TOURNAMENT_WEIGHT


# ===================================================================
# 3.  engineer_features
# ===================================================================
def engineer_features(
    df: pd.DataFrame,
    n_rolling: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Compute rolling team-level features for every match row.

    For each window size *N* in ``n_rolling`` and for both the home and
    away side, the following features are produced:

    * ``{side}_goals_scored_{N}``   – rolling mean of goals scored
    * ``{side}_goals_conceded_{N}`` – rolling mean of goals conceded
    * ``{side}_win_rate_{N}``       – rolling win proportion
    * ``{side}_days_since_last``    – days since the team's previous match
    * ``{side}_h2h_wins_5``         – wins in last 5 head-to-head meetings

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed match data (from :func:`preprocess`).
    n_rolling : list[int], optional
        Rolling window sizes.  Defaults to ``[5, 10, 20]``.

    Returns
    -------
    pd.DataFrame
        Original match columns plus all engineered feature columns.
        Rows where any feature is still NaN (early matches) are dropped.
    """
    if n_rolling is None:
        n_rolling = [5, 10, 20]

    df = df.copy().sort_values("date").reset_index(drop=True)
    print(f"[features] Engineering rolling features for windows {n_rolling} …")

    # -- Build per-team history ------------------------------------------------
    # We create a "long" view: one row per team per match, recording whether
    # the team was home or away, goals scored/conceded, and result.
    records: list[dict] = []
    for idx, row in df.iterrows():
        base = {
            "match_idx": idx,
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
        }
        # Home perspective
        records.append(
            {
                **base,
                "team": row["home_team"],
                "opponent": row["away_team"],
                "goals_scored": row["home_score"],
                "goals_conceded": row["away_score"],
                "win": int(row["home_score"] > row["away_score"]),
                "side": "home",
            }
        )
        # Away perspective
        records.append(
            {
                **base,
                "team": row["away_team"],
                "opponent": row["home_team"],
                "goals_scored": row["away_score"],
                "goals_conceded": row["home_score"],
                "win": int(row["away_score"] > row["home_score"]),
                "side": "away",
            }
        )

    long = pd.DataFrame(records).sort_values(["team", "date"]).reset_index(drop=True)

    # -- Rolling statistics per team -------------------------------------------
    team_features: dict[int, dict] = {}  # match_idx -> {col: val, ...}

    for team, grp in long.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for n in n_rolling:
            grp[f"goals_scored_{n}"] = (
                grp["goals_scored"].rolling(n, min_periods=1).mean().shift(1)
            )
            grp[f"goals_conceded_{n}"] = (
                grp["goals_conceded"].rolling(n, min_periods=1).mean().shift(1)
            )
            grp[f"win_rate_{n}"] = (
                grp["win"].rolling(n, min_periods=1).mean().shift(1)
            )

        # Days since last match ------------------------------------------------
        grp["days_since_last"] = grp["date"].diff().dt.days.fillna(0)

        # Write features back keyed by (match_idx, side) ----------------------
        for _, r in grp.iterrows():
            midx = r["match_idx"]
            side = r["side"]
            if midx not in team_features:
                team_features[midx] = {}
            for n in n_rolling:
                team_features[midx][f"{side}_goals_scored_{n}"] = r[
                    f"goals_scored_{n}"
                ]
                team_features[midx][f"{side}_goals_conceded_{n}"] = r[
                    f"goals_conceded_{n}"
                ]
                team_features[midx][f"{side}_win_rate_{n}"] = r[f"win_rate_{n}"]
            team_features[midx][f"{side}_days_since_last"] = r["days_since_last"]

    # -- Head-to-head record ---------------------------------------------------
    h2h_wins = _compute_h2h(df, last_n=5)
    for midx, vals in h2h_wins.items():
        if midx not in team_features:
            team_features[midx] = {}
        team_features[midx].update(vals)

    # -- Merge features back onto the match DataFrame -------------------------
    feat_df = pd.DataFrame.from_dict(team_features, orient="index")
    feat_df.index.name = "match_idx"
    df = df.join(feat_df)

    # Keep canonical output columns -------------------------------------------
    base_cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
        "time_weight",
        "tournament_weight",
        "sample_weight",
    ]
    feature_cols = sorted(
        [c for c in df.columns if c not in base_cols and c not in ("city", "country", "days_ago")]
    )
    df = df[base_cols + feature_cols]

    # Drop rows with NaN features (beginning of dataset) ----------------------
    n_before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    print(
        f"[features] Dropped {n_before - len(df)} rows with incomplete features, "
        f"{len(df):,} matches remain"
    )
    return df


def _compute_h2h(df: pd.DataFrame, last_n: int = 5) -> dict:
    """Compute head-to-head win counts for each match.

    For each match at index *i*, look back at the previous ``last_n``
    meetings between the same two teams and count wins for the home and
    away side respectively.

    Returns
    -------
    dict[int, dict]
        ``{match_idx: {"home_h2h_wins_5": int, "away_h2h_wins_5": int}}``
    """
    h2h: dict[int, dict] = {}

    # Group by the unordered pair of teams
    pair_key = df.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )
    df_tmp = df.copy()
    df_tmp["_pair"] = pair_key

    for _, grp in df_tmp.groupby("_pair"):
        grp = grp.sort_values("date")
        indices = grp.index.tolist()
        for pos, idx in enumerate(indices):
            row = grp.loc[idx]
            # Previous meetings
            prev = grp.iloc[max(0, pos - last_n) : pos]
            home_wins = 0
            away_wins = 0
            for _, prev_row in prev.iterrows():
                if prev_row["home_score"] > prev_row["away_score"]:
                    winner = prev_row["home_team"]
                elif prev_row["away_score"] > prev_row["home_score"]:
                    winner = prev_row["away_team"]
                else:
                    continue  # draw
                if winner == row["home_team"]:
                    home_wins += 1
                elif winner == row["away_team"]:
                    away_wins += 1
            h2h[idx] = {
                f"home_h2h_wins_{last_n}": home_wins,
                f"away_h2h_wins_{last_n}": away_wins,
            }

    return h2h


# ===================================================================
# 4.  Utility helpers
# ===================================================================
def get_team_list(df: pd.DataFrame) -> List[str]:
    """Return a sorted list of all unique team names in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Any DataFrame containing ``home_team`` and ``away_team`` columns.

    Returns
    -------
    list[str]
        Alphabetically sorted unique team names.
    """
    teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    return sorted(teams)


def get_team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return all matches involving a specific team.

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame with ``home_team`` and ``away_team`` columns.
    team : str
        Team name (case-insensitive title-case match).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame sorted by date.

    Raises
    ------
    ValueError
        If the team is not found in the dataset.
    """
    team = team.strip().title()
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    result = df[mask].sort_values("date").reset_index(drop=True)
    if result.empty:
        available = get_team_list(df)
        raise ValueError(
            f"Team '{team}' not found. Available teams: {available[:20]} …"
        )
    return result


# ===================================================================
# 5.  Quick smoke test
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Football Match Score Predictor – Data Loader Smoke Test")
    print("=" * 60)

    raw = fetch_data()
    print(f"\nRaw data shape: {raw.shape}")
    print(raw.head(3))

    processed = preprocess(raw)
    print(f"\nProcessed shape: {processed.shape}")
    print(processed[["date", "home_team", "away_team", "time_weight", "tournament_weight"]].head(5))

    featured = engineer_features(processed)
    print(f"\nFeatured shape: {featured.shape}")
    print(featured.columns.tolist())

    teams = get_team_list(featured)
    print(f"\n{len(teams)} teams. First 10: {teams[:10]}")

    spain = get_team_matches(featured, "Spain")
    print(f"\nSpain matches: {len(spain)}")
    print(spain.tail(3))
