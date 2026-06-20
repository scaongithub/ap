"""
Dixon-Coles model for football match score prediction.

Implements the Dixon & Coles (1997) model which extends the independent
Poisson model by adding a correction factor (tau) for low-scoring outcomes
and time-decay weighting for recent matches.

References:
    Dixon, M. J. & Coles, S. G. (1997). Modelling Association Football
    Scores and Inefficiencies in the Football Betting Market.
    Journal of the Royal Statistical Society: Series C, 46(2), 265-280.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def tau(x: int, y: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """
    Dixon-Coles correction factor for low-scoring outcomes.

    Adjusts the independent Poisson probabilities for match results
    (0,0), (1,0), (0,1), and (1,1) to account for the observed
    dependency between home and away scores at low values.

    Args:
        x: Home team goals scored.
        y: Away team goals scored.
        lambda_home: Expected goals for the home team.
        lambda_away: Expected goals for the away team.
        rho: Dependence parameter (typically small, around -0.1 to 0.1).

    Returns:
        Multiplicative correction factor. Returns 1.0 for scores
        not in {(0,0), (1,0), (0,1), (1,1)}.
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_home * lambda_away * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_home * rho
    elif x == 1 and y == 0:
        return 1.0 + lambda_away * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


def dc_log_likelihood(
    params: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    """
    Negative weighted log-likelihood for the Dixon-Coles model.

    This is the objective function minimized during model fitting. It computes
    the sum of weighted negative log-likelihoods across all matches, including
    the tau correction for low-scoring outcomes.

    Parameter vector layout:
        params[0 : n_teams]           -> attack parameters (one per team)
        params[n_teams : 2*n_teams]   -> defense parameters (one per team)
        params[2*n_teams]             -> home advantage (scalar)
        params[2*n_teams + 1]         -> rho (dependence parameter)

    Args:
        params: Flat parameter vector of length (2 * n_teams + 2).
        home_goals: Array of home goals scored per match.
        away_goals: Array of away goals scored per match.
        home_idx: Integer-encoded home team indices per match.
        away_idx: Integer-encoded away team indices per match.
        weights: Time-decay weights per match (higher = more recent/important).
        n_teams: Number of unique teams.

    Returns:
        Negative weighted log-likelihood (scalar). Lower is better.
    """
    attack = params[:n_teams]
    defense = params[n_teams : 2 * n_teams]
    home_adv = params[2 * n_teams]
    rho_val = params[2 * n_teams + 1]

    # Expected goals
    lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
    lambda_away = np.exp(attack[away_idx] + defense[home_idx])

    # Poisson log-likelihoods
    log_lik_home = poisson.logpmf(home_goals, lambda_home)
    log_lik_away = poisson.logpmf(away_goals, lambda_away)

    # Dixon-Coles tau correction (fully vectorized via boolean masks).
    # tau only deviates from 1.0 for the four low-scoring outcomes, so we
    # avoid a per-match Python loop (which is O(n) interpreted overhead on
    # every optimizer iteration) and instead assign each case with a mask.
    tau_values = np.ones(len(home_goals))
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_11 = (home_goals == 1) & (away_goals == 1)
    tau_values[mask_00] = 1.0 - lambda_home[mask_00] * lambda_away[mask_00] * rho_val
    tau_values[mask_01] = 1.0 + lambda_home[mask_01] * rho_val
    tau_values[mask_10] = 1.0 + lambda_away[mask_10] * rho_val
    tau_values[mask_11] = 1.0 - rho_val

    # Clamp tau to avoid log(0) or log(negative)
    tau_values = np.clip(tau_values, 1e-10, None)

    log_lik = log_lik_home + log_lik_away + np.log(tau_values)

    # Weighted sum (negate for minimization)
    return -np.sum(weights * log_lik)


class DixonColesModel:
    """
    Dixon-Coles model for predicting football match scores.

    Extends the independent Poisson model with:
    - A correction factor (tau/rho) for dependency in low-scoring outcomes.
    - Time-decay weighting so recent matches have more influence.
    - Home advantage parameter.
    - Sum-to-zero constraint on attack parameters for identifiability.

    Attributes:
        params: Dictionary of fitted model parameters. Keys include
            'attack', 'defense' (dicts mapping team -> value),
            'home_adv' (float), and 'rho' (float).
        teams: List of team names in the training data.
        team_to_idx: Mapping from team name to integer index.
        is_fitted: Whether the model has been fitted.

    Example:
        >>> model = DixonColesModel()
        >>> model.fit(df)
        >>> lam_h, lam_a = model.predict("Barcelona", "Real Madrid")
    """

    def __init__(self) -> None:
        """Initialize the Dixon-Coles model with empty parameters."""
        self.params: dict = {}
        self.teams: list[str] = []
        self.team_to_idx: dict[str, int] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "DixonColesModel":
        """
        Fit the Dixon-Coles model via maximum likelihood estimation.

        Uses L-BFGS-B optimization to find the MLE of attack parameters,
        defense parameters, home advantage, and the rho correction factor.
        A sum-to-zero constraint on attack parameters ensures identifiability.

        Args:
            df: Preprocessed DataFrame with required columns:
                - home_team (str): Name of the home team.
                - away_team (str): Name of the away team.
                - home_score (int): Goals scored by the home team.
                - away_score (int): Goals scored by the away team.
                - time_weight (float): Decay weight (higher = more recent).

        Returns:
            self: The fitted model instance (for method chaining).

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        required_cols = {"home_team", "away_team", "home_score", "away_score", "time_weight"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {missing}. "
                f"Expected columns: {sorted(required_cols)}"
            )

        # Build team index mapping
        self.teams = sorted(
            list(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        )
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        n_teams = len(self.teams)

        # Encode teams as integer indices
        home_idx = df["home_team"].map(self.team_to_idx).values
        away_idx = df["away_team"].map(self.team_to_idx).values
        home_goals = df["home_score"].values.astype(float)
        away_goals = df["away_score"].values.astype(float)
        # Prefer the combined sample weight (time decay x tournament importance)
        # when available so that World Cup / competitive matches dominate the
        # fit over low-signal friendlies. Falls back to pure time decay.
        weight_col = "sample_weight" if "sample_weight" in df.columns else "time_weight"
        weights = df[weight_col].values.astype(float)

        # Initial parameter guesses
        # attack_i = 0 for all teams, defense_i = 0, home_adv = 0.25, rho = -0.03
        x0 = np.zeros(2 * n_teams + 2)
        x0[2 * n_teams] = 0.25  # Initial home advantage
        x0[2 * n_teams + 1] = -0.03  # Initial rho

        # Sum-to-zero constraint on attack parameters
        def attack_sum_constraint(params):
            return np.sum(params[:n_teams])

        constraints = [{"type": "eq", "fun": attack_sum_constraint}]

        # Parameter bounds: rho bounded to avoid numerical issues
        bounds = (
            [(None, None)] * n_teams  # attack params unbounded
            + [(None, None)] * n_teams  # defense params unbounded
            + [(None, None)]  # home_adv unbounded
            + [(-0.99, 0.99)]  # rho bounded
        )

        print(f"Fitting Dixon-Coles model with {n_teams} teams and {len(df)} matches...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                dc_log_likelihood,
                x0,
                args=(home_goals, away_goals, home_idx, away_idx, weights, n_teams),
                method="SLSQP",  # Supports equality constraints
                constraints=constraints,
                bounds=bounds,
                options={"maxiter": 500, "disp": False, "ftol": 1e-8},
            )

        if not result.success:
            warnings.warn(
                f"Dixon-Coles optimization did not fully converge: {result.message}. "
                "Results may still be usable but should be interpreted with caution.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Store fitted parameters
        fitted = result.x
        attack_vals = fitted[:n_teams]
        defense_vals = fitted[n_teams : 2 * n_teams]
        home_adv = fitted[2 * n_teams]
        rho_val = fitted[2 * n_teams + 1]

        self.params = {
            "attack": {team: attack_vals[i] for i, team in enumerate(self.teams)},
            "defense": {team: defense_vals[i] for i, team in enumerate(self.teams)},
            "home_adv": home_adv,
            "rho": rho_val,
        }
        self.is_fitted = True

        print(f"Dixon-Coles model fitted successfully.")
        print(f"  Home advantage: {home_adv:.4f}")
        print(f"  Rho (correction): {rho_val:.4f}")
        print(f"  Optimization converged: {result.success}")

        return self

    def predict(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> tuple[float, float]:
        """
        Predict expected goals for a match between two teams.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.
            neutral: If True, the match is played at a neutral venue (e.g. a
                World Cup fixture). The home-advantage term is dropped because
                the home/away designation is arbitrary at neutral sites, so
                applying it would systematically bias the listed home team.

        Returns:
            Tuple of (lambda_home, lambda_away) representing the expected
            goals for the home and away teams respectively.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If either team is not in the training data.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before predict()."
            )

        self._validate_teams(home_team, away_team)

        attack_home = self.params["attack"][home_team]
        defense_home = self.params["defense"][home_team]
        attack_away = self.params["attack"][away_team]
        defense_away = self.params["defense"][away_team]
        # Zero out home advantage on neutral ground (World Cup default).
        home_adv = 0.0 if neutral else self.params["home_adv"]

        lambda_home = np.exp(attack_home + defense_away + home_adv)
        lambda_away = np.exp(attack_away + defense_home)

        return float(lambda_home), float(lambda_away)

    def get_team_ratings(self) -> pd.DataFrame:
        """
        Get a DataFrame of team ratings sorted by overall strength.

        Returns:
            DataFrame with columns:
                - team: Team name.
                - attack: Attack strength parameter (higher = better attack).
                - defense: Defense parameter (lower = better defense).
                - overall: Combined rating (attack - defense).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before get_team_ratings()."
            )

        ratings = []
        for team in self.teams:
            attack = self.params["attack"][team]
            defense = self.params["defense"][team]
            ratings.append(
                {
                    "team": team,
                    "attack": attack,
                    "defense": defense,
                    "overall": attack - defense,
                }
            )

        ratings_df = pd.DataFrame(ratings)
        ratings_df = ratings_df.sort_values("overall", ascending=False).reset_index(
            drop=True
        )
        return ratings_df

    def _validate_teams(self, *teams: str) -> None:
        """
        Validate that all provided team names exist in the training data.

        Args:
            *teams: One or more team name strings to validate.

        Raises:
            ValueError: If any team is not found, with a suggestion of
                similar team names from the training data.
        """
        for team in teams:
            if team not in self.team_to_idx:
                # Find similar team names for helpful error message
                available = sorted(self.teams)
                suggestions = [
                    t for t in available if team.lower() in t.lower()
                ]
                msg = f"Team '{team}' not found in training data."
                if suggestions:
                    msg += f" Did you mean one of: {suggestions}?"
                else:
                    msg += (
                        f" Available teams ({len(available)}): "
                        f"{available[:10]}{'...' if len(available) > 10 else ''}"
                    )
                raise ValueError(msg)
