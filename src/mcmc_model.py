"""
Bayesian MCMC model for football match score prediction.

Uses PyMC v5 with the NUTS sampler to estimate posterior distributions
of team attack/defense strengths and home advantage. The model assumes
Poisson-distributed goals with log-linear intensity parameters.

This approach provides full posterior uncertainty quantification,
unlike the point-estimate Dixon-Coles MLE model.
"""

import os
from pathlib import Path
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


class MCMCModel:
    """
    Bayesian MCMC model for predicting football match scores.

    Uses a hierarchical Poisson model with:
    - Normal priors on attack/defense parameters per team.
    - Informative prior on home advantage (slight positive bias).
    - Sum-to-zero constraint on attack params for identifiability.
    - NUTS sampling via PyMC v5.
    - Trace caching to disk for reuse.

    Attributes:
        trace_path: Path to save/load the MCMC trace (NetCDF format).
        trace: ArviZ InferenceData object containing posterior samples.
        teams: List of team names from training data.
        team_to_idx: Mapping from team name to integer index.
        is_fitted: Whether the model has been fitted.
        posterior_means: Dict of posterior mean parameters after fitting.

    Example:
        >>> model = MCMCModel(trace_path="data/mcmc_trace")
        >>> model.fit(df, draws=2000, tune=1000, chains=2)
        >>> lam_h, lam_a = model.predict("Barcelona", "Real Madrid")
    """

    def __init__(self, trace_path: str = "data/mcmc_trace") -> None:
        """
        Initialize the MCMC model.

        Args:
            trace_path: File path (without extension) for caching the
                MCMC trace. The trace is saved as a NetCDF file with
                a .nc extension appended automatically.
        """
        self.trace_path = trace_path
        self.trace: Optional[az.InferenceData] = None
        self.teams: list[str] = []
        self.team_to_idx: dict[str, int] = {}
        self.is_fitted: bool = False
        self.posterior_means: dict = {}

    def fit(
        self,
        df: pd.DataFrame,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 2,
    ) -> "MCMCModel":
        """
        Fit the Bayesian model using MCMC (NUTS) sampling.

        Builds a PyMC model with Poisson likelihoods for home/away goals
        and samples from the posterior using the No-U-Turn Sampler.

        Model specification:
            attack_i ~ Normal(0, 1)       for each team i
            defense_i ~ Normal(0, 1)      for each team i
            home_adv ~ Normal(0.3, 0.5)   (informative prior)

            lambda_home = exp(attack[home] + defense[away] + home_adv)
            lambda_away = exp(attack[away] + defense[home])

            home_score ~ Poisson(lambda_home)
            away_score ~ Poisson(lambda_away)

        The attack parameters are constrained to sum to zero via a
        ZeroSumNormal distribution for identifiability.

        Args:
            df: Preprocessed DataFrame with required columns:
                - home_team (str): Name of the home team.
                - away_team (str): Name of the away team.
                - home_score (int): Goals scored by the home team.
                - away_score (int): Goals scored by the away team.
            draws: Number of posterior samples to draw per chain.
            tune: Number of tuning (burn-in) samples per chain.
            chains: Number of independent MCMC chains to run.

        Returns:
            self: The fitted model instance (for method chaining).

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        required_cols = {"home_team", "away_team", "home_score", "away_score"}
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
        home_idx = df["home_team"].map(self.team_to_idx).values.astype(int)
        away_idx = df["away_team"].map(self.team_to_idx).values.astype(int)
        home_goals = df["home_score"].values.astype(int)
        away_goals = df["away_score"].values.astype(int)

        print(f"Building PyMC model with {n_teams} teams and {len(df)} matches...")
        print(f"  Sampling: {draws} draws, {tune} tune, {chains} chains")

        with pm.Model() as football_model:
            # --- Data containers ---
            home_team_idx = pm.MutableData("home_team_idx", home_idx)
            away_team_idx = pm.MutableData("away_team_idx", away_idx)

            # --- Priors ---
            # ZeroSumNormal enforces sum-to-zero constraint on attack params
            attack = pm.ZeroSumNormal(
                "attack",
                sigma=1.0,
                shape=n_teams,
            )

            defense = pm.Normal(
                "defense",
                mu=0,
                sigma=1.0,
                shape=n_teams,
            )

            # Informative prior: slight positive home advantage expected
            home_adv = pm.Normal("home_adv", mu=0.3, sigma=0.5)

            # --- Expected goals (log-linear model) ---
            lambda_home = pm.math.exp(
                attack[home_team_idx] + defense[away_team_idx] + home_adv
            )
            lambda_away = pm.math.exp(
                attack[away_team_idx] + defense[home_team_idx]
            )

            # --- Likelihoods ---
            home_score_obs = pm.Poisson(
                "home_score",
                mu=lambda_home,
                observed=home_goals,
            )
            away_score_obs = pm.Poisson(
                "away_score",
                mu=lambda_away,
                observed=away_goals,
            )

            # --- Sampling ---
            print("Starting NUTS sampling...")
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                return_inferencedata=True,
                progressbar=True,
                random_seed=42,
            )

        # Cache trace to disk
        self._save_trace()

        # Extract posterior means
        self._compute_posterior_means()

        self.is_fitted = True

        print("MCMC sampling complete.")
        print(f"  Home advantage (posterior mean): {self.posterior_means['home_adv']:.4f}")
        print(f"  Trace saved to: {self.trace_path}.nc")

        return self

    def predict(self, home_team: str, away_team: str) -> tuple[float, float]:
        """
        Predict expected goals using posterior means.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.

        Returns:
            Tuple of (lambda_home, lambda_away) representing the expected
            goals for the home and away teams respectively, computed from
            posterior mean parameters.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If either team is not in the training data.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before predict()."
            )

        self._validate_teams(home_team, away_team)

        home_i = self.team_to_idx[home_team]
        away_i = self.team_to_idx[away_team]

        attack_home = self.posterior_means["attack"][home_i]
        defense_home = self.posterior_means["defense"][home_i]
        attack_away = self.posterior_means["attack"][away_i]
        defense_away = self.posterior_means["defense"][away_i]
        home_adv = self.posterior_means["home_adv"]

        lambda_home = np.exp(attack_home + defense_away + home_adv)
        lambda_away = np.exp(attack_away + defense_home)

        return float(lambda_home), float(lambda_away)

    def get_team_ratings(self) -> pd.DataFrame:
        """
        Get a DataFrame of posterior mean team ratings.

        Returns:
            DataFrame with columns:
                - team: Team name.
                - attack: Posterior mean attack strength.
                - defense: Posterior mean defense parameter.
                - overall: Combined rating (attack - defense).
            Sorted by overall descending.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before get_team_ratings()."
            )

        ratings = []
        for i, team in enumerate(self.teams):
            attack = self.posterior_means["attack"][i]
            defense = self.posterior_means["defense"][i]
            ratings.append(
                {
                    "team": team,
                    "attack": float(attack),
                    "defense": float(defense),
                    "overall": float(attack - defense),
                }
            )

        ratings_df = pd.DataFrame(ratings)
        ratings_df = ratings_df.sort_values("overall", ascending=False).reset_index(
            drop=True
        )
        return ratings_df

    def load_trace(self, path: Optional[str] = None) -> "MCMCModel":
        """
        Load a previously cached MCMC trace from disk.

        Args:
            path: Path to the NetCDF trace file. If None, uses
                the default trace_path set at initialization.

        Returns:
            self: The model instance with loaded trace.

        Raises:
            FileNotFoundError: If the trace file does not exist.
        """
        load_path = path or f"{self.trace_path}.nc"
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"No cached trace found at '{load_path}'. "
                "Run fit() first to generate a trace."
            )

        print(f"Loading cached trace from: {load_path}")
        self.trace = az.from_netcdf(load_path)

        # Reconstruct team mapping from trace dimensions
        # The trace stores coordinate labels for attack/defense dimensions
        if hasattr(self.trace.posterior, "attack"):
            n_teams = self.trace.posterior["attack"].shape[-1]
            if not self.teams:
                print(
                    f"Warning: Team names not available. "
                    f"Loading {n_teams} teams with index-based names."
                )
                self.teams = [f"team_{i}" for i in range(n_teams)]
                self.team_to_idx = {t: i for i, t in enumerate(self.teams)}

        self._compute_posterior_means()
        self.is_fitted = True
        print("Trace loaded successfully.")
        return self

    def _save_trace(self) -> None:
        """Save the MCMC trace to disk as NetCDF."""
        if self.trace is None:
            return

        # Create directory if it doesn't exist
        trace_dir = os.path.dirname(self.trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)

        save_path = f"{self.trace_path}.nc"
        self.trace.to_netcdf(save_path)
        print(f"Trace cached to: {save_path}")

    def _compute_posterior_means(self) -> None:
        """Extract posterior mean parameters from the trace."""
        if self.trace is None:
            return

        posterior = self.trace.posterior

        self.posterior_means = {
            "attack": posterior["attack"].mean(dim=["chain", "draw"]).values,
            "defense": posterior["defense"].mean(dim=["chain", "draw"]).values,
            "home_adv": float(
                posterior["home_adv"].mean(dim=["chain", "draw"]).values
            ),
        }

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
