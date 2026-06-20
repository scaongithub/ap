"""
Ensemble Predictor — Optimized blending of MCMC and XGBoost predictions.

Combines Bayesian (MCMC) team-strength estimates with XGBoost feature-based
expected-goals predictions through a weighted average.  The optimal blending
weight is chosen by minimising the Brier Score over a held-out validation
period, using both a coarse grid search and fine-grained scalar optimisation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.poisson_matrix import build_score_matrix


class EnsemblePredictor:
    """Blend MCMC and XGBoost goal predictions with an optimised weight.

    The ensemble lambda is computed as::

        lambda_final = w * lambda_xgb + (1 - w) * lambda_mcmc

    where *w* ∈ [0, 1] is optimised to minimise the multiclass Brier Score
    on a validation set of historical matches.

    Attributes:
        mcmc_model: Fitted MCMC model (must expose ``predict``).
        xgboost_model: Fitted XGBoostGoalModel.
        dixon_coles_model: Optional Dixon-Coles model (reserved for future
            three-way blending).
        optimal_w: Optimal XGBoost weight after calling :meth:`optimize_weight`.
    """

    def __init__(
        self,
        mcmc_model: Any,
        xgboost_model: Any,
        dixon_coles_model: Any | None = None,
    ) -> None:
        self.mcmc_model = mcmc_model
        self.xgboost_model = xgboost_model
        self.dixon_coles_model = dixon_coles_model
        self.optimal_w: float | None = None

    # ------------------------------------------------------------------
    # Weight optimisation
    # ------------------------------------------------------------------

    def optimize_weight(self, df: pd.DataFrame, val_year: int = 2024) -> float:
        """Find the blending weight that minimises the Brier Score.

        For every match in the validation set (``date.year >= val_year``):

        1. Obtain ``(lambda_home, lambda_away)`` from both MCMC and XGBoost.
        2. For each candidate weight *w*, blend the lambdas and build a
           Poisson score-probability matrix.
        3. Compute the multiclass Brier Score against the actual outcome.

        A coarse grid search (21 points) is followed by
        ``scipy.optimize.minimize_scalar`` for finer precision.

        Args:
            df: Feature-engineered DataFrame.  Must contain ``date``,
                ``home_team``, ``away_team``, ``home_score``, and
                ``away_score`` columns plus all XGBoost feature columns.
            val_year: Year at which the validation window starts.

        Returns:
            The optimal weight *w* (also stored in ``self.optimal_w``).
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        val_df = df[df["date"].dt.year >= val_year].dropna(
            subset=["home_score", "away_score"]
        )

        if val_df.empty:
            print("[Ensemble] ⚠ No validation matches found. Defaulting w=0.5.")
            self.optimal_w = 0.5
            return self.optimal_w

        print(f"[Ensemble] Validation matches: {len(val_df):,}")
        print("[Ensemble] Collecting per-match predictions …")

        # Pre-compute predictions for every validation match
        match_preds: list[dict] = []
        skipped = 0

        for _, row in val_df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            actual_home = int(row["home_score"])
            actual_away = int(row["away_score"])

            try:
                mcmc_home, mcmc_away = self._get_mcmc_prediction(
                    home_team, away_team
                )
                xgb_home, xgb_away = self._get_xgb_prediction(
                    df, home_team, away_team
                )
            except Exception:  # noqa: BLE001
                skipped += 1
                continue

            match_preds.append(
                {
                    "mcmc_home": mcmc_home,
                    "mcmc_away": mcmc_away,
                    "xgb_home": xgb_home,
                    "xgb_away": xgb_away,
                    "actual_home": actual_home,
                    "actual_away": actual_away,
                }
            )

        if skipped:
            print(f"[Ensemble] Skipped {skipped} matches (prediction errors).")

        if not match_preds:
            print("[Ensemble] ⚠ No usable match predictions. Defaulting w=0.5.")
            self.optimal_w = 0.5
            return self.optimal_w

        print(f"[Ensemble] Usable predictions: {len(match_preds):,}")

        # --- Coarse grid search ---
        grid_weights = np.linspace(0, 1, 21)
        grid_scores: list[float] = []

        print("[Ensemble] Coarse grid search (21 points) …")

        for w in grid_weights:
            bs = self._compute_brier_for_weight(w, match_preds)
            grid_scores.append(bs)

        best_grid_idx = int(np.argmin(grid_scores))
        best_grid_w = grid_weights[best_grid_idx]
        best_grid_bs = grid_scores[best_grid_idx]

        print(f"  Grid best w = {best_grid_w:.2f}  (Brier = {best_grid_bs:.6f})")

        # --- Fine-grained optimisation ---
        print("[Ensemble] Fine-grained optimisation (minimize_scalar) …")

        result = minimize_scalar(
            lambda w: self._compute_brier_for_weight(w, match_preds),
            bounds=(0.0, 1.0),
            method="bounded",
        )

        fine_w = float(result.x)
        fine_bs = float(result.fun)
        print(f"  Optimised w  = {fine_w:.4f}  (Brier = {fine_bs:.6f})")

        # Pick the better of grid vs. fine
        if fine_bs <= best_grid_bs:
            self.optimal_w = fine_w
        else:
            self.optimal_w = best_grid_w

        # Summary
        print("\n" + "=" * 55)
        print("  ENSEMBLE WEIGHT OPTIMISATION SUMMARY")
        print("=" * 55)
        print(f"  Optimal w (XGBoost weight) : {self.optimal_w:.4f}")
        print(f"  MCMC weight                : {1 - self.optimal_w:.4f}")
        print(f"  Best Brier Score           : {min(fine_bs, best_grid_bs):.6f}")
        print(f"  Validation matches used    : {len(match_preds):,}")
        print("=" * 55 + "\n")

        return self.optimal_w

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        home_team: str,
        away_team: str,
        df: pd.DataFrame | None = None,
    ) -> tuple[float, float]:
        """Produce blended expected-goals using the optimal weight.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            df: Feature-engineered DataFrame (required for XGBoost
                ``predict_from_teams``).

        Returns:
            ``(lambda_home, lambda_away)`` blended predictions.

        Raises:
            RuntimeError: If ``optimize_weight`` has not been called.
        """
        if self.optimal_w is None:
            raise RuntimeError(
                "Optimal weight has not been set.  "
                "Call optimize_weight() first."
            )

        mcmc_home, mcmc_away = self._get_mcmc_prediction(home_team, away_team)
        xgb_home, xgb_away = self._get_xgb_prediction(df, home_team, away_team)

        w = self.optimal_w
        lambda_home = w * xgb_home + (1 - w) * mcmc_home
        lambda_away = w * xgb_away + (1 - w) * mcmc_away

        return lambda_home, lambda_away

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def get_model_comparison(
        self,
        home_team: str,
        away_team: str,
        df: pd.DataFrame | None = None,
    ) -> dict:
        """Return a diagnostic dict contrasting each model's predictions.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            df: Feature-engineered DataFrame (needed for XGBoost).

        Returns:
            Dictionary with keys ``mcmc``, ``xgboost``, ``ensemble``, each
            mapping to ``{"lambda_home": …, "lambda_away": …}``, plus
            ``optimal_w``.
        """
        mcmc_home, mcmc_away = self._get_mcmc_prediction(home_team, away_team)
        xgb_home, xgb_away = self._get_xgb_prediction(df, home_team, away_team)

        comparison: dict = {
            "mcmc": {"lambda_home": mcmc_home, "lambda_away": mcmc_away},
            "xgboost": {"lambda_home": xgb_home, "lambda_away": xgb_away},
            "optimal_w": self.optimal_w,
        }

        if self.optimal_w is not None:
            w = self.optimal_w
            comparison["ensemble"] = {
                "lambda_home": w * xgb_home + (1 - w) * mcmc_home,
                "lambda_away": w * xgb_away + (1 - w) * mcmc_away,
            }
        else:
            comparison["ensemble"] = None

        if self.dixon_coles_model is not None:
            try:
                dc_home, dc_away = self.dixon_coles_model.predict(
                    home_team, away_team
                )
                comparison["dixon_coles"] = {
                    "lambda_home": dc_home,
                    "lambda_away": dc_away,
                }
            except Exception:  # noqa: BLE001
                comparison["dixon_coles"] = None

        return comparison

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _get_mcmc_prediction(
        self, home_team: str, away_team: str
    ) -> tuple[float, float]:
        """Obtain MCMC model predictions, handling common interface variants."""
        try:
            result = self.mcmc_model.predict(home_team, away_team)
        except TypeError:
            # Some MCMC wrappers accept keyword arguments only
            result = self.mcmc_model.predict(
                home_team=home_team, away_team=away_team
            )
        if isinstance(result, dict):
            return float(result["lambda_home"]), float(result["lambda_away"])
        return float(result[0]), float(result[1])

    def _get_xgb_prediction(
        self,
        df: pd.DataFrame | None,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float]:
        """Obtain XGBoost predictions, falling back to ``predict_from_teams``."""
        if df is not None:
            return self.xgboost_model.predict_from_teams(df, home_team, away_team)
        # Attempt a direct predict if df unavailable (caller must ensure
        # the xgboost_model supports this).
        return self.xgboost_model.predict_from_teams(
            pd.DataFrame(), home_team, away_team
        )

    @staticmethod
    def _compute_brier_for_weight(
        w: float,
        match_preds: list[dict],
        max_goals: int = 7,
    ) -> float:
        """Compute mean multiclass Brier Score for a given blending weight.

        The Brier Score is defined as the mean squared difference between the
        predicted probability vector and the one-hot actual outcome, averaged
        across all matches and all score-line cells.

        Args:
            w: XGBoost blending weight in [0, 1].
            match_preds: List of dicts with per-match MCMC/XGBoost lambdas
                and actual scores.
            max_goals: Maximum goals per side in the score matrix.

        Returns:
            Mean Brier Score (lower is better).
        """
        total_brier = 0.0
        n = len(match_preds)

        for mp in match_preds:
            lam_h = w * mp["xgb_home"] + (1 - w) * mp["mcmc_home"]
            lam_a = w * mp["xgb_away"] + (1 - w) * mp["mcmc_away"]

            # Ensure positive rates
            lam_h = max(lam_h, 1e-6)
            lam_a = max(lam_a, 1e-6)

            prob_matrix = build_score_matrix(lam_h, lam_a, max_goals=max_goals)

            # One-hot actual outcome
            actual_h = min(mp["actual_home"], max_goals)
            actual_a = min(mp["actual_away"], max_goals)
            actual_matrix = np.zeros_like(prob_matrix)
            actual_matrix[actual_h, actual_a] = 1.0

            # Brier score for this match
            total_brier += float(np.sum((prob_matrix - actual_matrix) ** 2))

        return total_brier / n if n > 0 else 0.0
