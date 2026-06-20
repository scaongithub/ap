"""
XGBoost Goal Prediction Model.

Trains two separate XGBRegressor models (home and away) using Poisson regression
to predict expected goals for football matches. Includes hyperparameter tuning
via TimeSeriesSplit cross-validation and feature importance analysis.
"""

import itertools
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


def _cuda_available() -> bool:
    """Return True if an NVIDIA GPU is reachable via nvidia-smi.

    XGBoost's CUDA backend requires the CUDA toolkit and an NVIDIA GPU.
    We probe nvidia-smi rather than importing a heavy CUDA library so that
    the check is instant and has zero install cost.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:  # binary not found or timed out
        return False


# Feature columns used by the model.
FEATURE_COLS = [
    "home_goals_scored_5",
    "home_goals_conceded_5",
    "home_win_rate_5",
    "home_goals_scored_10",
    "home_goals_conceded_10",
    "home_win_rate_10",
    "home_goals_scored_20",
    "home_goals_conceded_20",
    "home_win_rate_20",
    "home_days_since_last",
    "home_h2h_wins_5",
    "away_goals_scored_5",
    "away_goals_conceded_5",
    "away_win_rate_5",
    "away_goals_scored_10",
    "away_goals_conceded_10",
    "away_win_rate_10",
    "away_goals_scored_20",
    "away_goals_conceded_20",
    "away_win_rate_20",
    "away_days_since_last",
    "away_h2h_wins_5",
    "tournament_weight",
    "neutral",
]


class XGBoostGoalModel:
    """XGBoost-based expected goals predictor.

    Trains two separate Poisson regression models to predict the expected number
    of goals scored by the home and away teams respectively, using rolling
    historical features.

    Attributes:
        home_model: Fitted XGBRegressor for home goals.
        away_model: Fitted XGBRegressor for away goals.
        best_params_home: Best hyperparameters for the home model.
        best_params_away: Best hyperparameters for the away model.
        feature_importances_home: Feature importances from the home model.
        feature_importances_away: Feature importances from the away model.
        use_gpu: Whether CUDA acceleration is active.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        """Initialise the model.

        Args:
            use_gpu: Defaults to False (CPU). Pass True to run tree building
                on an NVIDIA GPU via CUDA. If True but no usable GPU is found,
                the model warns and transparently falls back to CPU rather
                than crashing mid-pipeline.
        """
        self.home_model: XGBRegressor | None = None
        self.away_model: XGBRegressor | None = None
        self.best_params_home: dict | None = None
        self.best_params_away: dict | None = None
        self.feature_importances_home: np.ndarray | None = None
        self.feature_importances_away: np.ndarray | None = None
        self._is_fitted = False

        # CPU by default; only attempt CUDA when explicitly requested. When a
        # GPU is requested we verify it exists so a missing GPU degrades to a
        # warning + CPU fallback instead of an opaque XGBoost runtime error.
        if use_gpu and not _cuda_available():
            print(
                "[XGBoost] ⚠ GPU requested but no CUDA device detected "
                "(nvidia-smi unavailable). Falling back to CPU."
            )
            self.use_gpu = False
        else:
            self.use_gpu = use_gpu

        device_label = "cuda (GPU)" if self.use_gpu else "cpu"
        print(f"[XGBoost] Device: {device_label}")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, val_year: int = 2024) -> "XGBoostGoalModel":
        """Fit home and away XGBRegressor models with hyperparameter tuning.

        Args:
            df: Feature-engineered DataFrame (output of
                ``data_loader.engineer_features``).  Must contain all columns
                listed in ``FEATURE_COLS`` plus ``home_score``, ``away_score``,
                and ``date``.
            val_year: Year used for the time-based train/validation split.
                Data before this year is used for training; data from this year
                onward is used for validation.

        Returns:
            self, to allow method chaining.

        Raises:
            ValueError: If required columns are missing from *df*.
        """
        self._validate_columns(df)

        # Ensure date column is datetime
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Drop rows with NaN in features or targets
        required = FEATURE_COLS + ["home_score", "away_score"]
        df = df.dropna(subset=required)

        # Time-based split
        train = df[df["date"].dt.year < val_year]
        val = df[df["date"].dt.year >= val_year]

        print(f"[XGBoost] Training samples : {len(train):,}")
        print(f"[XGBoost] Validation samples: {len(val):,}")

        X_train = train[FEATURE_COLS].values
        X_val = val[FEATURE_COLS].values

        # --- Home model ---
        print("\n[XGBoost] Tuning HOME goals model …")
        self.home_model, self.best_params_home = self._tune_and_train(
            X_train,
            train["home_score"].values,
            X_val,
            val["home_score"].values,
            label="home",
            use_gpu=self.use_gpu,
        )
        self.feature_importances_home = self.home_model.feature_importances_

        # --- Away model ---
        print("\n[XGBoost] Tuning AWAY goals model …")
        self.away_model, self.best_params_away = self._tune_and_train(
            X_train,
            train["away_score"].values,
            X_val,
            val["away_score"].values,
            label="away",
            use_gpu=self.use_gpu,
        )
        self.feature_importances_away = self.away_model.feature_importances_

        self._is_fitted = True
        print("\n[XGBoost] ✓ Both models fitted successfully.")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: dict) -> tuple[float, float]:
        """Predict expected goals for a single match from raw feature values.

        Args:
            features: Dictionary mapping feature names (see ``FEATURE_COLS``)
                to their numeric values.

        Returns:
            ``(lambda_home, lambda_away)`` — the predicted expected goals for
            each side.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            KeyError: If a required feature is missing from *features*.
        """
        self._check_fitted()

        # Build feature vector in the correct column order
        try:
            x = np.array([[features[col] for col in FEATURE_COLS]])
        except KeyError as exc:
            missing = [c for c in FEATURE_COLS if c not in features]
            raise KeyError(
                f"Missing features for prediction: {missing}"
            ) from exc

        lambda_home = float(self.home_model.predict(x)[0])
        lambda_away = float(self.away_model.predict(x)[0])

        # Clamp to non-negative (Poisson rates must be ≥ 0)
        lambda_home = max(lambda_home, 0.0)
        lambda_away = max(lambda_away, 0.0)

        return lambda_home, lambda_away

    def predict_from_teams(
        self,
        df: pd.DataFrame,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        tournament_weight: float = 1.0,
    ) -> tuple[float, float]:
        """Predict expected goals by extracting the latest features for two teams.

        Convenience wrapper around :meth:`predict` that looks up the most
        recent rolling statistics for *home_team* and *away_team* in *df*.

        Args:
            df: Feature-engineered DataFrame containing historical matches.
            home_team: Name of the home team (must match values in the
                ``home_team`` column of *df*).
            away_team: Name of the away team.
            neutral: Whether the fixture is at a neutral venue. Defaults to
                False; the World Cup CLI passes True so the ``neutral`` model
                feature reflects the tournament context.
            tournament_weight: Importance weight of the fixture's competition
                (1.0 = World Cup). Drives the ``tournament_weight`` feature.

        Returns:
            ``(lambda_home, lambda_away)`` predicted expected goals.

        Raises:
            ValueError: If either team is not found in *df*.
        """
        self._check_fitted()

        features = self._extract_latest_features(
            df,
            home_team,
            away_team,
            neutral=neutral,
            tournament_weight=tournament_weight,
        )
        return self.predict(features)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances for both the home and away models.

        Returns:
            DataFrame with columns ``feature``, ``home_importance``, and
            ``away_importance``, sorted by average importance descending.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._check_fitted()

        importance_df = pd.DataFrame(
            {
                "feature": FEATURE_COLS,
                "home_importance": self.feature_importances_home,
                "away_importance": self.feature_importances_away,
            }
        )
        importance_df["avg_importance"] = (
            importance_df["home_importance"] + importance_df["away_importance"]
        ) / 2.0
        importance_df = importance_df.sort_values("avg_importance", ascending=False)
        return importance_df.reset_index(drop=True)

    # ==================================================================
    # Private helpers
    # ==================================================================

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """Raise ``ValueError`` if required columns are absent."""
        required = set(FEATURE_COLS) | {"home_score", "away_score", "date"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}"
            )

    def _check_fitted(self) -> None:
        """Raise ``RuntimeError`` if the model hasn't been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet.  Call fit() first."
            )

    # ------------------------------------------------------------------

    @staticmethod
    def _tune_and_train(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        label: str,
        use_gpu: bool = False,
    ) -> tuple[XGBRegressor, dict]:
        """Run grid search with TimeSeriesSplit CV, then retrain on full training set.

        Args:
            X_train: Training feature matrix.
            y_train: Training target array.
            X_val: Validation feature matrix.
            y_val: Validation target array.
            label: Human-readable label for progress messages (``"home"``
                or ``"away"``).
            use_gpu: If True, use CUDA for XGBoost tree building.

        Returns:
            ``(best_model, best_params)`` — the fitted model and its
            hyperparameters.
        """
        device = "cuda" if use_gpu else "cpu"

        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 500],
        }

        keys = list(param_grid.keys())
        combos = list(itertools.product(*param_grid.values()))
        total = len(combos)

        tscv = TimeSeriesSplit(n_splits=3)

        best_score = np.inf
        best_params: dict = {}

        print(f"  Searching {total} hyperparameter combinations (3-fold TimeSeriesSplit) …")

        for idx, values in enumerate(combos, 1):
            params = dict(zip(keys, values))
            fold_scores: list[float] = []

            for train_idx, test_idx in tscv.split(X_train):
                X_tr, X_te = X_train[train_idx], X_train[test_idx]
                y_tr, y_te = y_train[train_idx], y_train[test_idx]

                model = XGBRegressor(
                    objective="count:poisson",
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    n_estimators=params["n_estimators"],
                    device=device,  # 'cuda' for GPU, 'cpu' otherwise
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X_tr, y_tr, verbose=False)
                preds = model.predict(X_te)
                fold_scores.append(mean_absolute_error(y_te, preds))

            mean_mae = float(np.mean(fold_scores))
            if mean_mae < best_score:
                best_score = mean_mae
                best_params = params

            if idx % 9 == 0 or idx == total:
                print(f"    [{idx}/{total}] best CV MAE so far = {best_score:.4f}")

        print(f"  Best params ({label}): {best_params}")
        print(f"  Best CV MAE ({label}): {best_score:.4f}")

        # Retrain on the full training set with the best parameters
        best_model = XGBRegressor(
            objective="count:poisson",
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            n_estimators=best_params["n_estimators"],
            device=device,  # 'cuda' for GPU, 'cpu' otherwise
            random_state=42,
            verbosity=0,
        )
        best_model.fit(X_train, y_train, verbose=False)

        # Evaluate on the held-out validation set
        val_preds = best_model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        print(f"  Validation MAE ({label}): {val_mae:.4f}")

        return best_model, best_params

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_latest_features(
        df: pd.DataFrame,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        tournament_weight: float = 1.0,
    ) -> dict:
        """Build a feature dict from the most recent stats for each team.

        The method finds the last match where each team appeared (as home or
        away) and pulls the corresponding rolling statistics, re-mapping them
        to the canonical ``home_*`` / ``away_*`` feature names.

        Args:
            df: Feature-engineered DataFrame.
            home_team: Home team name.
            away_team: Away team name.
            neutral: Whether the fixture is at a neutral venue (World Cup).
            tournament_weight: Competition importance weight for the fixture.

        Returns:
            Dictionary of feature values keyed by ``FEATURE_COLS``.

        Raises:
            ValueError: If a team cannot be found in *df*.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        features: dict = {}

        # --- Home team features ---
        home_rows = df[df["home_team"] == home_team]
        if home_rows.empty:
            # Fallback: team might appear only as away
            home_rows = df[df["away_team"] == home_team]
            if home_rows.empty:
                raise ValueError(
                    f"Home team '{home_team}' not found in the dataset."
                )
            last = home_rows.iloc[-1]
            # Map away_* stats → home_* features (the team was playing away)
            for window in [5, 10, 20]:
                features[f"home_goals_scored_{window}"] = last.get(
                    f"away_goals_scored_{window}", 0.0
                )
                features[f"home_goals_conceded_{window}"] = last.get(
                    f"away_goals_conceded_{window}", 0.0
                )
                features[f"home_win_rate_{window}"] = last.get(
                    f"away_win_rate_{window}", 0.0
                )
            features["home_days_since_last"] = last.get("away_days_since_last", 0.0)
            features["home_h2h_wins_5"] = last.get("away_h2h_wins_5", 0.0)
        else:
            last = home_rows.iloc[-1]
            for window in [5, 10, 20]:
                features[f"home_goals_scored_{window}"] = last.get(
                    f"home_goals_scored_{window}", 0.0
                )
                features[f"home_goals_conceded_{window}"] = last.get(
                    f"home_goals_conceded_{window}", 0.0
                )
                features[f"home_win_rate_{window}"] = last.get(
                    f"home_win_rate_{window}", 0.0
                )
            features["home_days_since_last"] = last.get("home_days_since_last", 0.0)
            features["home_h2h_wins_5"] = last.get("home_h2h_wins_5", 0.0)

        # --- Away team features ---
        away_rows = df[df["away_team"] == away_team]
        if away_rows.empty:
            away_rows = df[df["home_team"] == away_team]
            if away_rows.empty:
                raise ValueError(
                    f"Away team '{away_team}' not found in the dataset."
                )
            last = away_rows.iloc[-1]
            for window in [5, 10, 20]:
                features[f"away_goals_scored_{window}"] = last.get(
                    f"home_goals_scored_{window}", 0.0
                )
                features[f"away_goals_conceded_{window}"] = last.get(
                    f"home_goals_conceded_{window}", 0.0
                )
                features[f"away_win_rate_{window}"] = last.get(
                    f"home_win_rate_{window}", 0.0
                )
            features["away_days_since_last"] = last.get("home_days_since_last", 0.0)
            features["away_h2h_wins_5"] = last.get("home_h2h_wins_5", 0.0)
        else:
            last = away_rows.iloc[-1]
            for window in [5, 10, 20]:
                features[f"away_goals_scored_{window}"] = last.get(
                    f"away_goals_scored_{window}", 0.0
                )
                features[f"away_goals_conceded_{window}"] = last.get(
                    f"away_goals_conceded_{window}", 0.0
                )
                features[f"away_win_rate_{window}"] = last.get(
                    f"away_win_rate_{window}", 0.0
                )
            features["away_days_since_last"] = last.get("away_days_since_last", 0.0)
            features["away_h2h_wins_5"] = last.get("away_h2h_wins_5", 0.0)

        # --- Match-level features ---
        # Default to World Cup context (neutral venue, max tournament weight)
        # when the caller does not override, since this tool targets the WC.
        features["tournament_weight"] = tournament_weight
        features["neutral"] = float(neutral)

        return features
