"""
Poisson Probability Matrix Builder
===================================

Builds a joint probability matrix for football match scorelines using
independent Poisson distributions for home and away goals. Provides
utilities to extract outcome probabilities, top scorelines, and
individual score lookups.
"""

import numpy as np
from scipy.stats import poisson


def build_score_matrix(
    lambda_home: float, lambda_away: float, max_goals: int = 7
) -> np.ndarray:
    """Compute the joint Poisson probability matrix for a football match.

    Assumes home and away goals are independent Poisson random variables:

        P(home=i, away=j) = Poisson(i; λ_home) × Poisson(j; λ_away)

    Parameters
    ----------
    lambda_home : float
        Expected goals (xG) for the home team.
    lambda_away : float
        Expected goals (xG) for the away team.
    max_goals : int, optional
        Maximum number of goals per team to consider (default 7).
        The resulting matrix has shape (max_goals+1, max_goals+1).

    Returns
    -------
    np.ndarray
        A (max_goals+1, max_goals+1) matrix where entry [i, j] is
        the probability of the scoreline i–j.
    """
    goals = np.arange(0, max_goals + 1)
    home_probs = poisson.pmf(goals, lambda_home)  # shape (max_goals+1,)
    away_probs = poisson.pmf(goals, lambda_away)  # shape (max_goals+1,)
    # Outer product gives joint probability matrix
    matrix = np.outer(home_probs, away_probs)
    return matrix


def get_outcome_probabilities(matrix: np.ndarray) -> dict:
    """Derive match outcome probabilities from a score probability matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Score probability matrix as returned by :func:`build_score_matrix`.
        Entry [i, j] is P(home=i, away=j).

    Returns
    -------
    dict
        Dictionary with keys ``'home_win'``, ``'draw'``, ``'away_win'``,
        each mapping to a float probability.

    Notes
    -----
    - **Home win**: sum of the *lower* triangle (i > j).
    - **Draw**: sum of the diagonal (i == j).
    - **Away win**: sum of the *upper* triangle (i < j).
    """
    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    return {"home_win": home_win, "draw": draw, "away_win": away_win}


def get_top_scores(
    matrix: np.ndarray, n: int = 10
) -> list[tuple[int, int, float]]:
    """Return the *n* most probable exact scorelines.

    Parameters
    ----------
    matrix : np.ndarray
        Score probability matrix.
    n : int, optional
        Number of top scorelines to return (default 10).

    Returns
    -------
    list[tuple[int, int, float]]
        Sorted list of ``(home_goals, away_goals, probability)`` tuples,
        highest probability first.
    """
    # Flatten, argsort descending, then unravel back to 2-D indices
    flat = matrix.ravel()
    top_indices = np.argsort(flat)[::-1][:n]
    rows, cols = np.unravel_index(top_indices, matrix.shape)

    return [
        (int(r), int(c), float(matrix[r, c])) for r, c in zip(rows, cols)
    ]


def get_score_probability(
    matrix: np.ndarray, home_goals: int, away_goals: int
) -> float:
    """Look up the probability of a specific scoreline.

    Parameters
    ----------
    matrix : np.ndarray
        Score probability matrix.
    home_goals : int
        Number of home goals.
    away_goals : int
        Number of away goals.

    Returns
    -------
    float
        Probability of the scoreline ``home_goals``–``away_goals``.

    Raises
    ------
    IndexError
        If the requested scoreline exceeds the matrix dimensions.
    """
    return float(matrix[home_goals, away_goals])
