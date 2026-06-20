"""
Football Match Score Predictor — CLI Entry Point.

Uses Typer for a clean CLI-first experience with an optional interactive mode.
Supports three prediction models: mcmc, xgboost, or ensemble (default).

Usage:
    python predict.py match --home "Spain" --away "Brazil"
    python predict.py match --home "Spain" --away "Brazil" --model mcmc
    python predict.py teams                    # list available teams
    python predict.py --interactive            # interactive exploration mode
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import box

# ── App Setup ──────────────────────────────────────────────────────────
app = typer.Typer(
    name="predict",
    help="⚽ Football Match Score Predictor — Bayesian MCMC + XGBoost ensemble.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# ── Shared State ───────────────────────────────────────────────────────
# Lazy-loaded so we don't pay import cost on --help
_state: dict = {}


def _load_pipeline(model_choice: str = "ensemble") -> dict:
    """Load data and fit models. Caches across calls within the same run."""
    if _state.get("loaded"):
        return _state

    from src.data_loader import fetch_data, preprocess, engineer_features, get_team_list
    from src.dixon_coles import DixonColesModel
    from src.mcmc_model import MCMCModel
    from src.xgboost_model import XGBoostGoalModel
    from src.ensemble import EnsemblePredictor
    from src.poisson_matrix import build_score_matrix, get_outcome_probabilities, get_top_scores
    from src.visualize import plot_all, render_rich_summary

    console.print(Panel("⚽ [bold cyan]Football Match Score Predictor[/bold cyan]", 
                        subtitle="Loading pipeline…", style="cyan"))

    # ── Data ───────────────────────────────────────────────────────
    with console.status("[bold green]Fetching and preprocessing data…"):
        raw_df = fetch_data()
        clean_df = preprocess(raw_df)
        feat_df = engineer_features(clean_df)
    console.print(f"  ✅ Data loaded: [cyan]{len(clean_df):,}[/cyan] matches, "
                  f"[cyan]{len(get_team_list(clean_df))}[/cyan] teams\n")

    _state["clean_df"] = clean_df
    _state["feat_df"] = feat_df
    _state["team_list"] = get_team_list(clean_df)

    # ── Dixon-Coles (always fitted — lightweight) ──────────────────
    with console.status("[bold green]Fitting Dixon-Coles model…"):
        dc_model = DixonColesModel()
        dc_model.fit(clean_df)
    console.print("  ✅ Dixon-Coles model fitted\n")
    _state["dc_model"] = dc_model

    # ── MCMC ───────────────────────────────────────────────────────
    if model_choice in ("mcmc", "ensemble"):
        console.print("  🔄 Fitting Bayesian MCMC model (this may take a few minutes)…\n")
        mcmc_model = MCMCModel()
        mcmc_model.fit(clean_df, draws=2000, tune=1000, chains=2)
        console.print("  ✅ MCMC model fitted\n")
        _state["mcmc_model"] = mcmc_model
    else:
        _state["mcmc_model"] = None

    # ── XGBoost ────────────────────────────────────────────────────
    if model_choice in ("xgboost", "ensemble"):
        with console.status("[bold green]Training XGBoost models…"):
            xgb_model = XGBoostGoalModel()
            xgb_model.fit(feat_df)
        console.print("  ✅ XGBoost model fitted\n")
        _state["xgb_model"] = xgb_model
    else:
        _state["xgb_model"] = None

    # ── Ensemble weight optimization ───────────────────────────────
    if model_choice == "ensemble" and _state["mcmc_model"] and _state["xgb_model"]:
        with console.status("[bold green]Optimizing ensemble weight…"):
            ensemble = EnsemblePredictor(
                mcmc_model=_state["mcmc_model"],
                xgboost_model=_state["xgb_model"],
                dixon_coles_model=_state["dc_model"],
            )
            ensemble.optimize_weight(feat_df)
        console.print(f"  ✅ Ensemble weight optimized: "
                      f"w(XGBoost) = [cyan]{ensemble.optimal_w:.3f}[/cyan], "
                      f"w(MCMC) = [cyan]{1 - ensemble.optimal_w:.3f}[/cyan]\n")
        _state["ensemble"] = ensemble

    _state["loaded"] = True
    _state["model_choice"] = model_choice
    return _state


def _predict_match(home_team: str, away_team: str, model: str = "ensemble",
                   show_plots: bool = True) -> None:
    """Run prediction for a single match and display results."""
    from src.poisson_matrix import build_score_matrix, get_outcome_probabilities, get_top_scores
    from src.visualize import plot_all, render_rich_summary

    state = _load_pipeline(model)

    # ── Resolve team names (case-insensitive fuzzy match) ──────────
    home_team = _resolve_team(home_team, state["team_list"])
    away_team = _resolve_team(away_team, state["team_list"])

    console.print(Panel(
        f"[bold white]{home_team}[/bold white] 🏠  vs  ✈️  [bold white]{away_team}[/bold white]",
        title="[bold cyan]Match Prediction[/bold cyan]",
        style="cyan",
        padding=(1, 4),
    ))

    # ── Get lambda values based on model choice ────────────────────
    model_info = {}

    if model == "ensemble" and state.get("ensemble"):
        comparison = state["ensemble"].get_model_comparison(
            home_team, away_team, df=state["feat_df"]
        )
        lambda_home = comparison["ensemble"]["lambda_home"]
        lambda_away = comparison["ensemble"]["lambda_away"]
        model_info = {
            "MCMC": (comparison["mcmc"]["lambda_home"], comparison["mcmc"]["lambda_away"]),
            "XGBoost": (comparison["xgboost"]["lambda_home"], comparison["xgboost"]["lambda_away"]),
            "Ensemble": (lambda_home, lambda_away),
            "optimal_w": comparison["optimal_w"],
        }
        if comparison.get("dixon_coles"):
            model_info["Dixon-Coles"] = (
                comparison["dixon_coles"]["lambda_home"],
                comparison["dixon_coles"]["lambda_away"],
            )

    elif model == "mcmc" and state.get("mcmc_model"):
        lambda_home, lambda_away = state["mcmc_model"].predict(home_team, away_team)
        model_info = {"MCMC": (lambda_home, lambda_away)}

    elif model == "xgboost" and state.get("xgb_model"):
        lambda_home, lambda_away = state["xgb_model"].predict_from_teams(
            state["feat_df"], home_team, away_team
        )
        model_info = {"XGBoost": (lambda_home, lambda_away)}

    else:
        # Fallback to Dixon-Coles
        lambda_home, lambda_away = state["dc_model"].predict(home_team, away_team)
        model_info = {"Dixon-Coles": (lambda_home, lambda_away)}

    # ── Build Poisson matrix and extract results ───────────────────
    matrix = build_score_matrix(lambda_home, lambda_away)
    outcomes = get_outcome_probabilities(matrix)
    top_scores = get_top_scores(matrix, n=10)

    # ── Rich terminal output ───────────────────────────────────────
    render_rich_summary(matrix, outcomes, top_scores, home_team, away_team,
                        model_info=model_info)

    # ── Matplotlib plots ───────────────────────────────────────────
    if show_plots:
        plot_all(matrix, home_team, away_team, output_dir="output")
        console.print(f"\n  📁 Charts saved to [cyan]output/[/cyan]\n")


def _resolve_team(name: str, team_list: list[str]) -> str:
    """Resolve a team name with case-insensitive matching and fuzzy fallback."""
    # Exact match
    if name in team_list:
        return name

    # Case-insensitive
    lower_map = {t.lower(): t for t in team_list}
    if name.lower() in lower_map:
        return lower_map[name.lower()]

    # Partial match
    matches = [t for t in team_list if name.lower() in t.lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        console.print(f"[yellow]⚠️  Ambiguous team name '{name}'. Did you mean one of:[/yellow]")
        for m in matches[:10]:
            console.print(f"    • {m}")
        raise typer.Exit(1)

    # No match
    console.print(f"[red]❌ Team '{name}' not found.[/red]")
    console.print("[dim]Use 'python predict.py teams' to see all available teams.[/dim]")
    raise typer.Exit(1)


# ═══════════════════════════════════════════════════════════════════════
#  CLI Commands
# ═══════════════════════════════════════════════════════════════════════


@app.command()
def match(
    home: str = typer.Option(..., "--home", "-h", help="Home team name"),
    away: str = typer.Option(..., "--away", "-a", help="Away team name"),
    model: str = typer.Option(
        "ensemble",
        "--model", "-m",
        help="Prediction model: mcmc, xgboost, or ensemble",
    ),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip matplotlib chart generation"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Match date (reserved for future use)"),
):
    """Predict the score probabilities for a specific match."""
    model = model.lower()
    if model not in ("mcmc", "xgboost", "ensemble"):
        console.print(f"[red]❌ Invalid model '{model}'. Choose from: mcmc, xgboost, ensemble[/red]")
        raise typer.Exit(1)

    _predict_match(home, away, model=model, show_plots=not no_plots)


@app.command()
def teams():
    """List all available national teams in the dataset."""
    from src.data_loader import fetch_data, preprocess, get_team_list

    with console.status("[bold green]Loading data…"):
        raw_df = fetch_data()
        clean_df = preprocess(raw_df)
        team_list = get_team_list(clean_df)

    console.print(Panel(
        f"[bold cyan]{len(team_list)} teams available[/bold cyan]",
        style="cyan",
    ))

    # Display in 4-column layout
    cols = 4
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    for _ in range(cols):
        table.add_column()

    for i in range(0, len(team_list), cols):
        row = team_list[i:i + cols]
        while len(row) < cols:
            row.append("")
        table.add_row(*row)

    console.print(table)


@app.command()
def ratings(
    model: str = typer.Option("dixon-coles", "--model", "-m",
                              help="Model for ratings: dixon-coles or mcmc"),
    top: int = typer.Option(30, "--top", "-n", help="Number of top teams to show"),
):
    """Show team strength ratings from the fitted model."""
    state = _load_pipeline(model if model != "dixon-coles" else "mcmc")

    if model == "dixon-coles":
        ratings_df = state["dc_model"].get_team_ratings()
    elif model == "mcmc" and state.get("mcmc_model"):
        ratings_df = state["mcmc_model"].get_team_ratings()
    else:
        console.print("[red]❌ Model not available.[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]Top {top} Team Ratings — {model.upper()}[/bold cyan]",
        style="cyan",
    ))

    table = Table(box=box.ROUNDED, show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("Team", style="bold white")
    table.add_column("Attack", style="green", justify="right")
    table.add_column("Defense", style="red", justify="right")
    table.add_column("Overall", style="cyan", justify="right")

    for i, row in ratings_df.head(top).iterrows():
        rank = str(ratings_df.index.get_loc(i) + 1)
        table.add_row(
            rank,
            str(row.get("team", "")),
            f"{row.get('attack', 0):.3f}",
            f"{row.get('defense', 0):.3f}",
            f"{row.get('overall', 0):.3f}",
        )

    console.print(table)


@app.command()
def interactive():
    """Launch an interactive exploration loop."""
    console.print(Panel(
        "[bold cyan]⚽ Interactive Match Predictor[/bold cyan]\n\n"
        "[dim]Type team names to predict matches. Type 'quit' to exit.[/dim]",
        style="cyan",
        padding=(1, 4),
    ))

    # Ask model choice once
    model_choice = Prompt.ask(
        "Select model",
        choices=["ensemble", "mcmc", "xgboost"],
        default="ensemble",
    )

    # Load pipeline
    _load_pipeline(model_choice)
    team_list = _state["team_list"]

    while True:
        console.print("\n" + "─" * 60)
        home = Prompt.ask("[bold green]🏠 Home team[/bold green]").strip()
        if home.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye! 👋[/dim]")
            break

        away = Prompt.ask("[bold red]✈️  Away team[/bold red]").strip()
        if away.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye! 👋[/dim]")
            break

        show_plots_answer = Prompt.ask(
            "Show matplotlib charts?",
            choices=["y", "n"],
            default="y",
        )

        try:
            _predict_match(home, away, model=model_choice,
                          show_plots=(show_plots_answer == "y"))
        except SystemExit:
            continue  # typer.Exit was raised due to invalid team — keep looping
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            continue


# ═══════════════════════════════════════════════════════════════════════
#  Entrypoint — auto-interactive if no arguments
# ═══════════════════════════════════════════════════════════════════════

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """⚽ Football Match Score Predictor — CLI entry point.
    
    If no command is specified, launches interactive mode automatically.
    """
    if ctx.invoked_subcommand is None:
        interactive()


if __name__ == "__main__":
    app()
