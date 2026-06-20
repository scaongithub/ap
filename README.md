# ⚽ Football Match Score Predictor

A machine learning pipeline that predicts the precise probability matrix of a football match's final score using historical data, Bayesian statistics, and gradient-boosted decision trees.

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Data Loader │────▶│   Dixon-Coles    │     │   Poisson Matrix │
│  (GitHub CSV)│     │  (Base Ratings)  │     │    (8×8 Grid)    │
└──────┬───────┘     └──────────────────┘     └───────▲──────────┘
       │                                              │
       ├──────────▶ MCMC (PyMC v5) ──────┐            │
       │           Bayesian Posterior     │   Ensemble │
       │                                 ├───Blend────┤
       └──────────▶ XGBoost ─────────────┘   (w opt.) │
                   Poisson Regression                  │
                                              ┌───────▼──────────┐
                                              │   Visualization  │
                                              │ Heatmap│Bars│Rich│
                                              └──────────────────┘
```

### Models

| Model | Type | Signal Captured |
|-------|------|-----------------|
| **Dixon-Coles** | MLE (scipy) | Baseline attack/defense ratings with time decay & low-score correction |
| **MCMC** | Bayesian (PyMC v5) | Full posterior distributions of team strengths; long-term underlying ability |
| **XGBoost** | Gradient Boosting | Non-linear feature interactions, recent form, situational context |
| **Ensemble** | Brier Score-optimized blend | `λ = w·λ_XGB + (1−w)·λ_MCMC`, weight optimized on validation set |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyMC v5 requires Python ≥ 3.10. On Windows, you may need Visual C++ Build Tools.

### 2. Predict a Match

```bash
# Ensemble prediction (default — best accuracy)
python predict.py match --home "Spain" --away "Brazil"

# Use a specific model
python predict.py match --home "Argentina" --away "France" --model mcmc
python predict.py match --home "Germany" --away "England" --model xgboost

# Skip chart generation (terminal output only)
python predict.py match --home "Spain" --away "Brazil" --no-plots
```

### 3. Interactive Mode

```bash
# Launch interactive explorer (or just run with no arguments)
python predict.py interactive
python predict.py
```

### 4. List Teams & Ratings

```bash
# See all available teams
python predict.py teams

# Show top team ratings
python predict.py ratings --model dixon-coles --top 20
python predict.py ratings --model mcmc --top 30
```

---

## 📊 Output

The predictor generates:

1. **Score Probability Heatmap** — 8×8 matrix showing likelihood of every exact scoreline
2. **Outcome Bar Chart** — Aggregated Home Win / Draw / Away Win probabilities
3. **Top 10 Scores** — Ranked list of the most likely exact scorelines
4. **Rich Terminal Summary** — Beautiful formatted output with colored probability bars

All charts are saved to `output/` as high-DPI PNG files.

---

## 📂 Project Structure

```
apuestas/
├── predict.py              # CLI entry point (Typer + Rich)
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Fetch, preprocess, feature engineering
│   ├── dixon_coles.py      # Dixon-Coles MLE model
│   ├── mcmc_model.py       # Bayesian MCMC (PyMC v5)
│   ├── xgboost_model.py    # XGBoost expected goals predictor
│   ├── ensemble.py         # Brier Score-optimized ensemble blend
│   ├── poisson_matrix.py   # Poisson probability matrix builder
│   └── visualize.py        # Heatmaps, bar charts, Rich terminal output
├── data/                   # Cached CSV data & MCMC traces (gitignored)
└── output/                 # Generated charts (gitignored)
```

---

## 🔬 How It Works

### Data Pipeline
- Downloads international football results from [martj42/international_results](https://github.com/martj42/international_results)
- Filters to matches from **2018 onwards** for modern relevance
- Engineers rolling statistics: goals scored/conceded averages, win rates, head-to-head records
- Applies time-decay weighting (exponential, λ=0.003/day) and tournament importance weights

### Poisson Scoring
The expected goals (λ) from each model are fed into the Poisson distribution:

$$P(x) = \frac{\lambda^x e^{-\lambda}}{x!}$$

The **outer product** of home and away Poisson distributions creates the final 2D score probability matrix.

### Ensemble Optimization
The blending weight `w` is treated as a meta-parameter optimized by minimizing the **multiclass Brier Score** on a held-out validation set:

$$\lambda_{final} = w \cdot \lambda_{XGB} + (1 - w) \cdot \lambda_{MCMC}$$

---

## 📋 Data Source

**Dataset:** [International Football Results from 1872 to Present](https://github.com/martj42/international_results)  
**License:** Open source, publicly maintained  
**Coverage:** 45,000+ international matches  

---

## ⚙️ Tech Stack

| Library | Role |
|---------|------|
| `pandas` / `numpy` | Data manipulation & matrix math |
| `pymc` v5 | Bayesian MCMC sampling (NUTS) |
| `xgboost` | Gradient-boosted decision trees |
| `scipy` | Poisson distributions & optimization |
| `matplotlib` / `seaborn` | Premium dark-themed charts |
| `typer` / `rich` | CLI framework & terminal rendering |
| `arviz` | MCMC diagnostics & trace management |

---

## 📄 License

MIT
