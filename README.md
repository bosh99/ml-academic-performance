# ml-academic-performance

Machine Learning course project (partial submission) that predicts students' academic performance using the [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) dataset (Cortez & Silva, 2008).

## Research Question

> Which socioeconomic, family, and study habit factors have the greatest influence on final academic performance (`G3`, scale 0–20)?

This is framed as a **regression** problem on `G3`, and explicitly evaluates the effect of *data leakage* from the partial grades `G1` and `G2` by training models under two scenarios:

- **Scenario A**: includes `G1` and `G2` (comparison baseline)
- **Scenario B**: excludes them (answers the actual research question)

## Models

- **Linear Regression** (baseline)
- **Random Forest** (with `GridSearchCV`)
- **K-Nearest Neighbors** (with `GridSearchCV`)

All with `random_state=42`, 5-fold cross-validation and a 70 / 15 / 15 split (train / val / test).

## Explainability

Following the rubric requirements, `explainability.ipynb` includes:

- Linear Regression coefficients (bar chart)
- Random Forest feature importances (top 15)
- SHAP values with `TreeExplainer` (beeswarm + bar plot)

Plots are saved to [`figures/`](figures/).

## Repository Structure

```
.
├── data/                       # Raw and processed datasets
├── figures/                    # Explainability plots
├── models/                     # Models serialized with joblib
├── data_processing.ipynb       # Full EDA
├── preprocessing.ipynb         # Encoding + split + save to CSV
├── modeling.ipynb              # Training, GridSearch and evaluation
├── explainability.ipynb        # Coefficients, feature importances and SHAP
├── pyproject.toml
├── .python-version
└── README.md
```

## Installation

Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

The required Python version is specified in [`.python-version`](.python-version) (**Python 3.13**).

Dependencies declared in [`pyproject.toml`](pyproject.toml): `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `joblib`, `ipykernel`.

## Execution Order

Notebooks must be run in this exact order:

1. [`data_processing.ipynb`](data_processing.ipynb) — Exploratory Data Analysis (distributions, correlations, target balance, null checks).
2. [`preprocessing.ipynb`](preprocessing.ipynb) — Binary encoding, one-hot encoding, target definition, 80/20 backup split, and saving `math_processed.csv` / `portuguese_processed.csv`.
3. [`modeling.ipynb`](modeling.ipynb) — Training and evaluation of all three models under scenarios A and B; saves models to `models/` and a comparison table to `models/results_comparison.csv`.
4. [`explainability.ipynb`](explainability.ipynb) — Loads the models, generates explainability plots, and explicitly answers the research question.

## Reproducibility

- Global seed `random_state=42` across all splits, CV, and stochastic models.
- Train/val/test partitions are deterministic given that seed and the CSVs produced by `preprocessing.ipynb`.
- Trained models are serialized in `models/` so explainability analyses can be reproduced without retraining.

## Dataset

- Original source: Cortez, P. & Silva, A. (2008). *Using data mining to predict secondary school student performance*.
- 395 Math students (primary modeling focus) and 649 Portuguese students, with 33 columns describing student attributes, family background, study habits, social life, and the three grades (`G1`, `G2`, `G3`).
