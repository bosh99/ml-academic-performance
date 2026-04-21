# ml-academic-performance

Proyecto de la materia de Machine Learning (entrega parcial) que predice el rendimiento académico de estudiantes a partir del dataset [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008).

## Pregunta de investigación

> ¿Qué factores socioeconómicos, familiares y de hábitos de estudio tienen mayor influencia sobre el rendimiento académico final (`G3`, escala 0–20)?

Se aborda como un problema de **regresión** sobre `G3`, y se evalúa explícitamente el efecto del *data leakage* de las notas parciales `G1` y `G2` entrenando los modelos en dos escenarios:

- **Escenario A**: incluye `G1` y `G2` (baseline de comparación)
- **Escenario B**: las excluye (responde a la pregunta real)

## Modelos

- **Regresión Lineal** (baseline)
- **Random Forest** (con `GridSearchCV`)
- **K-Nearest Neighbors** (con `GridSearchCV`)

Todos con `random_state=42`, 5-fold cross-validation y split 70 / 15 / 15 (train / val / test).

## Explicabilidad

Siguiendo el requisito del rubric, `explainability.ipynb` incluye:

- Coeficientes de la Regresión Lineal (bar chart)
- Feature importances del Random Forest (top 15)
- SHAP values con `TreeExplainer` (beeswarm + bar plot)

Los gráficos se guardan en [`figures/`](figures/).

## Estructura del repositorio

```
.
├── data/                       # Datasets crudos y procesados
├── figures/                    # Gráficos de explicabilidad
├── models/                     # Modelos serializados con joblib
├── data_processing.ipynb       # EDA completa
├── preprocessing.ipynb         # Encoding + split + guardado a CSV
├── modeling.ipynb              # Entrenamiento, GridSearch y evaluación
├── explainability.ipynb        # Coeficientes, feature importances y SHAP
├── pyproject.toml
├── .python-version
└── README.md
```

## Instalación

Se usa [uv](https://github.com/astral-sh/uv) para gestionar dependencias.

```bash
uv sync
```

La versión de Python requerida está en [`.python-version`](.python-version) (**Python 3.13**).

Las dependencias declaradas en [`pyproject.toml`](pyproject.toml) son: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `joblib`, `ipykernel`.

## Orden de ejecución

Los notebooks deben ejecutarse en este orden exacto:

1. [`data_processing.ipynb`](data_processing.ipynb) — Exploratory Data Analysis (distribuciones, correlaciones, balance del target, chequeo de nulos).
2. [`preprocessing.ipynb`](preprocessing.ipynb) — Encoding binario, one-hot encoding, definición de targets, split 80/20 de respaldo y guardado de `math_processed.csv` / `portuguese_processed.csv`.
3. [`modeling.ipynb`](modeling.ipynb) — Entrenamiento y evaluación de los tres modelos en los escenarios A y B; guarda modelos en `models/` y una tabla comparativa en `models/results_comparison.csv`.
4. [`explainability.ipynb`](explainability.ipynb) — Carga los modelos, genera los gráficos de explicabilidad y responde explícitamente a la pregunta de investigación.

## Reproducibilidad

- Semilla global `random_state=42` en todos los splits, CV y modelos aleatorios.
- Las particiones train/val/test son determinísticas dadas esa semilla y los CSV generados por `preprocessing.ipynb`.
- Los modelos entrenados quedan serializados en `models/` para poder reproducir los análisis de explicabilidad sin reentrenar.

## Dataset

- Fuente original: Cortez, P. & Silva, A. (2008). *Using data mining to predict secondary school student performance*.
- 395 estudiantes de Matemáticas (foco principal del modelado) y 649 de Portugués, con 33 columnas que describen atributos del estudiante, de su familia, hábitos de estudio, vida social y las tres notas (`G1`, `G2`, `G3`).
