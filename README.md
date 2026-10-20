# Telco churn classification

Predict which telecom customers will leave, using the Kaggle
"Telco Customer Churn" dataset (7,043 customers, 26.5% churn). The goal is to learn
classification on real, imbalanced, slightly messy data with production-style code:
a small library, thin scripts, tests and CI.

## Setup (Windows PowerShell)

```powershell
py -3.14 -m venv .venv          # Python 3.12+ works too
.\.venv\Scripts\Activate.ps1
pip install -e . -r requirements-dev.txt
pytest -q
```

Put the dataset at `data/telco_churn.csv` (original BlastChar upload, 7,043 rows, 21 columns).

## Project layout

```
src/churn/        the library (reusable code)
  config.py       paths, constants, placeholder business costs
  data.py         load_raw, prepare_features (pipeline step 1), clean
  eda.py          first-look report, churn rate by tenure
  features.py     train/test split, preprocessing (scaling, one-hot, polynomial, splines)
  models.py       model builders and the candidates() registry
  evaluate.py     precision, recall, F1, ROC-AUC, confusion matrix
  compare.py      fair cross-validated model comparison
  linearity.py    data-driven test: which numeric columns are curved?
  plots.py        churn rate vs numeric columns chart
  tuning.py       RandomizedSearchCV over polynomial degree and C
  leakage.py      data leakage checks
  thresholds.py   choosing the decision cutoff
  registry.py     versioned model storage, promotion rule, rollback
  train.py        train the final model and save it as a new registry version
  predict.py      ChurnPredictor: load once, score raw records
  schemas.py      request/response models for the API (validation)
  api.py          FastAPI app: /health, /model-info, /predict, /reload-model
scripts/          entry points, run with `python scripts\<name>.py`
tests/            pytest suite
Dockerfile, docker-compose.yml, .dockerignore    container setup
```

## Scripts

| script | what it shows |
|---|---|
| `first_look.py` | shape, class balance, blank `TotalCharges` rows |
| `baseline.py` | dummy model vs logistic regression (accuracy misleads) |
| `evaluate_baseline.py` | precision, recall, F1, ROC-AUC, confusion matrix |
| `compare_models.py` | logistic regression, random forest, gradient boosting with 5-fold CV |
| `polynomial_features.py` | tenure churn table, polynomial degrees 1 to 3 |
| `analyze_linearity.py` | which numeric columns are curved + `reports/churn_vs_numeric.png` |
| `compare_curve_treatments.py` | plain vs polynomial vs splines on curved columns only |
| `tune_polynomial.py` | tunes polynomial degree and regularisation `C` together |
| `check_leakage.py` | four leakage checks |
| `choose_threshold.py` | precision/recall trade-off at each cutoff |
| `train.py` | trains the final model, saves a new version, promotes it unless it is clearly worse |
| `registry.py` | `list` saved versions, or `rollback <version>` to make one production |
| `predict_demo.py` | scores raw customer rows with the production model |

## Train and serve

```powershell
python scripts\train.py                          # saves a version under models/versions/
uvicorn churn.api:app --reload                   # serves the production model on :8000
```

Open http://127.0.0.1:8000/docs and try `POST /predict` (the form is pre-filled with an example
customer). Endpoints:

| endpoint | purpose |
|---|---|
| `GET /health` | liveness check |
| `GET /model-info` | metadata of the loaded model: version, cutoff, test metrics, library versions |
| `POST /predict` | one customer in, churn probability + flag + model version out; invalid input gets HTTP 422 |
| `POST /reload-model` | switch to the current production version after a promotion or rollback, no restart; if loading fails the old model keeps serving |

## Model registry

Each training run is stored as its own version:

```
models/
  versions/<id>/model.joblib + metadata.json
  production.json              which version the API serves
```

A new version becomes production only if its test ROC-AUC did not drop by more than
`MAX_ACCEPTABLE_AUC_DROP` (0.01, in `config.py`) compared with the current production model.
Use `python scripts\registry.py list` to inspect versions and
`python scripts\registry.py rollback <id>` to switch manually, then call `POST /reload-model`
(or restart the API).

## Docker

One image both trains and serves, so the model is always loaded with the library versions that
created it. Data and models stay on the host and are mounted into the container (`/storage/...`).

```powershell
docker compose build
docker compose run --rm train      # trains inside the container, writes ./models
docker compose up -d api           # serves on http://127.0.0.1:8000
docker compose down
```

## Key results

- Always predicting "stays" scores 73.5% accuracy and catches 0 churners, so accuracy misleads.
- Logistic regression, random forest, gradient boosting, polynomial features and splines all land
  at about 0.845 ROC-AUC. The differences are smaller than the fold-to-fold noise (about 0.012).
- All three numeric columns are mildly non-linear (MonthlyCharges most, tenure least), but bending them
  barely helps: churn is driven mostly by categorical columns such as contract type.
- Leakage checks came back clean (no suspicious column, shuffled labels score about 0.5).

## Design decisions

- **Final model: plain logistic regression.** Everything ties within noise, so the simplest wins.
- **One pipeline.** `TotalCharges` cleaning, scaling and encoding are all pipeline steps,
  so the saved model accepts raw customer records (no train/serve skew).
- **Data-driven choices use training data only.** The cutoff and the curved-column list are chosen
  without looking at the test set; the test set is scored once.
- **The cutoff is a business decision.** Default is the best-F1 cutoff (0.35). The cost-based option
  uses placeholder costs in `config.py` and must be replaced with real numbers.
- **Versioned models with a promotion rule.** Nothing is overwritten; a clearly worse model is saved
  but never served, and any version can be rolled back to.
- **Paths come from the environment in containers.** `CHURN_DATA_PATH` and `CHURN_MODELS_DIR`
  override the local defaults.

## Known limits

- Blank or garbage `TotalCharges` at prediction time silently becomes 0.
- Promotion compares test ROC-AUC against the same fixed test split. That is fine for fixed data;
  with new data, use a fresh holdout set.
- The model is trained on 80% of the data; refitting on all data is a possible improvement.
- `/reload-model` has no authentication; put the API behind a trusted network or add auth before exposing it.
- The API scores one customer per request (no batch endpoint).

## Not built yet

MLflow tracking, drift monitoring, a batch prediction endpoint.
