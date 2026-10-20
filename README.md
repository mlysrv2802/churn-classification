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
  train.py        train the final model and save it
  predict.py      ChurnPredictor: load once, score raw records
scripts/          entry points, run with `python scripts\<name>.py`
tests/            pytest suite
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
| `train.py` | trains the final model, saves it to `models/` |
| `predict_demo.py` | scores raw customer rows with the saved model |

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

## Known limits

- Blank or garbage `TotalCharges` at prediction time silently becomes 0.
- Training overwrites the saved model; there is no versioned registry yet.
- The model is trained on 80% of the data; refitting on all data is a possible improvement.

## Not built yet

API (FastAPI) and Docker, versioned model registry, MLflow tracking, drift monitoring.
