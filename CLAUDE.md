# Context for Claude (handoff from earlier chat)

## About me
- Malay, Java + Angular fullstack developer (5 yrs Angular, but away from frontend ~3 yrs). Learning ML hands-on.
- Did a Python ML/DL course earlier, little practical experience. Wants production-grade code, not notebook toys.
- Windows, PowerShell, VS Code. Python 3.12 venv (3.14 breaks pandas/numpy wheels). Docker Desktop + WSL2 installed.

## How I want you to work
- Go step by step: give ONE step, I run it and report back, then give the next. Don't dump everything at once.
- Explain concepts with Java/Spring/Angular analogies where possible.
- Test code before handing it over when you can. Say plainly when you couldn't run something.
- Prefer small, incremental edits to my existing files (tell me exactly what to find and replace).

## Project 1: housing-price-project (regression, finished)
Location: D:\Malay\ws\ml\repo\housing-price-project
- housing-price-predictor-backend/  Python: config.py, data_loader.py, generate_data.py, train.py,
  predict.py, api.py (FastAPI), fetch_new_data.py, scheduler.py (APScheduler), model_registry.py,
  eda_feature_analysis.py, check_linearity.py, compare_models.py, tune_hyperparameters.py,
  tune_randomforest.py, tests/, Dockerfile, .dockerignore (must exclude venv/)
- housing-price-ui/  Angular 22 standalone app (reactive form -> POST /predict), nginx Dockerfile
- storage/  data/ and models/ live OUTSIDE the code folders and are bind-mounted into containers at /storage/...
- docker-compose.yml services: bootstrap (one-shot, trains first model), api (:8000), scheduler, ui (:4200)
  api and scheduler wait on bootstrap with condition: service_completed_successfully

Key design decisions already made:
- Preprocessing + model live in ONE sklearn Pipeline (no train/serve skew), saved with joblib.
- Every training run saves a versioned model; model_registry promotes it to production only if R2 does
  not drop more than 0.02 (MAX_ACCEPTABLE_R2_DROP). Manual rollback via `python model_registry.py rollback <id>`.
- API loads the production model once at startup; after retraining, `docker compose restart api`.
- Scheduler: fetch from /new-listings (deliberately messy data) -> validate/clean -> append CSV ->
  retrain only if new rows arrived. Interval is 30 min. APScheduler gotcha: next_run_time=None PAUSES the job.
- Inside Docker the scheduler reaches the API at http://api:8000 (env var API_BASE_URL), not 127.0.0.1.
- Results: Linear/Ridge/Lasso R2 ~0.82 (data is genuinely linear); tuned RandomForest ~0.76.

## Project 2: churn-classification (in progress)
Location: D:\Malay\ws\ml\repo\churn-classification  (data/telco_churn.csv from Kaggle "Telco Customer Churn")
Goal: learn classification on real, imbalanced, messy data.
Done: first_look.py (shape, class balance, TotalCharges is text with blanks), baseline.py (Dummy vs
LogisticRegression, shows accuracy misleads; blank TotalCharges -> 0, customerID dropped, stratified split).
Roadmap, in order:
1. First look and cleaning (done)
2. Baseline model (done, waiting on my real output)
3. Proper metrics: confusion matrix, precision, recall, F1, ROC-AUC
4. Other models: random forest, gradient boosting, compared fairly like compare_models.py
5. Polynomial features and non-linearity
6. RandomizedSearchCV tuning plus a data leakage check
Later: CI/CD with GitHub Actions, MLflow, drift monitoring, GenAI/LLM APIs (hallucinations belong here).

## Parked / ideas
- Containerize everything is done; maybe add Docker `include` so backend compose can run standalone.
- A /reload-model endpoint instead of restarting the api container.
