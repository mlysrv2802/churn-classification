# Context for Claude (handoff from earlier chat)

## About me
- Malay, Java + Angular fullstack developer (5 yrs Angular, but away from frontend ~3 yrs). Learning ML hands-on.
- Did a Python ML/DL course earlier, little practical experience. Wants production-grade code, not notebook toys.
- Windows, PowerShell, VS Code. Python 3.14.7 venv works now (tested Oct 2026: pandas 3.0, numpy 2.5 have 3.14 wheels; CI runs 3.12 and 3.14). Docker Desktop + WSL2 installed.

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

## Project 2: churn-classification (roadmap 1-6 done; model trained and saved)
Location: D:\Malay\ws\ml\repo\churn-classification  (data/telco_churn.csv from Kaggle "Telco Customer Churn")
GitHub: https://github.com/mlysrv2802/churn-classification (CI = pytest on Python 3.12 + 3.14, every push)
Goal: learn classification on real, imbalanced, messy data.
Layout: src/churn/ (library), scripts/ (entry points), tests/ (pytest). Setup + script list are in README.md.
Results: logistic regression, random forest, gradient boosting, polynomial, splines all ~0.845 ROC-AUC,
tied within noise (std ~0.012). Final model = plain logistic regression (simplest within noise).
Numeric columns are all mildly non-linear (MonthlyCharges most, tenure least) but curves barely help.
Leakage checks clean. Cutoff 0.35 chosen by F1 on train out-of-fold predictions; cost-based cutoff
uses PLACEHOLDER costs in config.py (replace with real ones).
Key decisions: TotalCharges cleaning is the pipeline's first step (prepare_features), so the saved model
takes raw records; anything chosen from data (cutoff, curved columns) uses TRAIN data only.
Saved model: models/churn_model.joblib + .json metadata (git-ignored); training overwrites it (no registry yet).
Not built yet: FastAPI + Docker, versioned model registry, MLflow, drift monitoring, GenAI/LLM APIs.
Maybe: tidy the redundant degree-1 PolynomialFeatures + double scaling in the plain pipeline.

## Parked / ideas
- Containerize everything is done; maybe add Docker `include` so backend compose can run standalone.
- A /reload-model endpoint instead of restarting the api container.
