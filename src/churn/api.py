from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request

from churn.config import MODELS_DIR
from churn.predict import ChurnPredictor
from churn.schemas import Customer, Prediction


def create_app(model_dir: Path = MODELS_DIR) -> FastAPI:
    """Build the API. A factory (rather than a global) lets tests point it at a temp model."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once at startup: load the saved model a single time and reuse it
        # for every request (like a Spring singleton bean).
        try:
            app.state.predictor = ChurnPredictor(model_dir)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"No production model in {model_dir}. Run `python scripts\\train.py` first."
            ) from exc
        yield

    app = FastAPI(title="Churn prediction API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info(request: Request) -> dict:
        return request.app.state.predictor.metadata

    # Plain `def` (not `async def`): scoring is CPU work, so FastAPI runs it in a
    # worker thread and the server stays responsive to other requests.
    @app.post("/predict")
    def predict(customer: Customer, request: Request) -> Prediction:
        predictor: ChurnPredictor = request.app.state.predictor
        scored = predictor.predict(pd.DataFrame([customer.model_dump()])).iloc[0]
        return Prediction(
            churn_probability=float(scored["churn_probability"]),
            flagged=bool(scored["flagged"]),
            threshold=predictor.metadata["threshold"],
            model_version=predictor.metadata["version"],
        )

    return app


app = create_app()  # `uvicorn churn.api:app`
