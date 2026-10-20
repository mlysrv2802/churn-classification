FROM python:3.12-slim

WORKDIR /app

# Where the data and the trained model live; mounted from the host (see docker-compose.yml)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHURN_DATA_PATH=/storage/data/telco_churn.csv \
    CHURN_MODELS_DIR=/storage/models

# Dependencies first: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Then our code. --no-deps because requirements.txt already installed the runtime packages
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps .
COPY scripts ./scripts

# Run as a normal user, not root
RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
