# ML API in Production - Spam Classification

End-to-end MLOps project to train, serve, and deploy a spam classifier.

## What this project includes

- Reproducible training pipeline with scikit-learn.
- FastAPI service for real-time inference.
- Prediction logging for basic monitoring.
- Dockerfile for containerized deployment.
- GitHub Actions workflow for CI.
- Tests for API and training pipeline.

## Project structure

```
.
|-- data/
|   `-- spam.csv
|-- src/
|   |-- __init__.py
|   |-- main.py
|   |-- model_utils.py
|   |-- schemas.py
|   `-- train.py
|-- tests/
|   |-- test_api.py
|   `-- test_train.py
|-- .github/workflows/ci.yml
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Train model

```bash
python -m src.train --data-path data/spam.csv --model-version v1
```

Artifacts created:

- `models/model_v1.joblib`
- `models/model_latest.joblib`
- `metrics/metrics_v1.json`

## 3) Run API locally

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/
```

Predict:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a free ticket. Reply now."}'
```

## 4) Monitoring (basic)

Every prediction is logged in `logs/predictions.jsonl` with:

- timestamp
- input text
- prediction label
- spam probability

This provides a simple baseline for production monitoring.

## 5) Docker

Build image:

```bash
docker build -t ml-spam-api .
```

Run container:

```bash
docker run -p 10000:10000 ml-spam-api
```

## 6) Deploy to Render

1. Push this repo to GitHub.
2. In Render create a new Web Service.
3. Connect your repository.
4. Use Docker as runtime.
5. Expose port `10000`.

## CV-ready line

Built and deployed a containerized ML service with FastAPI for real-time spam detection, including reproducible training pipelines, Docker-based deployment, CI checks, and prediction logging for baseline monitoring.
