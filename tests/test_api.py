from pathlib import Path

from fastapi.testclient import TestClient
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import main


def _build_dummy_model(model_path: Path) -> None:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    X = [
        "free money now",
        "claim your prize",
        "see you at lunch",
        "project update attached",
    ]
    y = [1, 1, 0, 0]
    pipeline.fit(X, y)
    dump(pipeline, model_path)


def test_root_endpoint() -> None:
    client = TestClient(main.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_requires_loaded_model() -> None:
    original_model = main.model_service.model
    main.model_service.model = None

    client = TestClient(main.app)
    response = client.post("/predict", json={"text": "free bitcoin now"})

    assert response.status_code == 503
    main.model_service.model = original_model


def test_predict_success_with_loaded_model(tmp_path: Path) -> None:
    model_path = tmp_path / "tmp_model.joblib"
    _build_dummy_model(model_path)

    original_model = main.model_service.model
    main.model_service.model = None
    main.model_service.model_path = model_path
    main.model_service.load_model_if_available()

    client = TestClient(main.app)
    response = client.post("/predict", json={"text": "you won free tickets"})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"ham", "spam"}
    assert 0.0 <= body["spam_probability"] <= 1.0

    main.model_service.model = original_model
