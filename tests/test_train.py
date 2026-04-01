from pathlib import Path

from src.train import run_training


def test_run_training_creates_artifacts(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"

    metrics = run_training(
        data_path="data/spam.csv",
        model_dir=str(model_dir),
        metrics_dir=str(metrics_dir),
        model_version="test",
        text_col="v2",
        label_col="v1",
    )

    assert "accuracy" in metrics
    assert (model_dir / "model_test.joblib").exists()
    assert (model_dir / "model_latest.joblib").exists()
    assert (metrics_dir / "metrics_test.json").exists()
