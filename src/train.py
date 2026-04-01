import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    import mlflow
except ImportError:
    mlflow = None


def load_dataset(data_path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(data_path, usecols=[text_col, label_col], encoding="latin-1")
    df = df.rename(columns={text_col: "text", label_col: "label"})
    return df


def normalize_labels(labels: pd.Series) -> pd.Series:
    mapping = {
        "ham": 0,
        "spam": 1,
        "0": 0,
        "1": 1,
        0: 0,
        1: 1,
    }
    mapped = labels.map(mapping)
    if mapped.isna().any():
        raise ValueError("Unsupported labels found. Use ham/spam or 0/1")
    return mapped.astype(int)


def train_pipeline(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
                max_features=20000,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
                C=5.0,
            )),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def save_artifacts(model: Pipeline, metrics: dict, model_dir: str, metrics_dir: str, model_version: str) -> None:
    model_path = Path(model_dir)
    metrics_path = Path(metrics_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    versioned_model = model_path / f"model_{model_version}.joblib"
    latest_model = model_path / "model_latest.joblib"
    metrics_file = metrics_path / f"metrics_{model_version}.json"

    joblib.dump(model, versioned_model)
    joblib.dump(model, latest_model)

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_training(data_path: str, model_dir: str, metrics_dir: str, model_version: str, text_col: str = "text", label_col: str = "label") -> dict:
    df = load_dataset(data_path, text_col=text_col, label_col=label_col)
    X = df["text"].astype(str)
    y = normalize_labels(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = train_pipeline(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    save_artifacts(model, metrics, model_dir, metrics_dir, model_version)

    if mlflow is not None:
        mlflow.set_experiment("spam-classifier")
        with mlflow.start_run(run_name=model_version):
            mlflow.log_params({
                "model_version": model_version,
                "data_path": data_path,
                "test_size": 0.2,
                "ngram_range": "1,2",
                "class_weight": "balanced",
                "C": 5.0,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train spam classification model")
    parser.add_argument("--data-path", default="data/spam.csv", help="Path to CSV dataset")
    parser.add_argument("--model-dir", default="models", help="Directory to save model artifacts")
    parser.add_argument("--metrics-dir", default="metrics", help="Directory to save metrics")
    parser.add_argument("--model-version", default="v1", help="Model version tag")
    parser.add_argument("--text-col", default="text", help="Name of the text column")
    parser.add_argument("--label-col", default="label", help="Name of the label column")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = run_training(
        data_path=args.data_path,
        model_dir=args.model_dir,
        metrics_dir=args.metrics_dir,
        model_version=args.model_version,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    print(json.dumps(metrics, indent=2))
