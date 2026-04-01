import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib


class ModelService:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self.load_model_if_available()

    def load_model_if_available(self) -> None:
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, text: str) -> Dict[str, Any]:
        prediction_num = int(self.model.predict([text])[0])
        proba = float(self.model.predict_proba([text])[0][1])
        prediction_label = "spam" if prediction_num == 1 else "ham"
        return {"prediction": prediction_label, "spam_probability": proba}


def log_prediction(text: str, result: Dict[str, Any], logs_path: str) -> None:
    log_file = Path(logs_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "prediction": result["prediction"],
        "spam_probability": result["spam_probability"],
    }

    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
