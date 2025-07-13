import json
import os
from datetime import datetime

HISTORY_FILE = "training_history.json"

def save_metrics(metrics: dict):
    record = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "MAE": round(metrics["MAE"], 2),
        "RMSE": round(metrics["RMSE"], 2)
    }

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(record)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)
