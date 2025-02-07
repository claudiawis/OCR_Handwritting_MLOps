# src/data/ingestion.py
from fastapi import FastAPI
import subprocess

app = FastAPI(title="Ingestion Service")

@app.get("/ingest")
def run_ingestion():
    try:
        # You can call the DVC pipeline command here (or call a Python function)
        # For example, using subprocess to call "dvc repro one_hot_encode_labels"
        result = subprocess.run(["dvc", "repro", "one_hot_encode_labels"],
                                capture_output=True, text=True, check=True)
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": e.stderr}
