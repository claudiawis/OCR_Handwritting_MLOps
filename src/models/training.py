# src/models/training.py
from fastapi import FastAPI
import subprocess

app = FastAPI(title="Training Service")

@app.get("/train")
def run_training():
    try:
        # Call the training pipeline, e.g., using dvc repro with appropriate downstream targets
        result = subprocess.run(["sh", "-c", "dvc repro --downstream setup_callbacks"],
                                capture_output=True, text=True, check=True)
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": e.stderr}
