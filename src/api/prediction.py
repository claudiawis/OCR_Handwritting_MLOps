# src/api/prediction.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Summary
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from PIL import Image
import numpy as np
import os

app = FastAPI(title="Prediction Service")

# Integrate Prometheus instrumentation if needed
Instrumentator().instrument(app).expose(app)

# MLflow configuration
MLFLOW_TRACKING_URI = "https://dagshub.com/KazemZh/OCR_Handwritting_MLOps.mlflow"
EXPERIMENT_NAME = "OCR_CNN_Training"

# Create a Summary to record inference time
inference_time_summary = Summary('inference_time_seconds', 'Time taken for inference')

def load_latest_model():
    print("🔍 Fetching the latest model from MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise Exception(f"Experiment '{EXPERIMENT_NAME}' not found.")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_accuracy DESC"],
        max_results=1,
    )
    if not runs:
        raise Exception("No runs found in the experiment.")

    latest_run = runs[0]
    model_uri = f"runs:/{latest_run.info.run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Latest model loaded successfully.")
    return model

# Load the model at startup
model = load_latest_model()

# Define your class labels (modify as needed)
class_labels = ['A', 'made', 'may', 'two', 'We', 'But', 'told', 'And', 'new', 'This', 
                'first', 'people', 'In', 'much', 'could', 'time', 'man', 'like', 'well', 'You']

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L")  # grayscale
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(-1, 28, 28, 1)
    return image_array

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        image_array = preprocess_image(image)
        with inference_time_summary.time():
            prediction = model.predict(image_array)
            predicted_label = class_labels[np.argmax(prediction)]
        return {"predicted_text": predicted_label}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
