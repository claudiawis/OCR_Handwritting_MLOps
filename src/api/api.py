from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Summary
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import mlflow.pyfunc
import os

# Initialize FastAPI app
app = FastAPI()

# Integrate Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# MLflow tracking URI and experiment details
MLFLOW_TRACKING_URI = "https://dagshub.com/KazemZh/OCR_Handwritting_MLOps.mlflow"
EXPERIMENT_NAME = "OCR_CNN_Training"

# Create an object Summary to save the inference time
inference_time_summary = Summary('inference_time_seconds', 'Time taken for inference')

# Function to retrieve the latest model from MLflow
def load_latest_model():
    print("🔍 Fetching the latest model from MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Get the experiment ID
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise Exception(f"Experiment '{EXPERIMENT_NAME}' not found.")
    
    # Get the latest run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_accuracy DESC"],  # Adjust based on your metric
        max_results=1,
    )
    if not runs:
        raise Exception("No runs found in the experiment.")

    # Fetch the model URI from the latest run
    latest_run = runs[0]
    model_uri = f"runs:/{latest_run.info.run_id}/model"

    # Load the model
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Latest model loaded successfully.")
    return model

# Load the latest trained model
model = load_latest_model()

# Define the class labels
class_labels = ['A', 'made', 'may', 'two', 'We', 'But', 'told', 'And', 'new', 'This', 'first', 'people', 'In', 'much', 'could', 'time', 'man', 'like', 'well', 'You']

# Image preprocessing function
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to the model's input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = image_array.reshape(-1, 28, 28, 1)  # Reshape for model input
    return image_array

#######################################
# Prediction Endpoint
@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        image = Image.open(file.file)
        image_array = preprocess_image(image)

        # Make the prediction
        with inference_time_summary.time():
            prediction = model.predict(image_array)
            predicted_label = class_labels[np.argmax(prediction)]

        return {"predicted_text": predicted_label}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})