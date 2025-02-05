from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
import subprocess
import numpy as np
import mlflow.pyfunc
from PIL import Image
import os
from passlib.context import CryptContext

# Initialize FastAPI app
app = FastAPI()
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# MLflow tracking details
MLFLOW_TRACKING_URI = "https://dagshub.com/KazemZh/OCR_Handwritting_MLOps.mlflow"
EXPERIMENT_NAME = "OCR_CNN_Training"

# Users dictionary with hashed passwords and roles
users = {
    "admin1": {
        "username": "admin1",
        "name": "Admin",
        "hashed_password": pwd_context.hash('1nimda'),
        "role": "admin",
    },
    "user1": {
        "username": "user1",
        "name": "User",
        "hashed_password": pwd_context.hash('1resu'),
        "role": "user",
    }
}

# Function to verify user credentials and get role
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    if username not in users or not pwd_context.verify(credentials.password, users[username]['hashed_password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return users[username]  # Returns user dictionary with role

# Function to load the latest model from MLflow
def load_latest_model():
    print("🔍 Fetching the latest model from MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

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
    print("✅ Latest model loaded successfully.")
    return mlflow.pyfunc.load_model(model_uri)

# Load model on startup
model = load_latest_model()

# Class labels
class_labels = ['A', 'made', 'may', 'two', 'We', 'But', 'told', 'And', 'new', 'This', 'first', 
                'people', 'In', 'much', 'could', 'time', 'man', 'like', 'well', 'You']

# Image preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L")  
    image = image.resize((28, 28))  
    image_array = np.array(image) / 255.0  
    image_array = image_array.reshape(-1, 28, 28, 1)  
    return image_array

# Prediction Endpoint (accessible by both users & admins)
@app.post("/predict/")
def predict(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    role = user["role"]
    print(f"📩 Received prediction request from user role: {role}")

    try:
        if role not in ["user", "admin"]:
            print("🚫 Unauthorized access attempt to /predict")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        # Load and preprocess the image
        image = Image.open(file.file)
        image_array = preprocess_image(image)

        # Make the prediction
        prediction = model.predict(image_array)
        predicted_label = class_labels[np.argmax(prediction)]
        
        print(f"✅ Prediction successful for user role: {role}, Predicted text: {predicted_label}")
        return {"predicted_text": predicted_label}

    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Retraining endpoint (only accessible by admins)
@app.post("/retrain")
def retrain_model(user: dict = Depends(get_current_user)):
    role = user["role"]
    print(f"🔄 Retraining request received from user role: {role}")

    if role != "admin":
        print("🚫 Unauthorized access attempt to /retrain")
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Step 1: Run ingestion
        print("📥 Starting ingestion process...")
        ingestion_process = subprocess.Popen([
            "docker", "run", "--rm",
            "-v", "$(pwd)/data/raw:/app/data/raw",
            "-v", "$(pwd)/.dvc:/app/.dvc",
            "-v", "$(pwd)/.git:/app/.git",
            "ingestion_image"
        ])
        ingestion_process.wait()
        print("✅ Ingestion completed.")

        # Step 2: Run training
        print("🎓 Starting model training...")
        training_process = subprocess.Popen([
            "docker", "run", "--rm",
            "-v", "$(pwd)/data:/app/data",
            "-v", "$(pwd)/models:/app/models",
            "-v", "$(pwd)/.dvc:/app/.dvc",
            "-v", "$(pwd)/.git:/app/.git",
            "training_image"
        ])
        training_process.wait()
        print("✅ Training completed.")

        return {"status": "Retraining process completed"}
    except Exception as e:
        print(f"❌ Retraining failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

