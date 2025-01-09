from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
import os

# Initialize FastAPI app
app = FastAPI()

# Remote model URL (update with your DagsHub link)
MODEL_URL = "https://dagshub.com/KazemZh/OCR_Handwritting_MLOps/raw/main/models/CNN.h5"

# Function to download the model from the repository if not already present locally
def download_model():
    response = requests.get(MODEL_URL)
    with open("models/CNN.h5", "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded successfully.")

# Download the model
download_model()

# Load the trained model
model = load_model("models/CNN.h5")

# Define the class labels
class_labels = ['A', 'made', 'may', 'two', 'We', 'But', 'told', 'And', 'new', 'This', 'first', 'people', 'In', 'much', 'could', 'time', 'man', 'like', 'well', 'You']

# Image preprocessing function
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesses the uploaded image to the format required by the model.
    """
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to the model's input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = image_array.reshape(-1, 28, 28, 1)  # Reshape for model input
    return image_array

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image = Image.open(file.file)
        
        # Preprocess the image
        image_array = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(image_array)
        predicted_label = class_labels[np.argmax(prediction)]
        
        return {"predicted_text": predicted_label}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
