from fastapi import FastAPI, HTTPException, UploadFile, File
import httpx
import asyncio

app = FastAPI(title="Gateway Service")

# These addresses assume that your docker-compose networking is used,
# and that service names are used as hostnames.
INGESTION_URL = "http://ingestion_service:8100"
TRAINING_URL = "http://training_service:8200"
PREDICTION_URL = "http://prediction_service:8300"

@app.get("/")
def home():
    return {"message": "Welcome to the OCR Gateway Service"}

@app.get("/ingest")
async def trigger_ingestion():
    """
    Forwards a GET request to the ingestion service.
    (Assuming the ingestion service exposes a /ingest endpoint.)
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{INGESTION_URL}/ingest")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    return {"ingestion_result": response.json()}

@app.get("/train")
async def trigger_training():
    """
    Forwards a GET request to the training service.
    (Assuming the training service exposes a /train endpoint.)
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{TRAINING_URL}/train")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    return {"training_result": response.json()}

@app.post("/predict")
async def trigger_prediction(file: UploadFile = File(...)):
    """
    Forwards an uploaded file to the prediction service's /predict endpoint.
    """
    async with httpx.AsyncClient() as client:
        try:
            # Read file contents and pass them along
            file_content = await file.read()
            files = {"file": (file.filename, file_content, file.content_type)}
            response = await client.post(f"{PREDICTION_URL}/predict", files=files)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    return {"prediction_result": response.json()}