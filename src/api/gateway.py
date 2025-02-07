from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
import subprocess
import httpx
import asyncio
import os
from passlib.context import CryptContext

# definition of app including security setup
app = FastAPI(title="Gateway Service")
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# These addresses assume that your docker-compose networking is used,
# and that service names are used as hostnames.
INGESTION_URL = "http://ingestion_service:8100"
TRAINING_URL = "http://training_service:8200"
PREDICTION_URL = "http://prediction_service:8300"

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
        print("🚫 Unauthorized access attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    print(f"✅ Authenticated user: {username}, Role: {users[username]['role']}")
    return users[username]

@app.get("/")
def home():
    return {"message": "Welcome to the OCR Gateway Service"}

@app.get("/ingest")
async def trigger_ingestion(user: dict = Depends(get_current_user)):
    if user["role"] != "admin":
        print("🚫 Unauthorized access to ingestion endpoint")
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    print("📥 Starting ingestion process...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{INGESTION_URL}/ingest")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"❌ Ingestion failed: {exc}")
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    print("✅ Ingestion completed successfully")
    return {"ingestion_result": response.json()}

@app.get("/train")
async def trigger_training(user: dict = Depends(get_current_user)):
    if user["role"] != "admin":
        print("🚫 Unauthorized access to training endpoint")
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    print("🎓 Starting training process...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{TRAINING_URL}/train")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"❌ Training failed: {exc}")
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    print("✅ Training completed successfully")
    return {"training_result": response.json()}

@app.post("/predict")
async def trigger_prediction(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    print(f"📩 Received prediction request from user: {user['username']}, Role: {user['role']}")
    
    async with httpx.AsyncClient() as client:
        try:
            file_content = await file.read()
            files = {"file": (file.filename, file_content, file.content_type)}
            response = await client.post(f"{PREDICTION_URL}/predict", files=files)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"❌ Prediction failed: {exc}")
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    
    print("✅ Prediction successful")
    return {"prediction_result": response.json()}