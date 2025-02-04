from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from passlib.context import CryptContext
import requests

# Initialize FastAPI app
app = FastAPI()
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Define allowed origins for CORS
origins = [
    "http://localhost:8000",  # Service API
    "http://localhost:8111",  # Gateway API (if running on this port)
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Function to verify user credentials
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    if username not in users or not pwd_context.verify(credentials.password, users[username]['hashed_password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return users[username]

# Redirect endpoint (for role-based access)
@app.get("/redirect")
def redirect_user(user: dict = Depends(get_current_user)):
    role = user["role"]
    
    # Target API URL (modify if needed)
    target_url = "http://localhost:8000/predict/"  
    
    # Redirect to the target API with role in headers
    return JSONResponse(content={"redirect_url": target_url}, status_code=status.HTTP_303_SEE_OTHER)

# Forward request with file upload (ensuring multipart/form-data is preserved)
@app.post("/forward_request/")
async def forward_request(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    role = user["role"]
    target_url = "http://localhost:8000/predict/"

    # Prepare the file as multipart/form-data
    files = {"file": (file.filename, file.file, file.content_type)}
    headers = {"X-User-Role": role}  # Include user role in headers

    # Send request to the service API
    response = requests.post(target_url, files=files, headers=headers)

    return JSONResponse(content=response.json(), status_code=response.status_code)



