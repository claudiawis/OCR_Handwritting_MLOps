from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

# OAuth2 password bearer token setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

# Mock database of users and roles
users_db = {
    "user1": {"password": "password1", "role": "user"},
    "admin1": {"password": "password2", "role": "admin"},
}

# Model to represent a user
class User(BaseModel):
    username: str
    role: str

# Dependency to get current user based on OAuth2 token
def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    if token not in users_db:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_data = users_db[token]
    return User(username=token, role=user_data["role"])

# Role-based dependency to ensure users can only access certain endpoints based on their roles
def role_required(required_role: str):
    def role_dependency(current_user: User = Depends(get_current_user)):
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have {required_role} privileges",
            )
        return current_user
    return role_dependency

@app.post("/access-ocr/")
async def access_ocr(current_user: User = Depends(role_required("user"))):
    """Allow users to access OCR functionality (delegated to another service)"""
    # redirect or pass the request to the actual OCR service here (not yet implemented)
    return {"message": "User is authorized to use OCR service."}

@app.post("/trigger-retraining/")
async def trigger_retraining(current_user: User = Depends(role_required("admin"))):
    """Allow admins to trigger retraining (delegated to another service)"""
    # send a signal to your retraining system here (not yet implemented)
    return {"message": "Admin is authorized to trigger model retraining."}

# Example for login (
@app.post("/token")
async def login(username: str, password: str):
    """Draft login endpoint to generate token"""
    for user, user_data in users_db.items():
        if user == username and user_data["password"] == password:
            return {"access_token": user, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

