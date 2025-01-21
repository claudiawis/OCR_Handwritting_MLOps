#this is the authentification and authorization API that is giving basic security to our pilot project. For life deployment this should be replaced with a higher security application
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from fastapi.responses import RedirectResponse

app = FastAPI()
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

users = {

    "admin1": {
        "username": "admin1",
        "name": "Admin",
        "hashed_password": pwd_context.hash('1nimda'),
    },

    "user1" : {
        "username" :  "user1",
        "name" : "User",
        "hashed_password" : pwd_context.hash('1resu'),
    }

}

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    if not(users.get(username)) or not(pwd_context.verify(credentials.password, users[username]['hashed_password'])):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/user")
def current_user(username: str = Depends(get_current_user)):
    # defining redirection
    redirect_url = "https://rpubs.com/helenabakic/women-in-open-data"  
    return RedirectResponse(url=redirect_url)