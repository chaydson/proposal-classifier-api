from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import init
from app.model.model import run_model

app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    proposal: str

@app.get("/")
def home():
    init()
    return {"status": 200}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    proposal = run_model(payload.text)
    print(payload.text)
    return {"proposal": proposal}