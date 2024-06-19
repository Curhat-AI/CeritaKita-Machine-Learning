import os
from typing import Union
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import *
from text_recognition_model import predict_text
from image_recognition_model import predict_image
from mental_issue_recognition_model import predict_mental_issue
from recommendation_system_conseulor import recommendation_result

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

class TextInput(BaseModel):
    text: str

class MentalTextInput(BaseModel):
    text: str

class RecommendationInput(BaseModel):
    gender: str = "f/m"
    counselourType: str = "peer/professional"
    dateUp: str = "YYYY-mm-dd"
    dateDown: str = "YYYY-mm-dd"
    timeUp: int = 20.5
    timeDown: int = 18
    # ageUp: int = 28
    # ageDown: int = 18
    # counselingType: str


@app.post("/predict/text")
async def predict(input: TextInput):
    try:
        predictions = predict_text(input.text)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        predicted_class = predict_image(contents)
        return {"predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict/mental-issue")
async def predict_mental(input: MentalTextInput):
    try:
        predictions = predict_mental_issue(input.text)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/reccomendation/counselour")
async def recommendation_counselour(input: RecommendationInput):
    try:
        recommendation = recommendation_result(input)
        return {"id_counselor_recommendation": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT", 8000))
