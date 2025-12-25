import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
from schemas import InputSchema, OutputSchema
from inference import make_prediction, make_batch_predictions

app = FastAPI(title="Machine Failure Prediction API")

# FIX: Add CORS Middleware to allow your React app (port 5173) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves the feature importance image from your local api_2 directory
current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=current_dir), name="static")

@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict", response_model=OutputSchema)
def predict_single(input_data: InputSchema):
    """Predict failure for a single record"""
    pred, prob = make_prediction(input_data.model_dump())
    return OutputSchema(
        prediction=pred,
        probability=round(prob, 4),
        message="Failure predicted" if pred == 1 else "No failure predicted"
    )

@app.post("/batch_predict", response_model=List[OutputSchema])
def predict_batch(inputs: List[InputSchema]):
    """Predict failure for a list of records"""
    results = make_batch_predictions([x.model_dump() for x in inputs])
    return [
        OutputSchema(
            prediction=pred,
            probability=round(prob, 4),
            message="Failure predicted" if pred == 1 else "No failure predicted"
        )
        for pred, prob in results
    ]