from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- Add this
from api.schemas import PredictionRequest, PredictionResponse
from api.inference import predict

app = FastAPI(title="Machine Failure Prediction API")

# ---------- Add CORS ----------
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)
# ---------- End CORS ----------

@app.post("/predict", response_model=PredictionResponse)
def predict_failure(req: PredictionRequest):
    try:
        prediction, probability = predict(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        message="Failure predicted" if prediction else "No failure predicted"
    )

@app.get("/")
def health():
    return {"status": "API running"}
