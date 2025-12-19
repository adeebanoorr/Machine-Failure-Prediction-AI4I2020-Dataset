from pydantic import BaseModel

class PredictionRequest(BaseModel):
    product_id: str
    type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    message: str
