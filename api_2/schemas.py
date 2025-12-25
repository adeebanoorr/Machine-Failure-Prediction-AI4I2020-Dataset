from pydantic import BaseModel

class InputSchema(BaseModel):
    product_id: str
    type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


class OutputSchema(BaseModel):
    prediction: int
    probability: float
    message: str
