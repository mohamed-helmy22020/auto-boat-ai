import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import Dict

# Create a FastAPI app and router
app = FastAPI()
api_router = APIRouter()

# Load model configuration
with open("model_config.json", "r") as f:
    config = json.load(f)

# Load all models
models = {
    name: ort.InferenceSession(model_info["model_path"])
    for name, model_info in config["models"].items()
}


# Define Pydantic model for input validation
class WeatherInput(BaseModel):
    mo: float
    da: float
    temp: float
    dewp: float
    slp: float
    stp: float
    visib: float
    wdsp: float
    mxpsd: float
    gust: float
    prcp: float
    sndp: float


@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Hello from the API!"}


# Root endpoint
@api_router.get("/", include_in_schema=False)
def read_root():
    return {"message": "Hello from the API!"}


# Predict for a single model
@api_router.post("/predict/{model_name}")
def predict(model_name: str, data: WeatherInput):
    if model_name not in models:
        return {"error": "Model not found"}

    # Convert input to numpy array
    input_array = np.array(
        [
            [
                data.mo,
                data.da,
                data.temp,
                data.dewp,
                data.slp,
                data.stp,
                data.visib,
                data.wdsp,
                data.mxpsd,
                data.gust,
                data.prcp,
                data.sndp,
            ]
        ],
        dtype=np.float32,
    )

    # Run inference
    model = models[model_name]
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    result = model.run([output_name], {input_name: input_array})

    return {model_name: result[0].tolist()[0]}


# New endpoint: Predict using all models
@api_router.post("/predict_all")
def predict_all(data: WeatherInput):
    predictions = {}

    # Convert input to numpy array
    input_array = np.array(
        [
            [
                data.mo,
                data.da,
                data.temp,
                data.dewp,
                data.slp,
                data.stp,
                data.visib,
                data.wdsp,
                data.mxpsd,
                data.gust,
                data.prcp,
                data.sndp,
            ]
        ],
        dtype=np.float32,
    )

    # Run inference for all models
    for model_name, model in models.items():
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        result = model.run([output_name], {input_name: input_array})

        # Store prediction in a dictionary
        predictions[model_name] = result[0].tolist()[0]

    return predictions


# Include the router with the /api prefix
app.include_router(api_router, prefix="/api")
