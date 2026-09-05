import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model once when the app starts
model = joblib.load("house_price_model.pkl")

# Create the FastAPI app
app = FastAPI()

# Define what a valid incoming request must look like
class HouseFeatures(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    GarageArea: int
    TotalBsmtSF: int
    FirstFlrSF: int
    FullBath: int
    YearBuilt: int

# Define what happens when a prediction request arrives
@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_data = np.array([[
        features.OverallQual,
        features.GrLivArea,
        features.GarageCars,
        features.GarageArea,
        features.TotalBsmtSF,
        features.FirstFlrSF,
        features.FullBath,
        features.YearBuilt
    ]])

    prediction = model.predict(input_data)

    return {"predicted_price": float(prediction[0])}