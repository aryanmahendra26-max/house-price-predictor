# House Price Predictor

A machine learning pipeline that predicts house sale prices from property 
features, trained on the Kaggle "House Prices - Advanced Regression Techniques" 
dataset, and deployed as a live REST API using FastAPI.

## What this project covers
- Data cleaning and exploratory analysis on 1460 real housing records, 
  including reasoned handling of missing values (e.g. distinguishing "feature 
  doesn't exist" from "data genuinely missing")
- Model comparison: Linear Regression (RMSE ~$33,263) vs Random Forest 
  (RMSE ~$29,272) on the full feature set
- A simplified 8-feature Random Forest model (RMSE ~$30,075) for deployment
- A FastAPI REST endpoint that serves live price predictions from the trained model

## Tech stack
Python, Pandas, scikit-learn, FastAPI, Uvicorn, joblib

## Running locally

1. Clone this repo and navigate into it
2. Create a virtual environment and activate it
3. Install dependencies:
4. Start the server:
5. Open `http://127.0.0.1:8000/docs` to test the `/predict` endpoint interactively

## Example request

```json
{
  "OverallQual": 7,
  "GrLivArea": 1800,
  "GarageCars": 2,
  "GarageArea": 500,
  "TotalBsmtSF": 1000,
  "FirstFlrSF": 1200,
  "FullBath": 2,
  "YearBuilt": 2005
}
```

Returns:
```json
{
  "predicted_price": 221703.34
}
```

## What I'd improve next
- Add proper preprocessing pipeline for the full feature set (currently 
  simplified to 8 numeric features for easier serving)
- Containerize with Docker
- Deploy to a public URL (Render/Railway)