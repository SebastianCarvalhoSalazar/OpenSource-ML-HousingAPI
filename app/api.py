"""
FastAPI REST API for Boston Housing price prediction.
Updated to work with Pipeline objects and Boston Housing Dataset.
"""
import os
import sys
import joblib
import pandas as pd
from typing import List, Union
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))

BASE_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
PRODUCTION_MODEL_PATH = os.getenv(
    'PRODUCTION_MODEL_PATH', os.path.join(BASE_DIR, 'models/production.pkl')
)

# Create FastAPI app
app = FastAPI(
    title="Boston Housing Price Prediction API",
    description="API for predicting house prices using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
pipeline = None
model_metadata = None


# Pydantic models for BOSTON HOUSING DATASET
class HousingFeatures(BaseModel):
    """Input features for Boston house price prediction."""
    CRIM: Union[float, int] = Field(...,
                                    description="Per capita crime rate by town")
    ZN: Union[float, int] = Field(
        ..., description="Proportion of residential land zoned for lots over 25,000 sq.ft.")
    INDUS: Union[float, int] = Field(
        ..., description="Proportion of non-retail business acres per town")
    CHAS: Union[int, float] = Field(
        ..., description="Charles River dummy variable (1 if tract bounds river; 0 otherwise)")
    NOX: Union[float, int] = Field(
        ..., description="Nitric oxides concentration (parts per 10 million)")
    RM: Union[float, int] = Field(...,
                                  description="Average number of rooms per dwelling")
    AGE: Union[float, int] = Field(
        ..., description="Proportion of owner-occupied units built prior to 1940")
    DIS: Union[float, int] = Field(
        ..., description="Weighted distances to five Boston employment centres")
    RAD: Union[int, float] = Field(...,
                                   description="Index of accessibility to radial highways")
    TAX: Union[float, int] = Field(...,
                                   description="Full-value property-tax rate per $10,000")
    PTRATIO: Union[float,
                   int] = Field(..., description="Pupil-teacher ratio by town")
    B: Union[float, int] = Field(
        ..., description="1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town")
    LSTAT: Union[float,
                 int] = Field(..., description="Percentage lower status of the population")

    @field_validator('CRIM')
    def validate_crim(cls, v):
        if v < 0:
            raise ValueError('CRIM must be non-negative')
        return float(v)

    @field_validator('RM')
    def validate_rm(cls, v):
        if v < 1 or v > 15:
            raise ValueError('RM should be between 1 and 15')
        return float(v)

    @field_validator('CHAS')
    def validate_chas(cls, v):
        if v not in [0, 1]:
            raise ValueError('CHAS must be 0 or 1')
        return int(v)

    @field_validator('ZN', 'INDUS', 'NOX', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'RAD')
    def convert_to_float(cls, v):
        return float(v)

    class Config:
        json_schema_extra = {
            "example": {
                "CRIM": 0.00632,
                "ZN": 18.0,
                "INDUS": 2.31,
                "CHAS": 0,
                "NOX": 0.538,
                "RM": 6.575,
                "AGE": 65.2,
                "DIS": 4.0900,
                "RAD": 1,
                "TAX": 296.0,
                "PTRATIO": 15.3,
                "B": 396.90,
                "LSTAT": 4.98
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_price: float = Field(...,
                                   description="Predicted house price in $1000s")
    predicted_price_dollars: float = Field(...,
                                           description="Predicted price in dollars")
    prediction_time: str = Field(..., description="Timestamp of prediction")
    model_version: str = Field(..., description="Model version used")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 24.5,
                "predicted_price_dollars": 24500,
                "prediction_time": "2024-01-20T10:30:00",
                "model_version": "Random Forest v1.0"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    houses: List[HousingFeatures] = Field(...,
                                          description="List of houses to predict")

    @field_validator('houses')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 predictions per batch')
        if len(v) < 1:
            raise ValueError('At least 1 house required')
        return v


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    model_metrics: dict
    timestamp: str


def load_production_model():
    """Load the production model pipeline."""
    global pipeline, model_metadata

    try:
        # Load pipeline
        pipeline = joblib.load(PRODUCTION_MODEL_PATH)
        if pipeline:
            print("✅ Pipeline loaded successfully")

        # Load metadata if available
        metadata_path = PRODUCTION_MODEL_PATH.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            print("✅ Metadata loaded:", model_metadata.get(
                'model_name', 'unknown'))
        else:
            model_metadata = {'model_type': 'unknown', 'model_name': 'unknown'}
            print("⚠️ Metadata not found, using 'unknown'")

        print(f"✓ Pipeline loaded from: {PRODUCTION_MODEL_PATH}")
        if model_metadata and 'test_metrics' in model_metadata:
            metrics = model_metadata['test_metrics']
            print(f"  R² Score: {metrics.get('r2_score', 'N/A'):.4f}")
            print(f"  RMSE: ${metrics.get('rmse', 'N/A'):.2f}k")

    except FileNotFoundError:
        print(f"⚠️  Production model not found at: {PRODUCTION_MODEL_PATH}")
        print("Please run model training and comparison first:")
        print("  1. python src/train_baseline.py")
        print("  2. python src/train_advanced.py")
        print("  3. python src/model_comparator.py")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()


# Load model at startup
load_production_model()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Boston Housing Price Prediction API",
        "version": "1.0.0",
        "dataset": "Boston Housing Dataset (13 features)",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs",
            "reload": "/reload-model"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    metrics = model_metadata.get('test_metrics', {}) if model_metadata else {}

    return HealthResponse(
        status="healthy" if pipeline is not None else "unhealthy",
        model_loaded=pipeline is not None,
        model_version=model_metadata.get(
            'model_name', 'unknown') if model_metadata else 'unknown',
        model_metrics=metrics,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HousingFeatures):
    """
    Predict house price for given features.

    Args:
        features: Housing features (Boston Housing Dataset)

    Returns:
        Predicted price in $1000s
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train and load a model first."
        )

    try:
        # Convert input to DataFrame with correct column order
        input_dict = features.dict()
        input_df = pd.DataFrame([input_dict])

        # Ensure column order matches training data
        expected_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        input_df = input_df[expected_columns]

        print("✓ Input DataFrame shape:", input_df.shape)
        print("✓ Input values:", input_df.iloc[0].to_dict())

        # Predict using pipeline (handles preprocessing internally)
        prediction = pipeline.predict(input_df)[0]

        print(f"✓ Prediction: ${prediction:.2f}k")

        # Boston Housing prices are in $1000s
        return PredictionResponse(
            predicted_price=round(float(prediction), 2),
            predicted_price_dollars=round(float(prediction * 1000), 2),
            prediction_time=datetime.now().isoformat(),
            model_version=model_metadata.get(
                'model_name', 'unknown') if model_metadata else 'unknown'
        )

    except Exception as e:
        print("❌ ERROR in prediction:")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict house prices for multiple houses.

    Args:
        request: Batch prediction request

    Returns:
        List of predictions
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )

    try:
        # Convert all inputs to DataFrame
        input_data = [house.dict() for house in request.houses]
        input_df = pd.DataFrame(input_data)

        # Ensure column order matches training data
        expected_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        input_df = input_df[expected_columns]

        print(f"✓ Batch prediction for {len(input_df)} houses")

        # Predict using pipeline
        predictions = pipeline.predict(input_df)

        # Format responses
        results = []
        model_version = model_metadata.get(
            'model_name', 'unknown') if model_metadata else 'unknown'

        for pred in predictions:
            results.append(PredictionResponse(
                predicted_price=round(float(pred), 2),
                predicted_price_dollars=round(float(pred * 1000), 2),
                prediction_time=datetime.now().isoformat(),
                model_version=model_version
            ))

        print(f"✓ Batch prediction completed: {len(results)} results")
        return results

    except Exception as e:
        print("❌ ERROR in batch prediction:")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn

    print(f" Starting API server on {API_HOST}:{API_PORT}")
    print(f" Documentation available at: http://{API_HOST}:{API_PORT}/docs")
    print(f" Health check at: http://{API_HOST}:{API_PORT}/health")

    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )

# curl -X POST "http://localhost:8000/predict" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "CRIM": 0.00632,
#     "ZN": 18.0,
#     "INDUS": 2.31,
#     "CHAS": 0,
#     "NOX": 0.538,
#     "RM": 6.575,
#     "AGE": 65.2,
#     "DIS": 4.0900,
#     "RAD": 1,
#     "TAX": 296.0,
#     "PTRATIO": 15.3,
#     "B": 396.90,
#     "LSTAT": 4.98
#   }'
