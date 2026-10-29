
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from scipy import sparse
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# Load the saved deployment artifacts.
model = keras.models.load_model(
    ARTIFACT_DIR / "housing_model.keras"
)

pipeline = joblib.load(
    ARTIFACT_DIR / "preprocessor.joblib"
)

feature_columns = joblib.load(
    ARTIFACT_DIR / "feature_columns.joblib"
)


app = FastAPI(
    title="California Housing Price Prediction API",
    version="1.0.0",
)


class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float = Field(ge=0)
    total_rooms: float = Field(ge=0)
    total_bedrooms: float = Field(ge=0)
    population: float = Field(ge=0)
    households: float = Field(ge=0)
    median_income: float = Field(ge=0)
    ocean_proximity: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HousingFeatures):

    # Create a raw input DataFrame with the expected columns.
    input_data = pd.DataFrame(
        [features.model_dump()]
    )[feature_columns]

    # Apply the fitted preprocessing pipeline.
    processed = pipeline.transformer.transform(
        input_data
    )

    if sparse.issparse(processed):
        processed = processed.toarray()

    processed = np.asarray(processed)

    # Verify compatibility with the saved model.
    if processed.shape[1] != model.input_shape[-1]:
        raise ValueError(
            "Preprocessed feature count does not match "
            "the model input dimension."
        )

    # Predict in scaled target space.
    prediction_scaled = model.predict(
        processed,
        verbose=0,
    )

    # Convert the prediction back to US dollars.
    prediction = np.asarray(
        pipeline.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        )
    ).ravel()[0]

    return {
        "predicted_house_value": round(
            float(prediction), 2
        ),
        "currency": "USD",
        "model": "model_3",
    }
