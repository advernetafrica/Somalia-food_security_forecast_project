# inference.py
import pandas as pd
import numpy as np
import joblib
from scipy.special import inv_boxcox
from tensorflow.keras.models import load_model
import os

MODEL_DIR = "../Models"
GB_MODEL_PATH = os.path.join(MODEL_DIR, "best_gb_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LAMBDA_PATH = os.path.join(MODEL_DIR, "lambda_boxcox.pkl")
REGION_ENCODER_PATH = os.path.join(MODEL_DIR, "region_encoder.pkl")
DISTRICT_ENCODER_PATH = os.path.join(MODEL_DIR, "district_encoder.pkl")

TIMESTEPS = 12

SELECTED_FEATURES = [
    "region",
    "district",
    "market_price_maize",
    "market_price_rice",
    "market_price_sorghum",
    "market_price_oil",
    "population",
    "exchange_rate_typical",
    "food_price_critical",
    "cpi_communication",
    "cpi_housing_utilities",
    "food_price_index_rolling_mean_3",
]


TARGET_COLUMN = "food_price_index"

scaler = joblib.load(SCALER_PATH)
lambda_boxcox = joblib.load(LAMBDA_PATH)
gb_model = joblib.load(GB_MODEL_PATH)

region_le = joblib.load(REGION_ENCODER_PATH)
district_le = joblib.load(DISTRICT_ENCODER_PATH)

def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()

    df["region"] = region_le.transform(df["region"])
    df["district"] = district_le.transform(df["district"])

    if "food_price_index_rolling_mean_3" not in df.columns:
        if TARGET_COLUMN in df.columns:
            df["food_price_index_rolling_mean_3"] = (
                df[TARGET_COLUMN].rolling(3).mean()
            )
        else:
            raise ValueError("Rolling mean cannot be computed")

    X = df[SELECTED_FEATURES]

    return scaler.transform(X)


def postprocess_prediction(y_hat: float) -> float:
    return inv_boxcox(y_hat, lambda_boxcox)


def predict_gb(input_df: pd.DataFrame) -> float:
    X_scaled = preprocess_data(input_df)
    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=SELECTED_FEATURES
    )
    y_hat = gb_model.predict(X_scaled_df.tail(1))[0]
    return postprocess_prediction(y_hat)


def recursive_forecast_gb(
    history_df: pd.DataFrame,
    exogenous_inputs: dict,
    start_date: pd.Timestamp,
    target_date: pd.Timestamp
) -> pd.DataFrame:

    df_extended = history_df.copy()
    predictions = []

    if exogenous_inputs["region"] not in region_le.classes_:
        raise ValueError(f"Unknown region: {exogenous_inputs['region']}")

    if exogenous_inputs["district"] not in district_le.classes_:
        raise ValueError(f"Unknown district: {exogenous_inputs['district']}")

    current_date = start_date

    while current_date <= target_date:

        rolling_mean_3 = df_extended["food_price_index"].tail(3).mean()

        model_input = pd.DataFrame([{
            "region": exogenous_inputs["region"],
            "district": exogenous_inputs["district"],
            "market_price_maize": exogenous_inputs["market_price_maize"],
            "market_price_rice": exogenous_inputs["market_price_rice"],
            "market_price_sorghum": exogenous_inputs["market_price_sorghum"],
            "market_price_oil": exogenous_inputs["market_price_oil"],
            "population": exogenous_inputs["population"],
            "exchange_rate_typical": exogenous_inputs["exchange_rate_typical"],
            "food_price_critical": exogenous_inputs["food_price_critical"],
            "cpi_communication": exogenous_inputs["cpi_communication"],
            "cpi_housing_utilities": exogenous_inputs["cpi_housing_utilities"],
            "food_price_index_rolling_mean_3": rolling_mean_3,
        }])

        y_hat = predict_gb(model_input)

        predictions.append({
            "Date": current_date,
            "food_price_index": y_hat,
            "type": "Predicted"
        })

        df_extended = pd.concat([
            df_extended,
            pd.DataFrame({
                "Date": [current_date],
                "food_price_index": [y_hat]
            })
        ], ignore_index=True)

        current_date += pd.DateOffset(months=1)

    return pd.DataFrame(predictions)
