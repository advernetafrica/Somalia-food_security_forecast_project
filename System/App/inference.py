# inference_hybrid_residual.py
import pandas as pd
import numpy as np
import joblib
import os
from scipy.special import inv_boxcox
from tensorflow.keras.models import load_model

MODEL_DIR = "../Models"
LSTM_BASE_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
GRU_RESIDUAL_PATH = os.path.join(MODEL_DIR, "gru_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LAMBDA_PATH = os.path.join(MODEL_DIR, "lambda_boxcox.pkl")
REGION_ENCODER_PATH = os.path.join(MODEL_DIR, "region_encoder.pkl")
DISTRICT_ENCODER_PATH = os.path.join(MODEL_DIR, "district_encoder.pkl")

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

# Load assets
scaler = joblib.load(SCALER_PATH)
lambda_boxcox = joblib.load(LAMBDA_PATH)
region_le = joblib.load(REGION_ENCODER_PATH)
district_le = joblib.load(DISTRICT_ENCODER_PATH)

# Load the two hybrid components
lstm_base = load_model(LSTM_BASE_PATH, compile=False)
gru_residual = load_model(GRU_RESIDUAL_PATH, compile=False)

def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    df["region"] = region_le.transform(df["region"])
    df["district"] = district_le.transform(df["district"])
    X = df[SELECTED_FEATURES]
    return scaler.transform(X)

def postprocess_prediction(y_hat: float) -> float:
    return inv_boxcox(y_hat, lambda_boxcox)

def predict_hybrid_residual(input_df: pd.DataFrame) -> float:
    """
    Predicts using the LSTM + GRU-Residual hybrid logic.
    """
    X_scaled = preprocess_data(input_df)
    
    # Reshape to (1, 1, num_features) as per your training code
    X_hybrid = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # Base prediction + Residual correction
    lstm_pred = lstm_base.predict(X_hybrid, verbose=0).flatten()[0]
    gru_corr = gru_residual.predict(X_hybrid, verbose=0).flatten()[0]
    
    final_y_hat = lstm_pred + gru_corr
    return postprocess_prediction(final_y_hat)

def recursive_forecast_hybrid(
    history_df: pd.DataFrame,
    exogenous_inputs: dict,
    start_date: pd.Timestamp,
    target_date: pd.Timestamp
) -> pd.DataFrame:
    df_extended = history_df.copy()
    predictions = []
    
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
        
        y_hat = predict_hybrid_residual(model_input)
        
        predictions.append({
            "Date": current_date,
            "food_price_index": y_hat,
            "type": "Predicted"
        })
        
        df_extended = pd.concat([
            df_extended,
            pd.DataFrame({"Date": [current_date], "food_price_index": [y_hat]})
        ], ignore_index=True)
        
        current_date += pd.DateOffset(months=1)
        
    return pd.DataFrame(predictions)
