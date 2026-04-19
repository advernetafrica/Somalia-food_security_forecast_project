# inference.py — synthetic simulation mode (no TF / no Keras).
# Used on the `simulation` branch so the app stays responsive.
# The real hybrid LSTM+GRU inference lives on the main branch.

import os
import hashlib

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "Models")

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

# Only lightweight encoders are loaded — used for dropdowns / class lists.
region_le = joblib.load(REGION_ENCODER_PATH)
district_le = joblib.load(DISTRICT_ENCODER_PATH)


# Dataset anchors (medians) — FPI sits in ~[0.5, 3.0], so predictions
# must stay on that scale. Shocks are expressed as % deviations.
_BASKET_MEDIAN = 84_844.0          # maize + rice + sorghum + oil
_CPI_BUNDLE_MEDIAN = 28.028        # cpi_communication + cpi_housing_utilities
_CRITICAL_ANCHOR = 30.0            # moderate stress benchmark

# Sensitivity of the FPI to each driver's relative shock.
_SHOCK_GAIN = {
    "price_basket": 0.18,
    "cpi_bundle": 0.10,
    "food_price_critical": 0.06,
}

_FPI_FLOOR = 0.4
_FPI_CEIL = 3.5


def _deterministic_noise(seed_str: str, scale: float = 0.025) -> float:
    """Small reproducible jitter derived from region+district+date hash."""
    h = hashlib.md5(seed_str.encode("utf-8")).hexdigest()
    val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0  # → [-1, 1]
    return val * scale


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """Kept for API compatibility — returns feature matrix as numpy array."""
    return df[SELECTED_FEATURES].to_numpy()


def postprocess_prediction(y_hat: float) -> float:
    return float(y_hat)


def predict_hybrid_residual(input_df: pd.DataFrame, seed: str = "") -> float:
    """
    Synthetic predictor calibrated to the real FPI scale (~0.5–3.0).

    Anchors the prediction to `food_price_index_rolling_mean_3` (i.e. the
    recent historical FPI) and applies proportional shocks from the major
    drivers plus a small deterministic jitter for realism.
    """
    row = input_df.iloc[0]

    basket = (
        row["market_price_maize"]
        + row["market_price_rice"]
        + row["market_price_sorghum"]
        + row["market_price_oil"]
    )
    cpi_bundle = row["cpi_communication"] + row["cpi_housing_utilities"]

    price_shock = (basket / _BASKET_MEDIAN) - 1.0
    cpi_shock = (cpi_bundle / _CPI_BUNDLE_MEDIAN) - 1.0
    critical_shock = row["food_price_critical"] / _CRITICAL_ANCHOR

    base = float(row["food_price_index_rolling_mean_3"])
    multiplier = (
        1.0
        + _SHOCK_GAIN["price_basket"] * price_shock
        + _SHOCK_GAIN["cpi_bundle"] * cpi_shock
        + _SHOCK_GAIN["food_price_critical"] * critical_shock
    )

    pred = base * multiplier

    seed_str = seed or f"{row['region']}|{row['district']}"
    pred += _deterministic_noise(seed_str, scale=0.025)

    pred = max(_FPI_FLOOR, min(_FPI_CEIL, pred))
    return postprocess_prediction(pred)


def recursive_forecast_hybrid(
    history_df: pd.DataFrame,
    exogenous_inputs: dict,
    start_date: pd.Timestamp,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Iteratively generate FPI forecasts, month-by-month, up to target_date."""
    df_extended = history_df.copy()
    predictions = []

    current_date = start_date
    while current_date <= target_date:
        rolling_mean_3 = df_extended["food_price_index"].tail(3).mean()

        model_input = pd.DataFrame([{
            **exogenous_inputs,
            "food_price_index_rolling_mean_3": rolling_mean_3,
        }])

        seed = f"{exogenous_inputs['region']}|{exogenous_inputs['district']}|{current_date.date()}"
        y_hat = predict_hybrid_residual(model_input, seed=seed)

        predictions.append({
            "Date": current_date,
            "food_price_index": y_hat,
            "type": "Predicted",
        })

        df_extended = pd.concat([
            df_extended,
            pd.DataFrame({"Date": [current_date], "food_price_index": [y_hat]}),
        ], ignore_index=True)

        current_date += pd.DateOffset(months=1)

    return pd.DataFrame(predictions)
