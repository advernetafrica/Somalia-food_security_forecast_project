import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import requests
from io import BytesIO
import pickle


def download_file(url, filename):
    """
    Download a file from the given URL and save it locally.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename
    else:
        st.error(
            f"Failed to download {filename} from {url}. HTTP Status code: {response.status_code}"
        )
        return None


def show_predictions_page(df):
    """
    Display the predictions page with model-based forecasting.

    Parameters:
    df (pandas.DataFrame): The dataset to use for predictions
    """
    # Apply custom header with gradient background
    st.markdown(
        """
    <div style="background: linear-gradient(to right, #00b09b, #96c93d); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center;"> Somalia Food Security Predictor</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create card-like container for the subheader
    st.markdown(
        """
    <div style="background-color: white; padding: 2px; border-radius: 5px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 10px;">
        <h3 style="color: #2c3e50; text-align: center;">Enter Details Below to Predict Food Security Indicators</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create a centered container with reduced padding for all predictions content
    col1, content_col, col2 = st.columns([0.05, 0.9, 0.05])

    with content_col:
        # Load the model locally
        import pickle

        try:
            model = pickle.load(open("../Models/best_gb_model.pkl", "rb"))
            feature_names = pickle.load(open("../Models/feature_names.pkl", "rb"))
            scaler = pickle.load(open("../Models/scaler.pkl", "rb"))
            lambda_boxcox = pickle.load(open("../Models/lambda_boxcox.pkl", "rb"))
            # encoder = pickle.load(open("../Models/encoder.pkl", 'rb'))
        except FileNotFoundError:
            st.error(
                "Model files not found. Please ensure the models are saved in ../Models/"
            )
            return

        # Row 2: Temporal features
        st.markdown(
            """
        <div style="background-color: #f1f8e9; padding: 2px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4CAF50;">
            <h4 style="color: #2c3e50; margin-top: 0;">Time Period Selection</h4>
        </div>
        """,
            unsafe_allow_html=True,
        )

        time_col1, time_col2, time_col3 = st.columns(3)
        with time_col1:
            month = st.number_input("Month", min_value=1, max_value=12, value=4)
        with time_col2:
            year = st.number_input("Year", min_value=2011, max_value=2030, value=2024)
        with time_col3:
            quarter = (month - 1) // 3 + 1
            st.markdown(
                f"""
            <div style="background-color: white; padding: 2px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-top: 23px;">
                <p style="font-weight: bold; margin: 0;">Quarter: Q{quarter}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
        <div style="background-color: #e3f2fd; padding: 2px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #2196F3;">
            <h4 style="color: #2c3e50; margin-top: 0;"> Features</h4>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Compute historical medians for exogenous variables
        historical_medians = {}

        cols = [
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

        for col in cols:
            if col in df.columns and df[col].notna().any():
                historical_medians[col] = float(df[col].median())
            else:
                historical_medians[col] = 0.0  # safe fallback

        # Exogenous features inputs
        exo_col1, exo_col2 = st.columns(2)

        with exo_col1:
            market_price_maize = st.number_input(
                "Market Price Maize",
                min_value=0.0,
                value=historical_medians["market_price_maize"],
            )
            market_price_rice = st.number_input(
                "Market Price Rice",
                min_value=0.0,
                value=historical_medians["market_price_rice"],
            )
            market_price_sorghum = st.number_input(
                "Market Price Sorghum",
                min_value=0.0,
                value=historical_medians["market_price_sorghum"],
            )
            market_price_oil = st.number_input(
                "Market Price Oil",
                min_value=0.0,
                value=historical_medians["market_price_oil"],
            )
            population = st.number_input(
                "Population",
                min_value=0.0,
                value=historical_medians["population"],
            )

        with exo_col2:
            exchange_rate_typical = st.number_input(
                "Exchange Rate Typical",
                min_value=0.0,
                value=historical_medians["exchange_rate_typical"],
            )
            food_price_critical = st.number_input(
                "Food Price Critical",
                min_value=0.0,
                value=historical_medians["food_price_critical"],
            )
            cpi_communication = st.number_input(
                "CPI Communication",
                min_value=0.0,
                value=historical_medians["cpi_communication"],
            )
            cpi_housing_utilities = st.number_input(
                "CPI Housing Utilities",
                min_value=0.0,
                value=historical_medians["cpi_housing_utilities"],
            )
            food_price_index_rolling_mean_3 = st.number_input(
                "Food Price Index Rolling Mean 3",
                min_value=0.0,
                value=historical_medians["food_price_index_rolling_mean_3"],
            )

        # Create input data for prediction
        input_data = pd.DataFrame(
            [
                {
                    "month": month,
                    "year": year,
                    "quarter": quarter,
                    "market_price_maize": market_price_maize,
                    "market_price_rice": market_price_rice,
                    "market_price_sorghum": market_price_sorghum,
                    "market_price_oil": market_price_oil,
                    "population": population,
                    "exchange_rate_typical": exchange_rate_typical,
                    "food_price_critical": food_price_critical,
                    "cpi_communication": cpi_communication,
                    "cpi_housing_utilities": cpi_housing_utilities,
                    "food_price_index_rolling_mean_3": food_price_index_rolling_mean_3,
                }
            ]
        )

        st.markdown("</div>", unsafe_allow_html=True)  # Close the card container

        # Show prediction on button click
        predict_button = st.button("Predict", use_container_width=True)

        def inverse_boxcox(y, lambda_val):
            if lambda_val == 0:
                return np.exp(y)
            return np.power(lambda_val * y + 1, 1 / lambda_val)

        if predict_button:
            X = pd.DataFrame(columns=scaler.feature_names_in_)
            X.loc[0] = np.nan  # initialize all features

            # 2. Fill values coming from UI
            for col in input_data.columns:
                if col in X.columns:
                    X.loc[0, col] = input_data.loc[0, col]

            # 3. Fill remaining features (climate, conflict, etc.) from historical medians
            for col in X.columns:
                if pd.isna(X.loc[0, col]):
                    if col in df.columns and df[col].notna().any():
                        X.loc[0, col] = df[col].median()
                    else:
                        raise ValueError(f"Missing required feature: {col}")


            # 4. Scale all features
            X_scaled = scaler.transform(X)

            # 5. Select the features used by the model
            X_scaled_df = pd.DataFrame(X_scaled, columns=scaler.feature_names_in_)
            X_selected = X_scaled_df[feature_names]

            # 6. Predict
            raw_prediction = model.predict(X_selected)[0]
            prediction = inverse_boxcox(raw_prediction, lambda_boxcox)

            st.markdown(
                f"""
            <div style="background-color: #e8f5e9; padding: 2px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin: 5px 0; text-align: center; border-left: 4px solid #4CAF50;">
                <span style="font-size: 24px;"> </span>
                <h2 style="margin: 10px 0; color: #2c3e50;">Predicted Food Price Index</h2>
                <p style="font-size: 32px; font-weight: bold; color: #4CAF50; margin: 10px 0;">{prediction:.2f} </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Show visualization of recent history and prediction
            if not df.empty:
                st.markdown(
                    """
                <div style="background-color: white; padding: 2px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-top: 5px;">
                    <h3 style="color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">Historical Data and Prediction</h3>
                """,
                    unsafe_allow_html=True,
                )

                # Create a combined visualization
                recent_df = df.tail(12).copy()

                # Create a prediction point
                prediction_date = pd.to_datetime(f"{year}-{month:02d}-01")
                prediction_df = pd.DataFrame(
                    {
                        "Date": [prediction_date],
                        "food_price_index": [prediction],
                        "type": ["Prediction"],
                    }
                )

                # Add type column to historical data
                recent_df["type"] = "Historical"

                # Combine data for visualization
                plot_df = pd.concat(
                    [recent_df[["Date", "food_price_index", "type"]], prediction_df]
                )

                # Create plot
                fig = px.line(
                    plot_df,
                    x="Date",
                    y="food_price_index",
                    color="type",
                    markers=True,
                    title="Historical Food Price Index and Prediction",
                    labels={"food_price_index": "Food Price Index", "Date": "Date"},
                )

                # Customize the plot
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Food Price Index",
                    legend_title="Data Type",
                    plot_bgcolor="rgba(255,255,255,0.9)",
                    paper_bgcolor="rgba(255,255,255,0)",
                    font=dict(color="#2c3e50"),
                    title_font=dict(size=20, color="#2c3e50"),
                    xaxis=dict(showgrid=True, gridcolor="#eee"),
                    yaxis=dict(showgrid=True, gridcolor="#eee"),
                )

                # Color customization
                fig.update_traces(line=dict(width=3), selector=dict(name="Historical"))
                fig.update_traces(
                    line=dict(width=4, dash="dot"),
                    marker=dict(size=12, symbol="diamond"),
                    selector=dict(name="Prediction"),
                )

                # Make plotly chart use the full width
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    "</div>", unsafe_allow_html=True
                )  # Close the card container
