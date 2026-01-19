import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import requests
from io import BytesIO
import pickle
from inference import recursive_forecast_gb, SELECTED_FEATURES


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
        st.markdown(
            """
        <div style="background-color: #f1f8e9; padding: 2px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4CAF50;">
            <h4 style="color: #2c3e50; margin-top: 0;">Region</h4>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Select region and district
        df.rename(columns={'adm1_name': 'region', 'mkt_name': 'district'}, inplace=True)
        df.drop(columns=['adm2_name'], inplace=True)

        reg_col1, reg_col2 = st.columns(2)
        with reg_col1:
            region = st.selectbox("Region", df["region"].unique())
        with reg_col2:
            district = st.selectbox("District", df[df["region"] == region]["district"].unique())


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
            year = st.number_input("Year", min_value=2011, max_value=2030, value=2025)
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

        cols = [
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

        historical_values = {}

        for col in cols:
            if col not in df.columns or not df[col].notna().any():
                historical_values[col] = 0.0
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                historical_values[col] = float(df[col].median())

            else:
                historical_values[col] = df[col].mode().iloc[0]

        exo_col1, exo_col2 = st.columns(2)

        with exo_col1:
            market_price_maize = st.number_input(
                "Market Price Maize",
                min_value=0.0,
                value=historical_values["market_price_maize"],
            )
            market_price_rice = st.number_input(
                "Market Price Rice",
                min_value=0.0,
                value=historical_values["market_price_rice"],
            )
            market_price_sorghum = st.number_input(
                "Market Price Sorghum",
                min_value=0.0,
                value=historical_values["market_price_sorghum"],
            )
            market_price_oil = st.number_input(
                "Market Price Oil",
                min_value=0.0,
                value=historical_values["market_price_oil"],
            )
            population = st.number_input(
                "Population",
                min_value=0.0,
                value=historical_values["population"],
            )

        with exo_col2:
            exchange_rate_typical = st.number_input(
                "Exchange Rate Typical",
                min_value=0.0,
                value=historical_values["exchange_rate_typical"],
            )
            food_price_critical = st.number_input(
                "Food Price Critical",
                min_value=0.0,
                value=historical_values["food_price_critical"],
            )
            cpi_communication = st.number_input(
                "CPI Communication",
                min_value=0.0,
                value=historical_values["cpi_communication"],
            )
            cpi_housing_utilities = st.number_input(
                "CPI Housing Utilities",
                min_value=0.0,
                value=historical_values["cpi_housing_utilities"],
            )

        st.markdown("</div>", unsafe_allow_html=True)  # Close the card container

        # Show prediction on button click
        predict_button = st.button("Predict", use_container_width=True)

        if predict_button:
            df["Date"] = pd.to_datetime(df["Date"])

            try:
                if df.empty:
                    st.error("Historical data is required for recursive forecasting.")
                    return

                history_df = (
                    df[
                        (df["region"] == region) &
                        (df["district"] == district)
                    ][["Date", "food_price_index"]]
                    .dropna()
                    .sort_values("Date")
                    .copy()
                )


                last_hist_date = history_df["Date"].max()
                target_date = pd.to_datetime(f"{year}-{month:02d}-01")

                if target_date <= last_hist_date:
                    st.error("Selected date must be after the last historical observation.")
                    return

                start_date = last_hist_date + pd.DateOffset(months=1)

                exogenous_inputs = {
                    "region": region,
                    "district": district,
                    "market_price_maize": market_price_maize,
                    "market_price_rice": market_price_rice,
                    "market_price_sorghum": market_price_sorghum,
                    "market_price_oil": market_price_oil,
                    "population": population,
                    "exchange_rate_typical": exchange_rate_typical,
                    "food_price_critical": food_price_critical,
                    "cpi_communication": cpi_communication,
                    "cpi_housing_utilities": cpi_housing_utilities,
                }

                forecast_df = recursive_forecast_gb(
                    history_df=history_df,
                    exogenous_inputs=exogenous_inputs,
                    start_date=start_date,
                    target_date=target_date
                )

                forecast_df["region"] = region
                forecast_df["district"] = district

                final_prediction = forecast_df.iloc[-1]["food_price_index"]

                st.markdown(
                    f"""
                    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;">
                        <h2>Predicted Food Price Index</h2>
                        <p style="font-size: 32px; font-weight: bold; color: #2e7d32;">
                            {final_prediction:.2f}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                plot_df = pd.concat(
                    [
                        history_df.tail(12).assign(type="Historical"),
                        forecast_df,
                    ],
                    ignore_index=True,
                )

                fig = px.line(
                    plot_df,
                    x="Date",
                    y="food_price_index",
                    color="type",
                    markers=True,
                    title="Historical Food Price Index and Recursive Forecast",
                    labels={"food_price_index": "Food Price Index", "Date": "Date"},
                )

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Food Price Index",
                    legend_title="Series",
                    plot_bgcolor="rgba(255,255,255,0.95)",
                    paper_bgcolor="rgba(255,255,255,0)",
                    font=dict(color="#2c3e50"),
                    title_font=dict(size=20),
                    xaxis=dict(showgrid=True, gridcolor="#eee"),
                    yaxis=dict(showgrid=True, gridcolor="#eee"),
                )

                fig.update_traces(
                    line=dict(width=3),
                    selector=dict(name="Historical"),
                )

                fig.update_traces(
                    line=dict(width=3, dash="dot"),
                    marker=dict(size=9, symbol="diamond"),
                    selector=dict(name="Predicted"),
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
