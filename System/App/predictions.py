# predictions_hybrid_residual.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from inference import recursive_forecast_hybrid, SELECTED_FEATURES

def show_predictions_page(df):
    st.markdown(
        """
    <div style="background: linear-gradient(to right, #00b09b, #96c93d); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center;"> Food Security Predictor</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create card-like container for the subheader
    st.markdown(
        """
    <div style="background-color: white; padding: 2px; border-radius: 5px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 10px;">
        <h3 style="color: #2c3e50; text-align: center;">Enter Details Below to Predict Food Price Index</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, content_col, col2 = st.columns([0.05, 0.9, 0.05])

    with content_col:        
        exo_col1, exo_col2 = st.columns(2)
        
        with exo_col1:
            region = st.selectbox("Region", df["region"].unique())
            month = st.number_input("Month", min_value=1, max_value=12, value=4)
            market_price_maize = st.number_input("Market Price Maize", value=float(df["market_price_maize"].median()))
            market_price_rice = st.number_input("Market Price Rice", value=float(df["market_price_rice"].median()))
            market_price_sorghum = st.number_input("Market Price Sorghum", value=float(df["market_price_sorghum"].median()))
            market_price_oil = st.number_input("Market Price Oil", value=float(df["market_price_oil"].median()))
            population = st.number_input("Population", value=float(df["population"].median()))
        
        with exo_col2:
            district = st.selectbox("District", df[df["region"] == region]["district"].unique())
            year = st.number_input("Year", min_value=2011, max_value=2030, value=2025)
            exchange_rate_typical = st.number_input("Exchange Rate Typical", value=float(df["exchange_rate_typical"].median()))
            food_price_critical = st.number_input("Food Price Critical", value=float(df["food_price_critical"].median()))
            cpi_communication = st.number_input("CPI Communication", value=float(df["cpi_communication"].median()))
            cpi_housing_utilities = st.number_input("CPI Housing Utilities", value=float(df["cpi_housing_utilities"].median()))

        if st.button("Predict", use_container_width=True):
            try:
                history_df = df[(df["region"] == region) & (df["district"] == district)][["Date", "food_price_index"]].dropna().sort_values("Date")
                
                last_hist_date = pd.to_datetime(history_df["Date"]).max()
                target_date = pd.to_datetime(f"{year}-{month:02d}-01")

                if target_date <= last_hist_date:
                    st.error("Selected date must be after the last historical observation.")
                    return

                exogenous_inputs = {
                    "region": region, "district": district,
                    "market_price_maize": market_price_maize, "market_price_rice": market_price_rice,
                    "market_price_sorghum": market_price_sorghum, "market_price_oil": market_price_oil,
                    "population": population, "exchange_rate_typical": exchange_rate_typical,
                    "food_price_critical": food_price_critical, "cpi_communication": cpi_communication,
                    "cpi_housing_utilities": cpi_housing_utilities
                }

                forecast_df = recursive_forecast_hybrid(
                    history_df=history_df,
                    exogenous_inputs=exogenous_inputs,
                    start_date=last_hist_date + pd.DateOffset(months=1),
                    target_date=target_date
                )

                forecast_df["region"] = region
                forecast_df["district"] = district

                final_prediction = forecast_df.iloc[-1]["food_price_index"]

                st.markdown(
                    f"""
                    <div style="background-color: #e8f5e9; padding: 10px; border-radius: 10px;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;">
                        <h2>Prediction</h2>
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
                    title="Food Price Index Forecast",
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
