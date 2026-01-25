import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
import pickle
import os
from inference import predict_hybrid_residual, SELECTED_FEATURES, region_le, district_le


def show_explainable_ai_page(df):
    """
    Display the explainable AI page to help users understand model predictions
    """
    st.markdown(
        """
    <div style="background: linear-gradient(to right, #00b09b, #96c93d); padding: 2px; border-radius: 10px; margin-bottom: 5px;">
        <h1 style="color: white; text-align: center;"> Explainable AI - Food Security Insights</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="background-color: white; padding: 5px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
        <p style="color: #333; text-align: center; font-size: 18px;">Explore how the model predicts food security indicators and which factors influence the results.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, content_col, col2 = st.columns([0.05, 0.9, 0.05])

    with content_col:
        try:
            show_what_if_analysis(df)

        except Exception as e:
            st.error(f"Error loading analysis: {str(e)}")

def show_what_if_analysis(df):
    """
    Interactive what-if analysis for the LSTM + GRU residual hybrid model
    """

    st.markdown(
        """
        <div style="background-color: white; padding: 15px; border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
        <h3 style="color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">
        What-If Analysis
        </h3>
        <p>
        Adjust inputs to explore how the model responds. This analysis shows
        <b>local</b> sensitivity around the current scenario.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox("Region", region_le.classes_, key="whatif_region")
        with col2:
            district = st.selectbox("District", district_le.classes_, key="whatif_district")

        numeric_features = [
            f for f in SELECTED_FEATURES
            if f not in ["region", "district", "food_price_index_rolling_mean_3"]
        ]

        ui_inputs = {}
        col1, col2 = st.columns(2)

        for i, feat in enumerate(numeric_features):
            median_val = float(df[feat].median()) if feat in df.columns else 0.0
            max_val = max(median_val * 3, median_val + 1)

            with col1 if i % 2 == 0 else col2:
                ui_inputs[feat] = st.slider(
                    feat.replace("_", " ").title(),
                    min_value=0.0,
                    max_value=float(max_val),
                    value=float(median_val),
                    step=float(max_val / 100),
                    key=f"whatif_{feat}",
                )

        rolling_mean_3 = df["food_price_index"].tail(3).mean()

        input_df = pd.DataFrame([{
            "region": region,
            "district": district,
            **ui_inputs,
            "food_price_index_rolling_mean_3": rolling_mean_3,
        }])

        prediction = predict_hybrid_residual(input_df)

        st.markdown(
            f"""
            <div style="background-color: #e8f5e9; padding: 5px; border-radius: 5px;
            margin: 5px 0; text-align: center; border-left: 4px solid #4CAF50;">
                <h2 style="margin: 0;">Predicted Food Price Index</h2>
                <p style="font-size: 36px; font-weight: bold; color: #4CAF50;">
                {prediction:.3f}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Feature Sensitivity")

        sens_feat = st.selectbox(
            "Select feature for sensitivity analysis",
            numeric_features,
            key="whatif_sens_feat",
        )

        base_val = ui_inputs[sens_feat]

        if base_val > 0:
            values = np.linspace(base_val * 0.8, base_val * 1.2, 25)
        else:
            values = np.linspace(0, 1, 25)

        preds = []
        for v in values:
            temp_df = input_df.copy()
            temp_df.loc[0, sens_feat] = v
            preds.append(predict_hybrid_residual(temp_df))

        sens_df = pd.DataFrame({
            "Value": values,
            "Prediction": preds,
        })

        fig = px.line(
            sens_df,
            x="Value",
            y="Prediction",
            title=f"Sensitivity to {sens_feat.replace('_', ' ').title()}",
        )
        fig.add_vline(
            x=base_val,
            line_dash="dash",
            annotation_text="Current value",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Local Feature Importance")

        baseline_pred = prediction
        importance_scores = {}

        for feat in numeric_features:
            base_val = ui_inputs[feat]

            if base_val > 0:
                deltas = np.linspace(base_val * 0.9, base_val * 1.1, 5)
            else:
                deltas = np.linspace(0, 1, 5)

            effects = []
            for v in deltas:
                temp_df = input_df.copy()
                temp_df.loc[0, feat] = v
                p = predict_hybrid_residual(temp_df)
                effects.append(abs(p - baseline_pred))

            importance_scores[feat] = np.mean(effects)

        imp_df = (
            pd.DataFrame({
                "Feature": importance_scores.keys(),
                "Importance": importance_scores.values(),
            })
            .sort_values("Importance", ascending=True)
        )

        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Local Feature Importance",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Importance reflects how sensitive the prediction is to small changes "
            "around the current scenario. This is a local, not global, explanation."
        )

    except Exception as e:
        st.error(f"What-if analysis failed: {str(e)}")
        st.exception(e)
