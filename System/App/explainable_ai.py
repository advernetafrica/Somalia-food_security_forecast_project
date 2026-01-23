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
from inference import BASE_DIR

MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "Models")

def inverse_boxcox(y, lambda_val):
    if lambda_val == 0:
        return np.exp(y)
    return np.power(lambda_val * y + 1, 1 / lambda_val)

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
            # Load model and preprocessing objects
            # Using relative paths as in the original script
            model_path = os.path.join(MODEL_DIR, "best_gb_model.pkl")
            feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")
            scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
            lambda_boxcox_path = os.path.join(MODEL_DIR, "lambda_boxcox.pkl")

            model = joblib.load(model_path)
            feature_names = joblib.load(feature_names_path)
            scaler = joblib.load(scaler_path)
            lambda_boxcox = joblib.load(lambda_boxcox_path)

            tab1, tab2 = st.tabs(
                ["SHAP Analysis", "What-If Analysis"]
            )

            with tab1:
                show_shap_analysis(model, scaler, feature_names, lambda_boxcox, df)

            with tab2:
                show_what_if_analysis(model, scaler, feature_names, lambda_boxcox, df)

        except Exception as e:
            st.error(f"Error loading model files: {str(e)}")
            st.info("Please ensure 'best_gb_model.pkl', 'feature_names.pkl', 'scaler.pkl', and 'lambda_boxcox.pkl' are in the '../Models/' directory.")

def show_historical_trends(df):
    """Display historical trends for food price index"""
    st.markdown(
        """
    <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
        <h3 style="color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">Historical Trends</h3>
        <p>View the historical food price index trends.</p>
    """,
        unsafe_allow_html=True,
    )

    if df.empty or "food_price_index" not in df.columns or "Date" not in df.columns:
        st.error("No data available or missing required columns.")
        return

    hist_df = df.sort_values("Date")

    fig = px.line(
        hist_df,
        x="Date",
        y="food_price_index",
        markers=True,
        title="Historical Trend for Food Price Index",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Food Price Index")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def show_shap_analysis(model, scaler, feature_names, lambda_boxcox, df):
    """Display SHAP values for model explanation"""
    st.markdown(
        """
    <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
        <h3 style="color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">SHAP Value Analysis</h3>
        <p>SHAP (SHapley Additive exPlanations) values help us understand how each feature contributes to predictions.</p>
    """,
        unsafe_allow_html=True,
    )

    try:
        st.subheader("Select parameters for prediction:")
        
        # Time features
        time_col1, time_col2 = st.columns(2)
        with time_col1:
            month = st.number_input("Month", min_value=1, max_value=12, value=4, key="shap_month")
        with time_col2:
            year = st.number_input("Year", min_value=2011, max_value=2030, value=2024, key="shap_year")
        
        quarter = (month - 1) // 3 + 1
        
        # Create a full feature set initialized with medians
        X_full = pd.DataFrame(columns=scaler.feature_names_in_)
        X_full.loc[0] = 0.0
        for col in X_full.columns:
            if col in df.columns and df[col].notna().any():
                X_full.loc[0, col] = df[col].median()
        
        # Update with UI values
        if "month" in X_full.columns: X_full.loc[0, "month"] = month
        if "year" in X_full.columns: X_full.loc[0, "year"] = year
        if "quarter" in X_full.columns: X_full.loc[0, "quarter"] = quarter
        
        # Scale and select features
        X_scaled = pd.DataFrame(scaler.transform(X_full), columns=scaler.feature_names_in_)
        X_selected = X_scaled[feature_names]
        
        # Predict
        raw_prediction = model.predict(X_selected)[0]
        prediction = inverse_boxcox(raw_prediction, lambda_boxcox)
        
        st.markdown(
            f"""
        <div style="background-color: #e8f5e9; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4CAF50;">
            <p style="margin: 0;"><strong>Predicted Food Price Index:</strong> {prediction:.4f}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # SHAP calculation
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_selected)
            
            # Waterfall plot for the single prediction
            fig, ax = plt.subplots(figsize=(10, 6))
            # For TreeExplainer on a single row, shap_values is a 2D array (1, n_features)
            # We need to create an Explanation object for the waterfall plot
            exp = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_selected.iloc[0].values,
                feature_names=feature_names
            )
            shap.plots.waterfall(exp, max_display=10, show=False)
            plt.title("SHAP Waterfall Plot - Feature Contributions")
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(
                """
            <div style="background-color: #f5f5f5; padding: 10px; border-radius: 8px; margin: 10px 0;">
                <h4 style="color: #2c3e50; margin-top: 0;">How to Interpret:</h4>
                <ul>
                    <li><strong>Base Value:</strong> The average prediction across the training set.</li>
                    <li><strong>Red bars:</strong> Features that increase the price index from the base value.</li>
                    <li><strong>Blue bars:</strong> Features that decrease the price index from the base value.</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Error in SHAP analysis: {str(e)}")
        st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)

def show_what_if_analysis(model, scaler, feature_names, lambda_boxcox, df):
    """Interactive what-if analysis"""
    st.markdown(
        """
    <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
        <h3 style="color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">What-If Analysis</h3>
        <p>Experiment with different input values to see how they affect predictions.</p>
    """,
        unsafe_allow_html=True,
    )

    try:
        st.subheader("Adjust Features")
        
        # Select a few key features for the what-if analysis
        # We'll pick features that are actually in the model
        available_features = [f for f in ["market_price_maize", "market_price_rice", "market_price_sorghum", "market_price_oil", "exchange_rate", "population"] if f in feature_names]
        
        if not available_features:
            available_features = feature_names[:5] # Fallback to first 5 features
            
        col1, col2 = st.columns(2)
        ui_inputs = {}
        
        for i, feat in enumerate(available_features):
            with col1 if i % 2 == 0 else col2:
                median_val = float(df[feat].median()) if feat in df.columns else 0.0
                ui_inputs[feat] = st.slider(
                    f"{feat.replace('_', ' ').title()}",
                    min_value=0.0,
                    max_value=median_val * 3 if median_val > 0 else 100.0,
                    value=median_val,
                    key=f"whatif_{feat}"
                )

        # Create full feature set
        X_full = pd.DataFrame(columns=scaler.feature_names_in_)
        X_full.loc[0] = 0.0
        for col in X_full.columns:
            if col in df.columns and df[col].notna().any():
                X_full.loc[0, col] = df[col].median()
        
        # Update with UI values
        for feat, val in ui_inputs.items():
            X_full.loc[0, feat] = val
            
        # Scale and select
        X_scaled = pd.DataFrame(scaler.transform(X_full), columns=scaler.feature_names_in_)
        X_selected = X_scaled[feature_names]
        
        # Predict
        raw_prediction = model.predict(X_selected)[0]
        prediction = inverse_boxcox(raw_prediction, lambda_boxcox)

        st.markdown(
            f"""
        <div style="background-color: #e8f5e9; padding: 5px; border-radius: 10px; margin: 5px 0; text-align: center; border-left: 4px solid #4CAF50;">
            <h2 style="margin: 0; color: #2c3e50;">Predicted Food Price Index</h2>
            <p style="font-size: 36px; font-weight: bold; color: #4CAF50; margin: 10px 0;">{prediction:.4f}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Sensitivity Analysis
        st.subheader("Sensitivity Analysis")
        sens_feat = st.selectbox("Select feature for sensitivity analysis:", available_features)
        
        base_val = ui_inputs[sens_feat]
        vals = np.linspace(0, base_val * 2 if base_val > 0 else 100, 20)
        preds = []
        
        for v in vals:
            X_temp = X_full.copy()
            X_temp.loc[0, sens_feat] = v
            X_temp_scaled = pd.DataFrame(scaler.transform(X_temp), columns=scaler.feature_names_in_)
            X_temp_selected = X_temp_scaled[feature_names]
            p_raw = model.predict(X_temp_selected)[0]
            preds.append(inverse_boxcox(p_raw, lambda_boxcox))
            
        sens_df = pd.DataFrame({"Value": vals, "Prediction": preds})
        fig = px.line(sens_df, x="Value", y="Prediction", title=f"Sensitivity to {sens_feat}")
        fig.add_vline(x=base_val, line_dash="dash", line_color="red", annotation_text="Current Value")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in what-if analysis: {str(e)}")
        st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)
