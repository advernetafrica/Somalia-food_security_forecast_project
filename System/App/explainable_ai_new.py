import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
import os


def show_explainable_ai_page(df):
    """
    Display the explainable AI page to help users understand model predictions

    Parameters:
    df (pandas.DataFrame): The dataset to use for predictions and explanations
    """
    # Apply custom header with gradient background
    st.markdown(
        """
    <div style="background: linear-gradient(to right, #00b09b, #96c93d); padding: 2px; border-radius: 10px; margin-bottom: 5px;">
        <h1 style="color: white; text-align: center;"> Explainable AI - Food Security Insights</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create a subtitle
    st.markdown(
        """
    <div style="background-color: white; padding: 5px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
        <p style="color: #333; text-align: center; font-size: 18px;">Explore how the model predicts food security indicators and which factors influence the results.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create a centered container
    col1, content_col, col2 = st.columns([0.05, 0.9, 0.05])

    with content_col:
        # Load model and encoder
        try:
            model_path = "Models/best_gb_model.pkl"
            feature_names_path = "Models/feature_names.pkl"
            encoder_path = "Models/encoder.pkl"

            if not os.path.exists(model_path) or not os.path.exists(feature_names_path) or not os.path.exists(encoder_path):
                st.error(
                    "Model files not found. Please ensure the model is trained and saved."
                )
                return

            # Load the model using joblib
            model = joblib.load(model_path)
            feature_names = joblib.load(feature_names_path)
            encoder = joblib.load(encoder_path)

            # Create tabs for different explanation approaches
            tab1, tab2, tab3 = st.tabs(
                ["Feature Importance", "SHAP Values", "What-If Analysis"]
            )

            with tab1:
                show_feature_importance(model, feature_names)

            with tab2:
                show_shap_analysis(model, feature_names, df)

            with tab3:
                show_what_if_analysis(model, feature_names, df)

        except FileNotFoundError:
            st.error(
                "Model files not found. Please ensure 'best_gb_model.pkl', 'feature_names.pkl', and 'encoder.pkl' are in the Models directory."
            )


def show_feature_importance(model, feature_names):
    """Display global feature importance for the predictive model"""
    st.markdown(
        """
    <div style="background-color: white; padding: 2px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
        <h3 style="color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">Global Feature Importance</h3>
        <p>This chart shows which features have the most influence on the model's predictions overall.</p>
    """,
        unsafe_allow_html=True,
    )

    try:
        # Get feature importances
        importances = model.feature_importances_

        # Create a DataFrame for visualization
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        # Calculate percentage importance
        importance_df["Percentage"] = (
            importance_df["Importance"] / importance_df["Importance"].sum() * 100
        )

        # Create a bar chart with Plotly
        fig = px.bar(
            importance_df,
            x="Percentage",
            y="Feature",
            orientation="h",
            title="Feature Importance (%)",
            labels={"Percentage": "Importance (%)", "Feature": "Feature Name"},
            color="Percentage",
            color_continuous_scale="Viridis",
        )

        # Customize layout
        fig.update_layout(
            plot_bgcolor="rgba(255,255,255,0.9)",
            paper_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#2c3e50"),
            xaxis=dict(showgrid=True, gridcolor="#eee"),
            yaxis=dict(showgrid=False),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add explanation about top features
        top_features = importance_df.head(3)["Feature"].tolist()

        st.markdown(
            f"""
        <div style="background-color: #e3f2fd; padding: 2px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #2196F3;">
            <p style="margin: 0;"><span style="font-size: 20px;"> </span> <strong>Key Insight:</strong> The top {len(top_features)} most influential features are: <strong>{", ".join(top_features)}</strong>. These features have the largest impact on predicted food price index.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error calculating feature importance: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)  # Close the card container


def show_shap_analysis(model, feature_names, df):
    """Display SHAP values for model explanation"""
    st.markdown(
        """
    <div style="background-color: white; padding: 2px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
        <h3 style="color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">SHAP Value Analysis</h3>
        <p>SHAP (SHapley Additive exPlanations) values help us understand how each feature contributes to predictions for individual examples.</p>
    """,
        unsafe_allow_html=True,
    )

    try:
        # Create sample input form
        st.subheader("Enter sample values to explain:")

        col1, col2 = st.columns(2)
        with col1:
            maize_price = st.number_input("Maize Price", min_value=0.0, value=100.0, key="shap_maize")
            rice_price = st.number_input("Rice Price", min_value=0.0, value=150.0, key="shap_rice")
            sorghum_price = st.number_input("Sorghum Price", min_value=0.0, value=120.0, key="shap_sorghum")
            oil_price = st.number_input("Oil Price", min_value=0.0, value=200.0, key="shap_oil")

        with col2:
            exchange_rate = st.number_input("Exchange Rate", min_value=0.0, value=1.0, key="shap_exchange")
            cpi = st.number_input("CPI", min_value=0.0, value=100.0, key="shap_cpi")
            shock = st.number_input("Shock Index", min_value=0.0, value=0.0, key="shap_shock")
            population = st.number_input("Population", min_value=0, value=100000, key="shap_pop")

        # Create sample data
        sample_data = pd.DataFrame(
            [
                {
                    "market_price_maize": maize_price,
                    "market_price_rice": rice_price,
                    "market_price_sorghum": sorghum_price,
                    "market_price_oil": oil_price,
                    "exchange_rate": exchange_rate,
                    "cpi_all_groups": cpi,
                    "conflict_critical": shock,
                    "population": population,
                }
            ]
        )

        # Predict
        prediction = model.predict(sample_data[feature_names])[0]

        # Display prediction
        st.markdown(
            f"""
        <div style="background-color: #e8f5e9; padding: 2px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #4CAF50;">
            <p style="margin: 0;"><span style="font-size: 20px;"> </span> <strong>Predicted Food Price Index:</strong> {prediction:.2f}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Calculate SHAP values
        st.subheader("SHAP Values Explanation")

        with st.spinner("Calculating SHAP values..."):
            # Create a background dataset for SHAP
            # We'll use a small random sample of the data for the background
            background_data = df.sample(min(100, len(df)))

            # Select only the features used by the model
            background_X = background_data[feature_names]

            # Initialize the SHAP explainer
            explainer = shap.Explainer(model, background_X)

            # Calculate SHAP values for the sample
            shap_values = explainer(sample_data[feature_names])

            # Create a SHAP force plot
            fig, ax = plt.subplots(figsize=(10, 3))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            plt.title("SHAP Waterfall Plot - Feature Contributions")
            plt.tight_layout()
            st.pyplot(fig)

            # Add interpretation
            st.markdown(
                """
            <div style="background-color: #f5f5f5; padding: 2px; border-radius: 8px; margin: 5px 0;">
                <h4 style="color: #2c3e50; margin-top: 0;">How to Interpret the Chart:</h4>
                <ul>
                    <li>Red bars push the prediction higher</li>
                    <li>Blue bars push the prediction lower</li>
                    <li>The final prediction is the sum of the base value and all feature contributions</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Create a SHAP summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                sample_data[feature_names],
                feature_names=feature_names,
                show=False,
            )
            plt.tight_layout()
            st.pyplot(plt)
    except Exception as e:
        st.error(f"Error in SHAP analysis: {str(e)}")
        st.info(
            "SHAP analysis requires scikit-learn, shap, and matplotlib libraries. Please ensure they are installed."
        )

    st.markdown("</div>", unsafe_allow_html=True)  # Close the card container


def show_what_if_analysis(model, feature_names, df):
    """Interactive what-if analysis to see how changing inputs affects predictions"""
    st.markdown(
        """
    <div style="background-color: white; padding: 2px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
        <h3 style="color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">What-If Analysis</h3>
        <p>Experiment with different input values to see how they affect predictions. This helps understand the model's sensitivity to various factors.</p>
    """,
        unsafe_allow_html=True,
    )

    try:
        # Base input form
        st.subheader("Set Base Values")

        col1, col2 = st.columns(2)
        with col1:
            maize_price = st.slider("Maize Price", min_value=0.0, max_value=500.0, value=100.0, step=10.0)
            rice_price = st.slider("Rice Price", min_value=0.0, max_value=500.0, value=150.0, step=10.0)
            sorghum_price = st.slider("Sorghum Price", min_value=0.0, max_value=500.0, value=120.0, step=10.0)
            oil_price = st.slider("Oil Price", min_value=0.0, max_value=1000.0, value=200.0, step=20.0)

        with col2:
            exchange_rate = st.slider("Exchange Rate", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            cpi = st.slider("CPI", min_value=50.0, max_value=200.0, value=100.0, step=5.0)
            shock = st.slider("Shock Index", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            population = st.slider("Population", min_value=0, max_value=1000000, value=100000, step=10000)

        # Create input data for prediction
        input_data = pd.DataFrame(
            [
                {
                    "market_price_maize": maize_price,
                    "market_price_rice": rice_price,
                    "market_price_sorghum": sorghum_price,
                    "market_price_oil": oil_price,
                    "exchange_rate": exchange_rate,
                    "cpi_all_groups": cpi,
                    "conflict_critical": shock,
                    "population": population,
                }
            ]
        )

        # Make prediction
        prediction = model.predict(input_data[feature_names])[0]

        # Display prediction
        st.markdown(
            f"""
        <div style="background-color: #e8f5e9; padding: 2px; border-radius: 10px; margin: 5px 0; text-align: center; border-left: 4px solid #4CAF50;">
            <h2 style="margin: 10px 0; color: #2c3e50;">Predicted Food Price Index with Current Settings</h2>
            <p style="font-size: 32px; font-weight: bold; color: #4CAF50; margin: 10px 0;">{prediction:.2f}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Sensitivity analysis
        st.subheader("Sensitivity Analysis")

        sensitivity_feature = st.selectbox(
            "Select feature to analyze sensitivity:",
            options=feature_names,
            format_func=lambda x: x.replace("market_price_", "").replace("_", " ").title(),
        )

        # Get base value for the selected feature
        base_value = input_data[sensitivity_feature].iloc[0]

        # Create range of values for sensitivity analysis
        if sensitivity_feature == "population":
            min_value = max(0, base_value - base_value * 0.5)
            max_value = base_value + base_value * 0.5
            values = np.linspace(min_value, max_value, 10)
        else:
            min_value = max(0, base_value - base_value * 0.5)
            max_value = base_value + base_value * 0.5
            if min_value == max_value:  # Handle case where base_value is 0
                min_value = 0
                max_value = 10
            values = np.linspace(min_value, max_value, 10)

        # Calculate predictions for each value
        sensitivity_results = []

        for value in values:
            # Create a copy of the input data and update the selected feature
            temp_data = input_data.copy()
            temp_data[sensitivity_feature] = value

            # Make prediction
            temp_prediction = model.predict(temp_data[feature_names])[0]

            # Store result
            sensitivity_results.append({"Value": value, "Prediction": temp_prediction})

        # Create DataFrame from results
        sensitivity_df = pd.DataFrame(sensitivity_results)

        # Create line chart
        fig = px.line(
            sensitivity_df,
            x="Value",
            y="Prediction",
            title=f"Sensitivity Analysis for {sensitivity_feature.replace('market_price_', '').replace('_', ' ').title()}",
            markers=True,
        )

        # Add vertical line at current value
        fig.add_vline(
            x=base_value,
            line_dash="dash",
            line_color="red",
            annotation_text="Current Value",
            annotation_position="top right",
        )

        # Add horizontal line at current prediction
        fig.add_hline(
            y=prediction,
            line_dash="dash",
            line_color="green",
            annotation_text="Current Prediction",
            annotation_position="left",
        )

        # Customize layout
        fig.update_layout(
            xaxis_title=f"{sensitivity_feature.replace('market_price_', '').replace('_', ' ').title()} Value",
            yaxis_title="Predicted Food Price Index",
            plot_bgcolor="rgba(255,255,255,0.9)",
            paper_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#2c3e50"),
            xaxis=dict(showgrid=True, gridcolor="#eee"),
            yaxis=dict(showgrid=True, gridcolor="#eee"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add explanation
        st.markdown(
            f"""
        <div style="background-color: #e3f2fd; padding: 2px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #2196F3;">
            <p style="margin: 0;"><span style="font-size: 20px;"> </span> <strong>Insight:</strong> The chart above shows how the predicted food price index changes when you vary the {sensitivity_feature.replace('market_price_', '').replace('_', ' ').lower()} value while keeping all other inputs constant. The steeper the line, the more sensitive the model is to this feature.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error in what-if analysis: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)  # Close the card container