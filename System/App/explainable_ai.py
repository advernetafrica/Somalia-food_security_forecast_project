import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
from predictions import download_file
from utils import filter_data, calculate_lag_features


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
            # URL of the model on GitHub (placeholder for Somalia Food Security model)
            model_url = (
                "https://github.com/user/repo/raw/main/Models/food_security_model.pkl"
            )
            encoder_url = "https://github.com/user/repo/raw/main/Models/encoder.pkl"

            # Download the model and encoder files
            model_path = download_file(model_url, "best_gb_model.pkl")
            encoder_path = download_file(encoder_url, "encoder.pkl")

            if not model_path or not encoder_path:
                return  # If downloading fails, exit early

            # Load the model using joblib
            model = joblib.load(model_path)
            encoder = joblib.load(encoder_path)

            # Create tabs for different explanation approaches
            tab1, tab2, tab3 = st.tabs(
                ["Feature Importance", "SHAP Values", "What-If Analysis"]
            )

            with tab1:
                show_feature_importance(model, df)

            with tab2:
                show_shap_analysis(model, encoder, df)

            with tab3:
                show_what_if_analysis(model, encoder, df)

        except FileNotFoundError:
            st.error(
                "Model files not found. Please ensure 'best_gb_model.pkl' and 'encoder.pkl' are in the application directory."
            )


def show_feature_importance(model, df):
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
        # Get feature names from the model
        feature_names = model.feature_names_in_

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
            <p style="margin: 0;"><span style="font-size: 20px;"> </span> <strong>Key Insight:</strong> The top {len(top_features)} most influential features are: <strong>{", ".join(top_features)}</strong>. These features have the largest impact on predicted demand.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error calculating feature importance: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)  # Close the card container


def show_shap_analysis(model, encoder, df):
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
        # Get feature names
        feature_names = model.feature_names_in_

        # Select a sample to explain
        # First, let users filter to a specific location
        st.subheader("Select a specific case to explain:")

        # Location filters
        col1, col2, col3 = st.columns(3)
        with col1:
            counties = sorted(df["county_name"].unique())
            county = st.selectbox("County", options=counties)

        with col2:
            filtered_df = df[df["county_name"] == county]
            subcounties = sorted(filtered_df["sub_county_name"].unique())
            subcounty = st.selectbox("Sub-County", options=subcounties)

        with col3:
            filtered_df = df[df["sub_county_name"] == subcounty]
            wards = sorted(filtered_df["ward_name"].unique())
            ward = st.selectbox("Ward", options=wards)

        # Further filter by facility and commodity
        loc_col1, loc_col2 = st.columns(2)

        with loc_col1:
            ward_df = filtered_df[filtered_df["ward_name"] == ward]
            valid_facilities = sorted(ward_df["facility_name"].unique())
            facility = st.selectbox("Facility", options=valid_facilities)

        with loc_col2:
            facility_df = ward_df[ward_df["facility_name"] == facility]
            commodities = sorted(facility_df["dataelement_name"].unique())
            commodity = st.selectbox("Commodity", options=commodities)

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

        # Create sample data
        filtered_df = facility_df[facility_df["dataelement_name"] == commodity]

        if not filtered_df.empty:
            # Get most recent data for time-based features
            lag_features = calculate_lag_features(filtered_df)

            # Create a sample for explanation
            sample_data = pd.DataFrame(
                [
                    {
                        "county_name": county,
                        "sub_county_name": subcounty,
                        "ward_name": ward,
                        "facility_name": facility,
                        "dataelement_name": commodity,
                        "month": month,
                        "year": year,
                        "quarter": quarter,
                        "lag_1": lag_features["lag_1"],
                        "lag_3": lag_features["lag_3"],
                        "rolling_mean_3": lag_features["rolling_mean_3"],
                    }
                ]
            )

            # Transform categorical features
            cat_cols = [
                "county_name",
                "sub_county_name",
                "ward_name",
                "facility_name",
                "dataelement_name",
            ]
            sample_data_encoded = sample_data.copy()
            sample_data_encoded[cat_cols] = encoder.transform(sample_data[cat_cols])

            # Predict
            features = [
                "month",
                "year",
                "quarter",
                "lag_1",
                "lag_3",
                "rolling_mean_3",
            ] + cat_cols
            prediction = model.predict(sample_data_encoded[features])[0]

            # Display prediction
            st.markdown(
                f"""
            <div style="background-color: #e8f5e9; padding: 2px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #4CAF50;">
                <p style="margin: 0;"><span style="font-size: 20px;"> </span> <strong>Predicted Demand:</strong> {prediction:.2f} units</p>
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

                # Encode the background data
                background_data_encoded = background_data.copy()
                background_data_encoded[cat_cols] = encoder.transform(
                    background_data[cat_cols]
                )

                # Initialize the SHAP explainer
                explainer = shap.Explainer(model)

                # Calculate SHAP values for the sample
                shap_values = explainer(sample_data_encoded[features])

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
                    sample_data_encoded[features],
                    feature_names=features,
                    show=False,
                )
                plt.tight_layout()
                st.pyplot(plt)
        else:
            st.warning(
                "No data available for the selected filters. Please choose different criteria."
            )
    except Exception as e:
        st.error(f"Error in SHAP analysis: {str(e)}")
        st.info(
            "SHAP analysis requires scikit-learn, shap, and matplotlib libraries. Please ensure they are installed."
        )

    st.markdown("</div>", unsafe_allow_html=True)  # Close the card container


def show_what_if_analysis(model, encoder, df):
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
        # Get valid categorical values from the encoder
        valid_counties = encoder.categories_[0]

        # Get features used by the model
        feature_names = model.feature_names_in_

        # Base configuration selection
        st.subheader("1. Select Base Configuration")

        # Location filters
        col1, col2, col3 = st.columns(3)
        with col1:
            county = st.selectbox("County", options=valid_counties, key="whatif_county")

        # Filter subcounties based on selected county
        county_df = df[df["county_name"] == county]
        valid_subcounties = sorted(county_df["sub_county_name"].unique())

        with col2:
            sub_county = st.selectbox(
                "Sub-County", options=valid_subcounties, key="whatif_subcounty"
            )

        # Filter wards based on selected subcounty
        subcounty_df = county_df[county_df["sub_county_name"] == sub_county]
        valid_wards = sorted(subcounty_df["ward_name"].unique())

        with col3:
            ward = st.selectbox("Ward", options=valid_wards, key="whatif_ward")

        # Filter facilities based on selected ward
        ward_df = subcounty_df[subcounty_df["ward_name"] == ward]
        valid_facilities = sorted(ward_df["facility_name"].unique())

        col4, col5 = st.columns(2)
        with col4:
            facility = st.selectbox(
                "Facility", options=valid_facilities, key="whatif_facility"
            )

        # Get valid commodities for the selected facility
        facility_df = ward_df[ward_df["facility_name"] == facility]
        valid_commodities = sorted(facility_df["dataelement_name"].unique())

        with col5:
            commodity = st.selectbox(
                "Commodity", options=valid_commodities, key="whatif_commodity"
            )

        # Time features
        st.subheader("2. Adjust Numerical Features")
        st.markdown(
            "Experiment with these values to see how they affect the prediction"
        )

        # Get base values for numerical features
        facility_commodity_df = df[
            (df["county_name"] == county)
            & (df["sub_county_name"] == sub_county)
            & (df["ward_name"] == ward)
            & (df["facility_name"] == facility)
            & (df["dataelement_name"] == commodity)
        ].sort_values("period")

        lag_features = calculate_lag_features(facility_commodity_df)

        # Create sliders for numerical inputs
        col1, col2 = st.columns(2)
        with col1:
            month = st.slider(
                "Month", min_value=1, max_value=12, value=4, key="whatif_month"
            )
            quarter = (month - 1) // 3 + 1
            st.info(f"Quarter: Q{quarter}")

            lag_1 = st.slider(
                "Lag 1 (previous month)",
                min_value=0.0,
                max_value=max(100.0, lag_features["lag_1"] * 2),
                value=lag_features["lag_1"],
                step=1.0,
                key="whatif_lag1",
            )

        with col2:
            year = st.slider(
                "Year", min_value=2020, max_value=2030, value=2024, key="whatif_year"
            )

            lag_3 = st.slider(
                "Lag 3 (three months ago)",
                min_value=0.0,
                max_value=max(100.0, lag_features["lag_3"] * 2),
                value=lag_features["lag_3"],
                step=1.0,
                key="whatif_lag3",
            )

            rolling_mean_3 = st.slider(
                "Rolling Mean (last 3 months)",
                min_value=0.0,
                max_value=max(100.0, lag_features["rolling_mean_3"] * 2),
                value=lag_features["rolling_mean_3"],
                step=1.0,
                key="whatif_rolling",
            )

        # Create input data for prediction
        input_data = pd.DataFrame(
            [
                {
                    "county_name": county,
                    "sub_county_name": sub_county,
                    "ward_name": ward,
                    "facility_name": facility,
                    "dataelement_name": commodity,
                    "month": month,
                    "year": year,
                    "quarter": quarter,
                    "lag_1": lag_1,
                    "lag_3": lag_3,
                    "rolling_mean_3": rolling_mean_3,
                }
            ]
        )

        # Transform categorical features
        cat_cols = [
            "county_name",
            "sub_county_name",
            "ward_name",
            "facility_name",
            "dataelement_name",
        ]
        input_data[cat_cols] = encoder.transform(input_data[cat_cols])
        features = [
            "month",
            "year",
            "quarter",
            "lag_1",
            "lag_3",
            "rolling_mean_3",
        ] + cat_cols

        # Make prediction
        prediction = model.predict(input_data[features])[0]

        # Display prediction
        st.markdown(
            f"""
        <div style="background-color: #e8f5e9; padding: 2px; border-radius: 10px; margin: 5px 0; text-align: center; border-left: 4px solid #4CAF50;">
            <h2 style="margin: 10px 0; color: #2c3e50;">Predicted Demand with Current Settings</h2>
            <p style="font-size: 32px; font-weight: bold; color: #4CAF50; margin: 10px 0;">{prediction:.2f} units</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Sensitivity analysis
        st.subheader("3. Sensitivity Analysis")

        sensitivity_feature = st.selectbox(
            "Select feature to analyze sensitivity:",
            options=["lag_1", "lag_3", "rolling_mean_3"],
            format_func=lambda x: {
                "lag_1": "Previous Month (Lag 1)",
                "lag_3": "Three Months Ago (Lag 3)",
                "rolling_mean_3": "Rolling Average (Last 3 Months)",
            }[x],
        )

        # Get base value for the selected feature
        base_value = input_data[sensitivity_feature].iloc[0]

        # Create range of values for sensitivity analysis
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
            temp_prediction = model.predict(temp_data[features])[0]

            # Store result
            sensitivity_results.append({"Value": value, "Prediction": temp_prediction})

        # Create DataFrame from results
        sensitivity_df = pd.DataFrame(sensitivity_results)

        # Create line chart
        fig = px.line(
            sensitivity_df,
            x="Value",
            y="Prediction",
            title=f"Sensitivity Analysis for {sensitivity_feature}",
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
            xaxis_title=f"{sensitivity_feature} Value",
            yaxis_title="Predicted Demand",
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
            <p style="margin: 0;"><span style="font-size: 20px;"> </span> <strong>Insight:</strong> The chart above shows how the predicted demand changes when you vary the {sensitivity_feature} value while keeping all other inputs constant. The steeper the line, the more sensitive the model is to this feature.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error in what-if analysis: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)  # Close the card container
