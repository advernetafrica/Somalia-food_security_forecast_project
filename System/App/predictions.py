import streamlit as st
import pandas as pd
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

        # Get valid categorical values from the encoder
        valid_counties = encoder.categories_[0]

        # Row 1: Location dropdowns
        loc_col1, loc_col2, loc_col3 = st.columns(3)
        with loc_col1:
            county = st.selectbox("County", options=valid_counties)

        # Filter subcounties based on selected county
        county_df = df[df["county_name"] == county]
        valid_subcounties = sorted(county_df["sub_county_name"].unique())

        with loc_col2:
            sub_county = st.selectbox("Sub-County", options=valid_subcounties)

        # Filter wards based on selected subcounty
        subcounty_df = county_df[county_df["sub_county_name"] == sub_county]
        valid_wards = sorted(subcounty_df["ward_name"].unique())

        with loc_col3:
            ward = st.selectbox("Ward", options=valid_wards)

        # Filter facilities based on selected ward
        ward_df = subcounty_df[subcounty_df["ward_name"] == ward]
        valid_facilities = sorted(ward_df["facility_name"].unique())

        fac_col1, fac_col2 = st.columns(2)
        with fac_col1:
            facility = st.selectbox("Facility", options=valid_facilities)

        # Get valid commodities for the selected facility
        facility_df = ward_df[ward_df["facility_name"] == facility]
        valid_commodities = sorted(facility_df["dataelement_name"].unique())

        with fac_col2:
            commodity = st.selectbox("Commodity", options=valid_commodities)

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

        # Calculate time series features dynamically if historical data exists
        facility_commodity_df = df[
            (df["county_name"] == county)
            & (df["sub_county_name"] == sub_county)
            & (df["ward_name"] == ward)
            & (df["facility_name"] == facility)
            & (df["dataelement_name"] == commodity)
        ].sort_values("period")

        # Initialize lag values
        lag_1_value = 0
        lag_3_value = 0
        rolling_mean_3_value = 0

        # Calculate lag features if we have historical data
        if not facility_commodity_df.empty:
            # Get the most recent values for dynamic calculation
            recent_values = facility_commodity_df["value"].tail(12).tolist()

            # Calculate lag features from historical data if available
            if len(recent_values) >= 1:
                lag_1_value = recent_values[-1]

            if len(recent_values) >= 3:
                lag_3_value = recent_values[-3]

            if len(recent_values) >= 3:
                rolling_mean_3_value = sum(recent_values[-3:]) / 3

        st.markdown(
            """
        <div style="background-color: #f1f8e9; padding: 2px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4CAF50;">
            <h4 style="color: #2c3e50; margin-top: 0;">Time Series Features</h4>
        </div>
        """,
            unsafe_allow_html=True,
        )

        lag_col1, lag_col2, lag_col3 = st.columns(3)
        with lag_col1:
            lag_1 = st.number_input(
                "Lag 1 (previous month)", min_value=0.0, value=float(lag_1_value)
            )
        with lag_col2:
            lag_3 = st.number_input(
                "Lag 3 (three months ago)", min_value=0.0, value=float(lag_3_value)
            )
        with lag_col3:
            rolling_mean_3 = st.number_input(
                "Rolling Mean (last 3 months)",
                min_value=0.0,
                value=float(rolling_mean_3_value),
            )

        # Add a visual indicator to show the auto-calculated values
        if not facility_commodity_df.empty:
            st.markdown(
                """
            <div style="background-color: #e3f2fd; padding: 2px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196F3;">
                <p style="margin: 0;"><span style="font-size: 20px;"> </span> Time series features have been auto-calculated based on historical data. You can adjust them if needed.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div style="background-color: #fff3e0; padding: 2px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #FF9800;">
                <p style="margin: 0;"><span style="font-size: 20px;">⚠️</span> No historical data found for this selection. Please enter lag values manually.</p>
            </div>
            """,
                unsafe_allow_html=True,
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

        st.markdown("</div>", unsafe_allow_html=True)  # Close the card container

        # Show prediction on button click
        predict_button = st.button("Predict", use_container_width=True)

        if predict_button:
            prediction = model.predict(input_data[features])[0]

            st.markdown(
                f"""
            <div style="background-color: #e8f5e9; padding: 2px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin: 5px 0; text-align: center; border-left: 4px solid #4CAF50;">
                <span style="font-size: 24px;"> </span>
                <h2 style="margin: 10px 0; color: #2c3e50;">Predicted Demand</h2>
                <p style="font-size: 32px; font-weight: bold; color: #4CAF50; margin: 10px 0;">{prediction:.2f} units</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Show visualization of recent history and prediction
            if not facility_commodity_df.empty:
                st.markdown(
                    """
                <div style="background-color: white; padding: 2px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-top: 5px;">
                    <h3 style="color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">Historical Data and Prediction</h3>
                """,
                    unsafe_allow_html=True,
                )

                # Create a combined visualization
                recent_df = facility_commodity_df.tail(12).copy()

                # Create a prediction point
                prediction_date = pd.to_datetime(f"{year}-{month:02d}-01")
                prediction_df = pd.DataFrame(
                    {
                        "period": [prediction_date],
                        "value": [prediction],
                        "type": ["Prediction"],
                    }
                )

                # Add type column to historical data
                recent_df["type"] = "Historical"

                # Combine data for visualization
                plot_df = pd.concat(
                    [recent_df[["period", "value", "type"]], prediction_df]
                )

                # Create plot
                fig = px.line(
                    plot_df,
                    x="period",
                    y="value",
                    color="type",
                    markers=True,
                    title=f"Historical Data and Prediction for {commodity}",
                    labels={"value": "Dispensed Units", "period": "Period"},
                )

                # Customize the plot
                fig.update_layout(
                    xaxis_title="Time Period",
                    yaxis_title="Dispensed Units",
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
