import streamlit as st
from map import render_map
import base64
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"


def get_image_base64(image_path):
    print(f"Loading image from: {image_path}")
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def show_home_page(df):
    """
    Display the home page with map visualization

    Parameters:
    df (pandas.DataFrame): The dataset to visualize
    """
    st.markdown(
        """
    <style>
    /* Fix for map container display */
    .leaflet-container {
        width: 100% !important;
        height: 550px !important;
        z-index: 0 !important;
    }
    
    /* Ensure streamlit elements don't overlap */
    .stApp .element-container:has(.stHeading) {
        margin-bottom: 0 !important;
    }
    
    /* Ensure map wrapper has proper dimensions */
    .element-container iframe {
        width: 100% !important;
    }
    
    /* Add rounded corners to the map */
    .st-emotion-cache-r421ms {
        border-radius: 0.5rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="background: linear-gradient(to right, #4b6cb7, #182848); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center;"> Welcome to the Somalia Food Security Dashboard</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric("Comodities", 4, ASSETS_DIR / "commodities.png")
    with col2:
        display_metric(
            "Regions Covered", df["adm1_name"].nunique(), ASSETS_DIR / "regions.png"
        )

    with col3:
        display_metric(
            "Markets Monitored", df["mkt_name"].nunique(), ASSETS_DIR / "markets.png"
        )

    with col4:
        display_metric(
            "Avg Food Price Index",
            f"{df['food_price_index'].mean():.2f}",
            ASSETS_DIR / "gauge.png",
        )

    st.markdown(
        """
    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h3 style="color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">Somalia Food Security Overview</h3>
        <p style="color: #333; line-height: 1.6;">
            This interactive dashboard provides comprehensive insights into food security indicators across Somalia. 
            Monitor market prices, climate conditions, conflict levels, and socio-economic factors that impact food security. 
            The visualizations help identify vulnerable regions and inform targeted interventions for food security programs.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<h2>Geographic Distribution</h2>", unsafe_allow_html=True)

    render_map(df)

    st.markdown(
        """
    <div style="margin-top: 30px; text-align: center; color: #666; font-size: 14px;">
        <p>Data last updated: January 2026 | Dashboard Version 1.0</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_metric(label, value, icon_path):
    """
    Display a metric card with a local icon (Base64 embedded)
    """
    icon_base64 = get_image_base64(icon_path)

    st.markdown(
        f"""
        <div style="
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin-bottom: 20px;
        ">
            <div style="margin-bottom: 10px;">
                <img src="data:image/png;base64,{icon_base64}" width="40" />
            </div>
            <div style="font-size: 28px; font-weight: bold; color: #4CAF50;">
                {value}
            </div>
            <div style="font-size: 16px; color: #666;">
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
