import streamlit as st
from map import render_map


def show_home_page(df):
    """
    Display the home page with map visualization

    Parameters:
    df (pandas.DataFrame): The dataset to visualize
    """
    # Apply custom CSS for map rendering and layout fixes
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

    # Apply custom header with gradient background
    st.markdown(
        """
    <div style="background: linear-gradient(to right, #4b6cb7, #182848); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center;"> Welcome to the Somalia Food Security Dashboard</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Display metrics summary
    col1, col2, col3 = st.columns(3)
    with col1:
        display_metric("Regions Covered", df["adm1_name"].nunique(), "🗺️")

    with col2:
        display_metric("Markets Monitored", df["mkt_name"].nunique(), "🏪")

    with col3:
        display_metric(
            "Avg Food Price Index", f"{df['food_price_index'].mean():.2f}", "📈"
        )

    # Create card-like container for the description
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

    # Geographic Distribution section with the map
    st.markdown("<h2>Geographic Distribution</h2>", unsafe_allow_html=True)

    # Render the map - moving CSS above ensures this displays correctly
    render_map(df)

    # Add footer with info
    st.markdown(
        """
    <div style="margin-top: 30px; text-align: center; color: #666; font-size: 14px;">
        <p>Data last updated: January 2026 | Dashboard Version 1.0</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_metric(label, value, icon):
    """
    Display a metric in a visually appealing card
    """
    st.markdown(
        f"""
    <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; height: 150px; display: flex; flex-direction: column; justify-content: center; margin-bottom: 20px;">
        <div style="font-size: 40px; margin-bottom: 10px;">{icon}</div>
        <div style="font-size: 28px; font-weight: bold; color: #4CAF50;">{value}</div>
        <div style="font-size: 16px; color: #666;">{label}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
