import os
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from inference import BASE_DIR
from home import show_home_page
from visualizations import show_visualizations_page
from predictions import show_predictions_page
from explainable_ai import show_explainable_ai_page


APP_DIR = Path(__file__).resolve().parent
STYLES_PATH = APP_DIR / "styles.css"


st.set_page_config(
    page_title="Somalia Food Security",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)


def load_css():
    with open(STYLES_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# Hide only the footer — keep the header + toolbar visible so the
# sidebar collapse toggle, settings menu, and deploy button stay accessible.
st.markdown(
    """
    <style>
      footer { visibility: hidden; }
      [data-testid="stHeader"] {
          background: transparent !important;
      }
      [data-testid="collapsedControl"] {
          visibility: visible !important;
          display: block !important;
          color: #0f172a !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


DATA_DIR = os.path.join(BASE_DIR, "..", "..", "Data")


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "processed_dataset.csv"))
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    return df


df = load_data()
df.rename(columns={"adm1_name": "region", "mkt_name": "district"}, inplace=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 4px 6px 18px 6px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:36px;height:36px;border-radius:10px;
                            background:linear-gradient(135deg,#10b981,#0ea5e9);
                            display:flex;align-items:center;justify-content:center;
                            font-size:18px;">🌾</div>
                <div>
                    <div style="font-weight:700;font-size:1rem;color:#f8fafc;line-height:1.1;">
                        Food Security
                    </div>
                    <div style="color:#94a3b8;font-size:0.78rem;letter-spacing:0.04em;">
                        Somalia · Dashboard
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = option_menu(
        menu_title=None,
        options=["Home", "Visualizations", "Predictions", "Explainable AI"],
        icons=["house-door-fill", "bar-chart-line-fill", "graph-up-arrow", "lightbulb-fill"],
        default_index=0,
        styles={
            "container": {
                "padding": "4px 0",
                "background-color": "transparent",
            },
            "icon": {"color": "#10b981", "font-size": "17px"},
            "nav-link": {
                "color": "#e2e8f0",
                "font-size": "0.95rem",
                "text-align": "left",
                "padding": "10px 14px",
                "margin": "2px 0",
                "border-radius": "10px",
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg,#059669,#10b981)",
                "color": "#ffffff",
                "font-weight": "600",
            },
        },
    )

    st.markdown(
        """
        <div style="position:absolute;bottom:24px;left:24px;right:24px;
                    color:#94a3b8;font-size:0.75rem;">
            <div style="opacity:0.8;">v1.0 · Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
if selected_page == "Home":
    show_home_page(df)
elif selected_page == "Visualizations":
    show_visualizations_page(df)
elif selected_page == "Predictions":
    show_predictions_page(df)
elif selected_page == "Explainable AI":
    show_explainable_ai_page(df)
