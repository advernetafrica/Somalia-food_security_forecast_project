import json
import os

import branca.colormap as cm
import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from inference import BASE_DIR

DATA_DIR = os.path.join(BASE_DIR, "..", "Data")

COMMODITY_OPTIONS = {
    "Food Price Index": "food_price_index",
    "Maize Price": "market_price_maize",
    "Rice Price": "market_price_rice",
    "Sorghum Price": "market_price_sorghum",
    "Cooking Oil Price": "market_price_oil",
}

# Risk palette — green (safe) → red (critical). Used for every
# choropleth/heatmap layer that encodes a food-security stress signal.
RISK_PALETTE = ["#0ea5e9", "#10b981", "#facc15", "#f97316", "#ef4444"]

# Hard thresholds aligned with the interpretation bands on the Home page.
FPI_BAND_INDEX = [0.0, 0.90, 1.20, 1.60, 2.00, 3.50]


def render_map(df):
    """Choropleth of Somalia districts coloured by the selected commodity."""

    map_col, ctrl_col = st.columns([0.78, 0.22], gap="large")

    with ctrl_col:
        st.markdown(
            "<div class='eyebrow' style='margin-bottom:8px;'>Layer</div>",
            unsafe_allow_html=True,
        )
        selected_label = st.radio(
            "Indicator",
            options=list(COMMODITY_OPTIONS.keys()),
            index=0,
            label_visibility="collapsed",
            key="map_commodity_radio",
        )
        selected_commodity = COMMODITY_OPTIONS[selected_label]

    with map_col:
        try:
            with open(os.path.join(DATA_DIR, "somalia.geojson"), "r", encoding="utf-8") as f:
                somalia_geo = json.load(f)

            data = (
                df.groupby("adm2_name")[[selected_commodity]]
                .mean()
                .mean(axis=1)
                .reset_index()
                .rename(columns={0: "avg_value"})
            )
            data["region"] = data["adm2_name"].str.upper()
            value_dict = dict(zip(data["region"], data["avg_value"]))

            for feature in somalia_geo["features"]:
                region_name = (feature["properties"].get("adm2_name") or "").upper()
                value = value_dict.get(region_name, 0)
                feature["properties"]["FOOD_PRICE_INDEX"] = value
                feature["properties"]["FORMATTED_INDEX"] = f"{value:.2f}"

            min_value = data["avg_value"].min() if not data.empty else 0
            max_value = data["avg_value"].max() if not data.empty else 1

            m = folium.Map(
                location=[5.1521, 46.1996],
                zoom_start=6,
                tiles="CartoDB positron",
                control_scale=True,
            )

            if selected_commodity == "food_price_index":
                # Hard-banded risk scale that matches the Home-page legend.
                color_scale = cm.StepColormap(
                    RISK_PALETTE,
                    index=FPI_BAND_INDEX,
                    vmin=FPI_BAND_INDEX[0],
                    vmax=FPI_BAND_INDEX[-1],
                )
            else:
                # Commodity prices — higher = worse affordability → same risk gradient.
                color_scale = cm.LinearColormap(
                    RISK_PALETTE, vmin=min_value, vmax=max_value
                )
            color_scale.caption = f"{selected_label} · district average"

            def style_function(feature):
                region_name = (feature["properties"].get("adm2_name") or "").upper()
                value = value_dict.get(region_name, 0)
                return {
                    "fillColor": color_scale(value),
                    "color": "#475569",
                    "weight": 0.8,
                    "fillOpacity": 0.82,
                }

            def highlight_function(_feature):
                return {
                    "weight": 2.5,
                    "color": "#0f172a",
                    "fillOpacity": 0.95,
                }

            tooltip = folium.GeoJsonTooltip(
                fields=["adm2_name", "FORMATTED_INDEX"],
                aliases=["District", f"{selected_label}"],
                localize=True,
                sticky=True,
                style=(
                    "background-color: #0f172a;"
                    "color: #f8fafc;"
                    "font-family: Inter, sans-serif;"
                    "font-size: 12px;"
                    "padding: 8px 12px;"
                    "border-radius: 8px;"
                    "border: none;"
                ),
            )

            folium.GeoJson(
                somalia_geo,
                style_function=style_function,
                highlight_function=highlight_function,
                tooltip=tooltip,
                name="Somalia Districts",
            ).add_to(m)

            m.add_child(color_scale)

            avg_index = data["avg_value"].mean() if not data.empty else 0
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:10px 14px;background:#f8fafc;
                            border:1px solid #e2e8f0;border-radius:10px;margin-bottom:10px;">
                    <span style="color:#64748b;font-size:0.85rem;font-weight:600;
                                 letter-spacing:0.05em;text-transform:uppercase;">
                      National mean · {selected_label}
                    </span>
                    <span style="color:#0f172a;font-weight:700;font-size:1.1rem;">
                      {avg_index:.2f}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            components.html(m._repr_html_(), height=580)

        except FileNotFoundError:
            st.error("Somalia GeoJSON file not found in the Data directory.")
        except Exception as e:
            st.error(f"Error rendering map: {e}")
