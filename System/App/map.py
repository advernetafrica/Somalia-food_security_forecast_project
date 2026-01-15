import streamlit as st
import folium
import json
import branca.colormap as cm
import pandas as pd
import streamlit.components.v1 as components


def render_map(df):
    """
    Renders a choropleth map of Kenya counties with health commodity distribution data
    """

    # Create a layout with map on the left and checkboxes on the right
    col1, col2 = st.columns([0.8, 0.2])

    # Get unique regions
    unique_regions = sorted(df["adm2_name"].unique())

    # In the right column, create vertical checkboxes for regions
    with col2:
        st.write("**Filter by District:**")

        # Create a dictionary to hold checkbox states
        selected_regions = {}
        for region in unique_regions:
            selected_regions[region] = st.checkbox(
                region, value=True, key=f"map_region_{region}"
            )

        # Get list of selected regions
        regions_to_show = [r for r, selected in selected_regions.items() if selected]

        # Show selection summary
        if len(regions_to_show) == len(unique_regions):
            st.info("Showing all districts")
        else:
            st.info(
                f"Showing {len(regions_to_show)} of {len(unique_regions)} districts"
            )

    # Filter data based on selected regions
    filtered_df = df[df["adm2_name"].isin(regions_to_show)]

    # In the left column, render the map
    with col1:
        try:
            # Load GeoJSON file
            with open("Data/somalia.geojson", "r", encoding="utf-8") as f:
                somalia_geo = json.load(f)

            # Aggregate data by district
            data = (
                filtered_df.groupby("adm2_name")["food_price_index"]
                .mean()
                .reset_index()
            )
            data["region"] = data["adm2_name"].str.upper()

            value_dict = dict(zip(data["region"], data["food_price_index"]))

            # Add the value data to each feature in the GeoJSON
            for feature in somalia_geo["features"]:
                region_name = feature["properties"].get("adm2_name", "")
                if region_name is None:
                    region_name = ""
                region_name = region_name.upper()
                value = value_dict.get(region_name, 0)

                # Add value to feature properties for tooltip
                feature["properties"]["FOOD_PRICE_INDEX"] = value
                # Format the value
                feature["properties"]["FORMATTED_INDEX"] = f"{value:.2f}"

            min_value = data["food_price_index"].min() if not data.empty else 0
            max_value = data["food_price_index"].max() if not data.empty else 1

            # Create the base map with explicit tile provider
            m = folium.Map(
                location=[5.1521, 46.1996], zoom_start=6, tiles="CartoDB positron"
            )

            # Create color scale
            colors = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]
            color_scale = cm.LinearColormap(colors, vmin=min_value, vmax=max_value)

            # Style functions
            def style_function(feature):
                region_name = feature["properties"].get("adm2_name", "")
                if region_name is None:
                    region_name = ""
                region_name = region_name.upper()
                value = value_dict.get(region_name, 0)
                return {
                    "fillColor": color_scale(value),
                    "color": "black",
                    "weight": 1,
                    "fillOpacity": 0.7,
                }

            def highlight_function(feature):
                return {
                    "weight": 3,
                    "color": "#666",
                    "dashArray": "",
                    "fillOpacity": 0.9,
                }

            # Add GeoJSON layer with enhanced tooltip
            tooltip = folium.GeoJsonTooltip(
                fields=["adm2_name", "FORMATTED_INDEX"],
                aliases=["District:", "Avg Food Price Index:"],
                localize=True,
                sticky=True,
                style=(
                    "background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
                ),
            )

            folium.GeoJson(
                somalia_geo,
                style_function=style_function,
                highlight_function=highlight_function,
                tooltip=tooltip,
                name="Somalia Districts",
            ).add_to(m)

            # Add color scale
            color_scale.caption = "Average Food Price Index"
            m.add_child(color_scale)

            # Get the average index for display
            avg_index = filtered_df["food_price_index"].mean()

            # Display average index
            st.markdown(
                f"**Average Food Price Index Across Selected Districts: {avg_index:.2f}**"
            )

            # Get the HTML representation of the map
            map_html = m._repr_html_()

            # Display the map using components.html instead of st_folium
            components.html(map_html, height=600)

        except FileNotFoundError:
            st.error(
                "❌ Error: Somalia GeoJSON file not found. Please make sure 'somalia.geojson' is in the Data directory."
            )
        except Exception as e:
            st.error(f"❌ Error rendering map: {str(e)}")
            st.info("Try refreshing the page or check console for more details.")
