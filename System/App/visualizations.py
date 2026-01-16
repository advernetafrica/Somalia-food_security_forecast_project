import streamlit as st
import plotly.express as px


def show_visualizations_page(df):
    """
    Display the visualizations page with interactive charts and filters

    Parameters:
    df (pandas.DataFrame): The dataset to visualize
    """
    # Apply custom header with gradient background
    st.markdown(
        """
    <div style="background: linear-gradient(to right, #00b09b, #96c93d); padding: 2px; border-radius: 10px; margin-bottom: 5px;">
        <h1 style="color: white; text-align: center;"> Somalia Food Security Visualizer</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create a subtitle
    st.markdown(
        """
    <div style="background-color: white; padding: 2px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
        <p style="color: #333; text-align: center; font-size: 18px;">Use the filters below to explore food security indicators.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create a centered container with reduced padding for all visualizations content
    col1, content_col, col2 = st.columns([0.05, 0.9, 0.05])

    with content_col:
        # Create a card-like container for filters
        st.markdown(
            """
        <div style="background-color: white; padding: 2px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
        """,
            unsafe_allow_html=True,
        )

        # Tab Filters for Region, District, Market
        tab1, tab2, tab3 = st.tabs(["Region", "District", "Market"])

        with tab1:
            region_options = ["All Regions"] + list(df["adm1_name"].unique())
            selected_region = st.selectbox("Select Region", region_options)

        # Filter by region only if a specific region is selected
        if selected_region != "All Regions":
            filtered_df = df[df["adm1_name"] == selected_region]
        else:
            filtered_df = df.copy()  # Use all regions

        # --- District selection ---
        with tab2:
            district_options = ["All Districts"] + list(filtered_df["adm2_name"].unique())
            selected_district = st.selectbox("Select District", district_options)

        if selected_district != "All Districts":
            filtered_df = filtered_df[filtered_df["adm2_name"] == selected_district]

        # --- Market selection ---
        with tab3:
            market_options = ["All Markets"] + list(filtered_df["mkt_name"].unique())
            selected_market = st.selectbox("Select Market", market_options)

        if selected_market != "All Markets":
            filtered_df = filtered_df[filtered_df["mkt_name"] == selected_market]

        st.markdown("</div>", unsafe_allow_html=True)  # Close the card container

        # Create a card-like container for the first chart
        st.markdown(
            """
        <div style="background-color: white; padding: 2px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
            <h3 style="color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">Food Price Index Over Time</h3>
        """,
            unsafe_allow_html=True,
        )

        # Visualizing Food Price Index Over Time
        time_series = filtered_df.groupby("Date")["food_price_index"].mean().reset_index()
        fig = px.line(
            time_series,
            x="Date",
            y="food_price_index",
            markers=True,
            title=f"Food Price Index Over Time ({selected_region})",
            labels={"food_price_index": "Food Price Index", "Date": "Date"},
        )

        # Enhance chart styling
        fig.update_layout(
            plot_bgcolor="rgba(255,255,255,0.9)",
            paper_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#2c3e50"),
            title_font=dict(size=20, color="#2c3e50"),
            xaxis=dict(showgrid=True, gridcolor="#eee"),
            yaxis=dict(showgrid=True, gridcolor="#eee"),
            margin=dict(l=20, r=20, t=60, b=20),
        )

        # Enhance line style
        fig.update_traces(
            line=dict(width=3, color="#4CAF50"), marker=dict(size=8, color="#4CAF50")
        )

        # Make plotly chart use the full width
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)  # Close the card container

        # Create a card-like container for commodity comparison
        st.markdown(
            """
        <div style="background-color: white; padding: 2px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 5px;">
            <h3 style="color: #2c3e50; border-bottom: 2px solid #FFA726; padding-bottom: 10px;">
                <span style="font-size: 24px;"> </span> Select Commodities to Compare
            </h3>
        """,
            unsafe_allow_html=True,
        )

        # Use columns to display checkboxes more efficiently
        checkbox_cols = st.columns(3)
        price_columns = [
            "market_price_maize",
            "market_price_rice",
            "market_price_sorghum",
            "market_price_oil",
        ]
        selected_prices = []

        for i, price_col in enumerate(price_columns):
            with checkbox_cols[i % 3]:
                # Custom styled checkbox container
                if st.checkbox(
                    price_col.replace("market_price_", "").title(),
                    value=True,
                    key=f"viz_checkbox_{price_col}",
                ):
                    selected_prices.append(price_col)

                # Market Price Trend Visualization
        if selected_prices:
            st.markdown(
                """
            <div style="background-color: #f5f5f5; padding: 2px; border-radius: 8px; margin: 5px 0;">
                <h4 style="color: #2c3e50; margin-top: 0;">Trends for Selected Market Prices</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Melt the dataframe for plotting
            melted_df = filtered_df.melt(
                id_vars=["Date"],
                value_vars=selected_prices,
                var_name="Commodity",
                value_name="Price",
            )
            melted_df["Commodity"] = (
                melted_df["Commodity"].str.replace("market_price_", "").str.title()
            )

            # Compute median per Date and Commodity
            median_df = (
                melted_df.groupby(["Date", "Commodity"])["Price"]
                .median()
                .reset_index()
            )

            fig = px.line(
                median_df,
                x="Date",
                y="Price",
                color="Commodity",
                markers=True,
                labels={"Price": "Market Price", "Date": "Date", "Commodity": "Commodity"},
            )

            # Enhance chart styling
            fig.update_layout(
                plot_bgcolor="rgba(255,255,255,0.9)",
                paper_bgcolor="rgba(255,255,255,0)",
                font=dict(color="#2c3e50"),
                xaxis=dict(showgrid=True, gridcolor="#eee"),
                yaxis=dict(showgrid=True, gridcolor="#eee"),
                legend_title_font=dict(size=14),
                legend=dict(
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#dddddd",
                    borderwidth=1,
                    orientation="h",
                ),
                margin=dict(l=20, r=20, t=20, b=20),
            )

            # Enhance line style
            fig.update_traces(line=dict(width=2.5), marker=dict(size=6))

            # Display chart
            st.plotly_chart(fig, use_container_width=True)


        # Optional raw data display
        show_data = st.checkbox("Show Raw Data", value=False)
        if show_data:
            st.markdown(
                """
            <div style="background-color: #f5f5f5; padding: 2px; border-radius: 8px; margin: 5px 0;">
                <h4 style="color: #2c3e50; margin-top: 0;">Raw Data</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Drop the 'Unnamed: 0' column and reset index before displaying
            display_df = filtered_df.copy()
            if "Unnamed: 0" in display_df.columns:
                display_df = display_df.drop("Unnamed: 0", axis=1)
            display_df = display_df.reset_index(drop=True)
            st.dataframe(display_df.sort_values("period"), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)  # Close the card container

        # Add footer with info
        st.markdown(
            """
        <div style="margin-top: 10px; text-align: center; color: #666; font-size: 14px;">
            <p>Try selecting different markets or price types to explore the data in more detail.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
