import pandas as pd
import plotly.express as px
import streamlit as st

from inference import recursive_forecast_hybrid


def show_predictions_page(df):
    """Forecast the Food Price Index for a region/district at a target date."""

    st.markdown(
        """
        <div class="app-hero">
            <span class="app-badge">Forecaster</span>
            <h1>Food Price Index Forecaster</h1>
            <p>Configure a scenario below and project the Food Price Index forward
            month-by-month until the target date.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Scenario card ----------
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Scenario Inputs</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        region = st.selectbox("Region", sorted(df["region"].unique()))
        month = st.number_input("Target month", min_value=1, max_value=12, value=4)
        market_price_maize = st.number_input(
            "Maize price", value=float(df["market_price_maize"].median()), step=1.0
        )
        market_price_rice = st.number_input(
            "Rice price", value=float(df["market_price_rice"].median()), step=1.0
        )
        market_price_sorghum = st.number_input(
            "Sorghum price", value=float(df["market_price_sorghum"].median()), step=1.0
        )
        market_price_oil = st.number_input(
            "Oil price", value=float(df["market_price_oil"].median()), step=1.0
        )
        population = st.number_input(
            "Population", value=float(df["population"].median()), step=1000.0
        )

    with right:
        district = st.selectbox(
            "District", sorted(df[df["region"] == region]["district"].unique())
        )
        year = st.number_input("Target year", min_value=2011, max_value=2030, value=2025)
        exchange_rate_typical = st.number_input(
            "Exchange rate (typical)",
            value=float(df["exchange_rate_typical"].median()),
            step=10.0,
        )
        food_price_critical = st.number_input(
            "Food price critical",
            value=float(df["food_price_critical"].median()),
            step=0.1,
        )
        cpi_communication = st.number_input(
            "CPI · Communication", value=float(df["cpi_communication"].median()), step=0.1
        )
        cpi_housing_utilities = st.number_input(
            "CPI · Housing & Utilities",
            value=float(df["cpi_housing_utilities"].median()),
            step=0.1,
        )

    run_forecast = st.button("Run Forecast", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not run_forecast:
        st.markdown(
            """
            <div class="app-card" style="text-align:center;">
              <div class="eyebrow">Ready</div>
              <p style="margin:6px 0 0;">Adjust the scenario above then click
              <b>Run Forecast</b> to simulate the Food Price Index.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        history_df = (
            df[(df["region"] == region) & (df["district"] == district)][
                ["Date", "food_price_index"]
            ]
            .dropna()
            .sort_values("Date")
        )
        history_df["Date"] = pd.to_datetime(history_df["Date"])

        last_hist_date = history_df["Date"].max()
        target_date = pd.to_datetime(f"{year}-{month:02d}-01")

        if target_date <= last_hist_date:
            st.error(
                f"Target date must be after the last historical observation "
                f"({last_hist_date.date()})."
            )
            return

        exogenous_inputs = {
            "region": region,
            "district": district,
            "market_price_maize": market_price_maize,
            "market_price_rice": market_price_rice,
            "market_price_sorghum": market_price_sorghum,
            "market_price_oil": market_price_oil,
            "population": population,
            "exchange_rate_typical": exchange_rate_typical,
            "food_price_critical": food_price_critical,
            "cpi_communication": cpi_communication,
            "cpi_housing_utilities": cpi_housing_utilities,
        }

        forecast_df = recursive_forecast_hybrid(
            history_df=history_df,
            exogenous_inputs=exogenous_inputs,
            start_date=last_hist_date + pd.DateOffset(months=1),
            target_date=target_date,
        )

        final_prediction = forecast_df.iloc[-1]["food_price_index"]
        latest_history = history_df.iloc[-1]["food_price_index"]
        delta = final_prediction - latest_history
        delta_pct = (delta / latest_history * 100) if latest_history else 0

        # ---------- Prediction highlight ----------
        direction = "▲" if delta >= 0 else "▼"
        delta_color = "#b91c1c" if delta >= 0 else "#047857"

        st.markdown(
            f"""
            <div class="prediction-card">
              <div class="label">Forecasted Food Price Index · {target_date.strftime("%b %Y")}</div>
              <div class="value">{final_prediction:.2f}</div>
              <div style="margin-top:8px;color:{delta_color};font-weight:600;">
                {direction} {abs(delta):.2f} ({delta_pct:+.1f}%) vs last observation
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- Forecast chart ----------
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Forecast Trajectory</div>',
            unsafe_allow_html=True,
        )

        hist_plot = history_df.tail(18).copy()
        hist_plot["type"] = "Historical"
        plot_df = pd.concat([hist_plot, forecast_df], ignore_index=True)

        fig = px.line(
            plot_df,
            x="Date",
            y="food_price_index",
            color="type",
            markers=True,
            color_discrete_map={"Historical": "#0ea5e9", "Predicted": "#10b981"},
            labels={"food_price_index": "Food Price Index", "Date": "Date"},
        )

        fig.update_layout(
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#0f172a", family="Inter, sans-serif"),
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0", color="#334155"),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", color="#334155"),
            legend=dict(
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=-0.2,
            ),
            margin=dict(l=16, r=16, t=20, b=16),
        )

        fig.update_traces(
            line=dict(width=3), selector=dict(name="Historical")
        )
        fig.update_traces(
            line=dict(width=3, dash="dot"),
            marker=dict(size=9, symbol="diamond"),
            selector=dict(name="Predicted"),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "<div class='eyebrow' style='margin-top:4px;'>Dashed line · model projection</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Forecast failed: {e}")
