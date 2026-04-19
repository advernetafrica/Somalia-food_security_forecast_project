import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PLOT_LAYOUT = dict(
    plot_bgcolor="rgba(255,255,255,0)",
    paper_bgcolor="rgba(255,255,255,0)",
    font=dict(color="#0f172a", family="Inter, sans-serif", size=12),
    xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False, color="#334155",
               tickfont=dict(color="#475569")),
    yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False, color="#334155",
               tickfont=dict(color="#475569")),
    margin=dict(l=16, r=16, t=20, b=16),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        font=dict(color="#0f172a"),
    ),
    title=dict(text=""),
    hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#f8fafc", family="Inter, sans-serif")),
)

COMMODITY_PALETTE = {
    "Maize": "#10b981",
    "Rice": "#0ea5e9",
    "Sorghum": "#f59e0b",
    "Oil": "#8b5cf6",
}

PRICE_COLS = {
    "Maize": "market_price_maize",
    "Rice": "market_price_rice",
    "Sorghum": "market_price_sorghum",
    "Oil": "market_price_oil",
}

# Risk gradient for FPI (shared with home.py legend + map choropleth).
# Green = safe, red = critical. Stops anchored to the FPI bands on a
# 0.5 – 3.0 normalized axis.
FPI_COLORSCALE = [
    [0.00, "#0ea5e9"],   # Below baseline
    [0.16, "#10b981"],   # Normal (0.90)
    [0.28, "#10b981"],   # Normal upper (1.20)
    [0.44, "#facc15"],   # Elevated (1.60)
    [0.60, "#f97316"],   # Stressed (2.00)
    [1.00, "#ef4444"],   # Critical (3.00+)
]


def _fpi_band_color(value: float) -> str:
    """Pick a band colour for a scalar FPI — keeps box-plot tints consistent."""
    if value < 0.90:   return "#0ea5e9"
    if value < 1.20:   return "#10b981"
    if value < 1.60:   return "#facc15"
    if value < 2.00:   return "#f97316"
    return "#ef4444"


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def show_visualizations_page(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    st.markdown(
        """
        <div class="app-hero">
            <span class="app-badge">Explorer</span>
            <h1>Food Price Index · Explorer</h1>
            <p>Track, compare and rank Food Price Index dynamics across Somalia.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _kpi_strip(df)
    region, district, filtered = _filters_card(df)
    _fpi_trend_card(filtered, region, district)
    _seasonal_heatmap_card(filtered)
    _commodity_trends_card(filtered)
    _regional_distribution_card(df if region == "All Regions" else filtered)
    _district_ranking_card(df if region == "All Regions" else filtered)
    _raw_data_card(filtered)


# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
def _kpi_strip(df: pd.DataFrame):
    latest_date = df["Date"].max()
    last_12 = df[df["Date"] >= latest_date - pd.DateOffset(months=12)]
    prev_12 = df[
        (df["Date"] < latest_date - pd.DateOffset(months=12))
        & (df["Date"] >= latest_date - pd.DateOffset(months=24))
    ]

    latest_fpi = last_12.groupby("Date")["food_price_index"].mean().iloc[-1]
    yoy_change = 0.0
    if not prev_12.empty:
        yoy_change = latest_fpi - prev_12["food_price_index"].mean()

    peak_region_df = df.groupby("region")["food_price_index"].mean().sort_values(ascending=False)
    peak_region = peak_region_df.index[0]
    peak_val = peak_region_df.iloc[0]

    volatility = df.groupby("Date")["food_price_index"].mean().pct_change().std() * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Latest National FPI", f"{latest_fpi:.2f}", f"as of {latest_date.strftime('%b %Y')}")
    with c2:
        arrow = "▲" if yoy_change >= 0 else "▼"
        color = "#b45309" if yoy_change >= 0 else "#047857"
        _kpi(
            "12-mo Change",
            f"{arrow} {abs(yoy_change):.3f}",
            f"{'Rising' if yoy_change >= 0 else 'Easing'} vs prior year",
            value_color=color,
        )
    with c3:
        _kpi("Peak Region", peak_region, f"Mean FPI {peak_val:.2f}")
    with c4:
        _kpi("Monthly Volatility", f"{volatility:.2f}%", "National FPI month-over-month")


def _kpi(label: str, value: str, sub: str, value_color: str = "#0f172a"):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{value_color};">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def _filters_card(df: pd.DataFrame):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)

    tab_region, tab_district = st.tabs(["Region", "District"])
    with tab_region:
        region_options = ["All Regions"] + sorted(df["region"].unique().tolist())
        selected_region = st.selectbox("Region", region_options, key="viz_region")

    filtered = df if selected_region == "All Regions" else df[df["region"] == selected_region]

    with tab_district:
        district_options = ["All Districts"] + sorted(filtered["district"].unique().tolist())
        selected_district = st.selectbox("District", district_options, key="viz_district")

    if selected_district != "All Districts":
        filtered = filtered[filtered["district"] == selected_district]

    st.markdown(
        f"""
        <div class='eyebrow' style='margin-top:6px;'>
            Showing &nbsp;·&nbsp; {selected_region} &nbsp;·&nbsp;
            {selected_district} &nbsp;·&nbsp; {len(filtered):,} records
            &nbsp;·&nbsp; {filtered['Date'].min().strftime('%b %Y')} →
            {filtered['Date'].max().strftime('%b %Y')}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    return selected_region, selected_district, filtered


# ---------------------------------------------------------------------------
# Main FPI trend + 3-month rolling mean
# ---------------------------------------------------------------------------
def _fpi_trend_card(filtered: pd.DataFrame, region: str, district: str):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Food Price Index · Trend</div>', unsafe_allow_html=True)

    ts = filtered.groupby("Date")["food_price_index"].mean().reset_index()
    ts["Rolling 3-mo"] = ts["food_price_index"].rolling(3, min_periods=1).mean()

    fig = px.area(
        ts, x="Date", y="food_price_index",
        labels={"food_price_index": "Food Price Index", "Date": "Date"},
    )
    fig.update_traces(
        line=dict(width=0.8, color="#10b981"),
        fillcolor="rgba(16, 185, 129, 0.18)",
        name="FPI",
        showlegend=True,
    )
    fig.add_scatter(
        x=ts["Date"], y=ts["Rolling 3-mo"],
        mode="lines", name="3-mo rolling mean",
        line=dict(color="#0284c7", width=2.5, dash="dot"),
    )
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"<div class='eyebrow'>{region} · {district} &nbsp;·&nbsp; "
        f"shaded area shows observed FPI; dotted line shows 3-month trend</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Seasonal heatmap (year × month)
# ---------------------------------------------------------------------------
def _seasonal_heatmap_card(filtered: pd.DataFrame):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Seasonality · Year × Month</div>', unsafe_allow_html=True)

    pivot = (
        filtered.assign(
            year=filtered["Date"].dt.year,
            month=filtered["Date"].dt.month,
        )
        .groupby(["year", "month"])["food_price_index"]
        .mean()
        .reset_index()
        .pivot(index="year", columns="month", values="food_price_index")
    )
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_labels[m - 1] for m in pivot.columns]

    fig = px.imshow(
        pivot,
        color_continuous_scale=FPI_COLORSCALE,
        zmin=0.5, zmax=3.0,
        aspect="auto",
        labels=dict(x="Month", y="Year", color="FPI"),
        text_auto=".2f",
    )
    fig.update_traces(textfont=dict(color="#0f172a", size=11))
    fig.update_layout(**{**PLOT_LAYOUT, "margin": dict(l=16, r=16, t=20, b=16)})
    fig.update_coloraxes(colorbar=dict(
        tickfont=dict(color="#334155"), title=dict(font=dict(color="#334155"))
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<div class='eyebrow'>Darker = higher FPI. Reveals months of consistent stress.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Commodity trends
# ---------------------------------------------------------------------------
def _commodity_trends_card(filtered: pd.DataFrame):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Commodity Prices · Monthly Median</div>', unsafe_allow_html=True)

    check_cols = st.columns(4)
    selected_labels = []
    for i, label in enumerate(PRICE_COLS.keys()):
        with check_cols[i]:
            if st.checkbox(label, value=True, key=f"viz_cb_{label}"):
                selected_labels.append(label)

    if not selected_labels:
        st.info("Select at least one commodity.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    cols = [PRICE_COLS[l] for l in selected_labels]
    melted = filtered.melt(
        id_vars=["Date"], value_vars=cols,
        var_name="Commodity", value_name="Price",
    )
    melted["Commodity"] = melted["Commodity"].str.replace("market_price_", "").str.title()
    median_df = melted.groupby(["Date", "Commodity"])["Price"].median().reset_index()

    fig = px.line(
        median_df, x="Date", y="Price", color="Commodity", markers=False,
        color_discrete_map=COMMODITY_PALETTE,
        labels={"Price": "Market Price (SOS)", "Date": "Date"},
    )
    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FPI distribution by region (box plot)
# ---------------------------------------------------------------------------
def _regional_distribution_card(df: pd.DataFrame):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">FPI Distribution · by Region</div>', unsafe_allow_html=True)

    medians = df.groupby("region")["food_price_index"].median().sort_values(ascending=True)
    order = medians.index.tolist()
    color_map = {r: _fpi_band_color(v) for r, v in medians.items()}

    fig = px.box(
        df, x="food_price_index", y="region",
        color="region",
        category_orders={"region": order},
        color_discrete_map=color_map,
        labels={"food_price_index": "Food Price Index", "region": ""},
        points=False,
    )
    fig.update_traces(marker=dict(opacity=0.7), line=dict(width=1.3))
    fig.update_layout(
        **{**PLOT_LAYOUT,
           "height": max(360, 22 * len(order) + 120),
           "showlegend": False},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<div class='eyebrow'>Boxes show the middle 50% of each region's FPI observations.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Top / bottom districts
# ---------------------------------------------------------------------------
def _district_ranking_card(df: pd.DataFrame):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">District Ranking · Mean FPI</div>', unsafe_allow_html=True)

    mode = st.radio(
        "", options=["Highest 10", "Lowest 10"],
        horizontal=True, key="viz_rank_mode", label_visibility="collapsed",
    )

    agg = df.groupby("district")["food_price_index"].mean().reset_index()
    ascending = (mode == "Lowest 10")
    ranked = agg.sort_values("food_price_index", ascending=ascending).head(10)
    ranked = ranked.sort_values("food_price_index", ascending=not ascending)  # flip for bar

    bar_colors = [_fpi_band_color(v) for v in ranked["food_price_index"]]
    fig = px.bar(
        ranked, x="food_price_index", y="district", orientation="h",
        labels={"food_price_index": "Mean Food Price Index", "district": ""},
        text=ranked["food_price_index"].round(2),
    )
    fig.update_traces(
        marker_color=bar_colors,
        marker_line_color="#0f172a",
        marker_line_width=0.3,
        textposition="outside",
        textfont=dict(color="#0f172a", size=11),
    )
    fig.update_layout(**{**PLOT_LAYOUT, "height": 420, "showlegend": False})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
def _raw_data_card(filtered: pd.DataFrame):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Raw Data</div>', unsafe_allow_html=True)

    show_data = st.checkbox("Show underlying records", value=False, key="viz_show_raw")
    if show_data:
        display_df = filtered.copy()
        display_df = display_df.drop(columns=["Unnamed: 0", "adm2_name"], errors="ignore")
        display_df = display_df.reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, height=380)
    else:
        st.markdown(
            "<div class='eyebrow'>Tick the box above to inspect the filtered records.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
