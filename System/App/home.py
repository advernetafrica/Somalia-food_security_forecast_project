import base64
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from map import render_map


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"


# Interpretation bands for the Food Price Index.
# Tuned to the dataset's empirical distribution (p25 ≈ 0.89, p75 ≈ 1.35, max ≈ 3.05).
FPI_BANDS = [
    ("Below Baseline", 0.00, 0.90, "#0ea5e9", "Prices below the long-term norm — low stress."),
    ("Normal",         0.90, 1.20, "#10b981", "Typical market conditions — stable access to staples."),
    ("Elevated",       1.20, 1.60, "#facc15", "Early-warning territory — watch trends and commodities."),
    ("Stressed",       1.60, 2.00, "#f97316", "Intervention recommended — household purchasing power eroding."),
    ("Critical",       2.00, 99.0, "#ef4444", "Acute food-insecurity risk — urgent response needed."),
]


def _img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def show_home_page(df):
    st.markdown(
        """
        <div class="app-hero">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <span class="app-badge">Live Dashboard</span>
            </div>
            <h1>Somalia Food Security Dashboard</h1>
            <p>A decision-support tool that turns market prices, climate signals and socio-economic
            indicators into an at-a-glance read on food-security risk across Somalia's regions and districts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _kpi_strip(df)
    _about_card()
    _interpretation_card(df)
    _impact_card()
    _map_card(df)
    _guide_card()

    st.markdown(
        """
        <div class="app-footer">
            Data window · Jan 2015 – Dec 2024 &nbsp;·&nbsp; Dashboard v1.0
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
def _kpi_strip(df):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric("Commodities Tracked", "4", "Maize · Rice · Sorghum · Oil", ASSETS_DIR / "commodities.png")
    with c2:
        _metric("Regions Covered", f"{df['region'].nunique()}", "Administrative regions",
                ASSETS_DIR / "regions.png")
    with c3:
        _metric("Districts Monitored", f"{df['district'].nunique()}", "Active market locations",
                ASSETS_DIR / "markets.png")
    with c4:
        avg = df["food_price_index"].mean()
        _metric("Mean Food Price Index", f"{avg:.2f}",
                f"Historical average · {_band_name(avg)}", ASSETS_DIR / "gauge.png")


def _metric(label: str, value: str, sub: str, icon_path: Path):
    icon = _img_b64(icon_path)
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-icon">
            <img src="data:image/png;base64,{icon}" />
          </div>
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# About the project
# ---------------------------------------------------------------------------
def _about_card():
    st.markdown(
        """
        <div class="app-card">
            <div class="section-title">About the Project</div>
            <p>
              Somalia experiences some of the most persistent food-insecurity pressures in the world —
              shaped by conflict, recurrent drought, currency volatility and fragile supply chains.
              This platform consolidates <b>price, climate, conflict and macro-economic signals</b> from 2015
              onward into a single <b>Food Price Index (FPI)</b> — a normalized measure of how expensive a
              typical household food basket is relative to historical baselines.
            </p>
            <p>
              The goal is practical: surface <b>which districts are under stress</b>, help responders
              <b>prioritize interventions</b>, and let analysts <b>simulate shocks</b> before they materialise.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Interpretation scale + live risk distribution
# ---------------------------------------------------------------------------
def _interpretation_card(df: pd.DataFrame):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How to Read the Food Price Index</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="margin-top:-4px;">
          FPI is an <b>index</b>, not a currency amount. A value of
          <b>1.00</b> anchors to the historical norm; values <b>above</b> 1.0 signal rising food costs
          relative to that norm, values <b>below</b> 1.0 signal easing.
          The bands below summarise what each range typically means for households on the ground.
        </p>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.55, 0.45], gap="large")

    # Left: interpretation bands
    with left:
        st.markdown(
            """
            <div style="display:flex;flex-direction:column;gap:8px;margin-top:6px;">
            """
            + "".join(_band_row(b) for b in FPI_BANDS) +
            """
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Right: current district distribution across bands
    with right:
        dist = _band_distribution(df)
        fig = _donut_from_bands(dist)
        st.plotly_chart(fig, use_container_width=True)
        total_districts = int(dist["count"].sum())
        critical_n = int(dist[dist["band"].isin(["Stressed", "Critical"])]["count"].sum())
        st.markdown(
            f"""
            <div class='eyebrow' style='text-align:center;'>
                {total_districts} districts analysed &nbsp;·&nbsp;
                <span style='color:#b45309;'>{critical_n} currently at Stressed or Critical level</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _band_row(band):
    name, lo, hi, color, desc = band
    range_label = f"{lo:.2f} – {hi:.2f}" if hi < 10 else f"≥ {lo:.2f}"
    return f"""
    <div style="display:flex;align-items:flex-start;gap:12px;padding:10px 12px;
                border:1px solid #e2e8f0;border-radius:10px;background:#f8fafc;">
      <div style="width:6px;align-self:stretch;border-radius:3px;background:{color};flex-shrink:0;"></div>
      <div style="flex:1;">
        <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;">
          <span style="font-weight:700;color:#0f172a;">{name}</span>
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;
                       color:{color};font-weight:700;">{range_label}</span>
        </div>
        <div style="color:#475569;font-size:0.88rem;line-height:1.4;margin-top:2px;">{desc}</div>
      </div>
    </div>
    """


def _band_distribution(df: pd.DataFrame) -> pd.DataFrame:
    latest = df["Date"].max() if pd.api.types.is_datetime64_any_dtype(df["Date"]) \
        else pd.to_datetime(df["Date"]).max()
    recent = df[pd.to_datetime(df["Date"]) >= pd.to_datetime(latest) - pd.DateOffset(months=6)]
    per_district = recent.groupby("district")["food_price_index"].mean()

    rows = []
    for name, lo, hi, color, _ in FPI_BANDS:
        count = int(((per_district >= lo) & (per_district < hi)).sum())
        rows.append({"band": name, "count": count, "color": color})
    return pd.DataFrame(rows)


def _donut_from_bands(dist: pd.DataFrame):
    fig = go.Figure(go.Pie(
        labels=dist["band"],
        values=dist["count"],
        hole=0.62,
        marker=dict(colors=dist["color"].tolist(), line=dict(color="#ffffff", width=2)),
        textinfo="label+value",
        textfont=dict(color="#0f172a", family="Inter, sans-serif", size=12),
        hovertemplate="<b>%{label}</b><br>%{value} districts<extra></extra>",
        sort=False,
    ))
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        margin=dict(l=6, r=6, t=6, b=6),
        height=300,
        annotations=[dict(
            text="<b>Current</b><br>district<br>risk mix",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#0f172a", size=13, family="Inter, sans-serif"),
        )],
    )
    return fig


def _band_name(value: float) -> str:
    for name, lo, hi, _c, _d in FPI_BANDS:
        if lo <= value < hi:
            return name
    return "Critical"


# ---------------------------------------------------------------------------
# Impact
# ---------------------------------------------------------------------------
def _impact_card():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Who This Helps</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        _impact_col(
            "Humanitarian Responders",
            "Prioritise districts crossing Stressed or Critical bands — target food-assistance "
            "programmes before markets fail.",
            "#10b981",
        )
    with c2:
        _impact_col(
            "Policymakers",
            "Quantify how exchange-rate and CPI shocks propagate into household food costs — "
            "stress-test policies before rollout.",
            "#0ea5e9",
        )
    with c3:
        _impact_col(
            "Market Analysts & NGOs",
            "Track commodity-price momentum and forecast the FPI months ahead to plan "
            "procurement and supply chains.",
            "#8b5cf6",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _impact_col(title: str, body: str, accent: str):
    st.markdown(
        f"""
        <div style="padding:16px 18px;border:1px solid #e2e8f0;border-radius:12px;
                    background:#f8fafc;border-top:3px solid {accent};height:100%;">
            <div style="font-weight:700;color:#0f172a;font-size:1rem;margin-bottom:6px;">{title}</div>
            <div style="color:#475569;font-size:0.92rem;line-height:1.55;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
def _map_card(df):
    st.markdown(
        '<div class="app-card" style="padding:20px 24px;">'
        '<div class="section-title">Geographic Distribution</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    render_map(df)


# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------
def _guide_card():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Navigate the Dashboard</div>', unsafe_allow_html=True)

    steps = [
        ("Home", "You're here — snapshot of the index, bands, and risk distribution.", "🏠"),
        ("Visualizations", "Trend, seasonality, regional distributions and district rankings.", "📊"),
        ("Predictions", "Configure a scenario and forecast the FPI month-by-month.", "📈"),
        ("Explainable AI", "Run shock scenarios and see which drivers move the index most.", "💡"),
    ]

    cols = st.columns(4)
    for (title, desc, icon), col in zip(steps, cols):
        with col:
            st.markdown(
                f"""
                <div style="padding:16px 18px;border:1px solid #e2e8f0;border-radius:12px;
                            background:#ffffff;height:100%;box-shadow:0 1px 2px rgba(15,23,42,0.04);">
                    <div style="font-size:1.6rem;margin-bottom:8px;">{icon}</div>
                    <div style="font-weight:700;color:#0f172a;font-size:0.98rem;">{title}</div>
                    <div style="color:#475569;font-size:0.86rem;line-height:1.5;margin-top:4px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
