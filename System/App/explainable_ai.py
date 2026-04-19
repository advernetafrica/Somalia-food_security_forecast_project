import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from inference import predict_hybrid_residual, region_le, district_le


# Presets for shock scenarios — multipliers applied to baseline medians.
PRESETS = {
    "Baseline": dict(basket_mult=1.00, cpi_mult=1.00, critical=2),
    "Moderate Shock": dict(basket_mult=1.20, cpi_mult=1.08, critical=20),
    "Severe Shock": dict(basket_mult=1.55, cpi_mult=1.18, critical=55),
}

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
        orientation="h", yanchor="bottom", y=-0.25,
        font=dict(color="#0f172a"),
    ),
    title=dict(text=""),
    hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#f8fafc")),
)


def show_explainable_ai_page(df):
    st.markdown(
        """
        <div class="app-hero">
            <span class="app-badge">Explainability</span>
            <h1>What Moves the Food Price Index?</h1>
            <p>Test shock scenarios, compare against a baseline, and see which
            drivers contribute most to the projected Food Price Index.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        _render(df)
    except Exception as e:
        st.error(f"Analysis failed: {e}")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
def _render(df: pd.DataFrame):
    baselines = _baseline_medians(df)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">1 · Choose a Scenario</div>', unsafe_allow_html=True)

    preset_cols = st.columns(3)
    for i, name in enumerate(PRESETS.keys()):
        with preset_cols[i]:
            if st.button(name, use_container_width=True, key=f"preset_{name}"):
                st.session_state["xai_basket"] = PRESETS[name]["basket_mult"]
                st.session_state["xai_cpi"] = PRESETS[name]["cpi_mult"]
                st.session_state["xai_crit"] = PRESETS[name]["critical"]

    region_col, district_col = st.columns(2)
    with region_col:
        region = st.selectbox("Region", region_le.classes_, key="whatif_region")
    with district_col:
        district = st.selectbox("District", district_le.classes_, key="whatif_district")

    s1, s2, s3 = st.columns(3)
    with s1:
        basket_mult = st.slider(
            "Market Basket Multiplier",
            min_value=0.5, max_value=2.0,
            value=st.session_state.get("xai_basket", 1.0),
            step=0.05, key="xai_basket",
            help="Scales all four staple prices (maize, rice, sorghum, oil).",
        )
    with s2:
        cpi_mult = st.slider(
            "CPI Multiplier",
            min_value=0.8, max_value=1.5,
            value=st.session_state.get("xai_cpi", 1.0),
            step=0.02, key="xai_cpi",
            help="Scales cost-of-living pressure (housing + communication CPI).",
        )
    with s3:
        critical_level = st.slider(
            "Critical Food Stress Level",
            min_value=0, max_value=70,
            value=int(st.session_state.get("xai_crit", 2)),
            step=1, key="xai_crit",
            help="IPC-style critical stress indicator. Higher = more households at risk.",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Build scenario + baseline inputs
    scenario = _build_input(baselines, region, district, basket_mult, cpi_mult, critical_level)
    baseline = _build_input(baselines, region, district, 1.0, 1.0, 2)

    scenario_pred = predict_hybrid_residual(scenario, seed=f"{region}|{district}|scenario")
    baseline_pred = predict_hybrid_residual(baseline, seed=f"{region}|{district}|baseline")
    delta = scenario_pred - baseline_pred
    delta_pct = (delta / baseline_pred * 100) if baseline_pred else 0

    # --- Prediction highlight ---
    arrow = "▲" if delta >= 0 else "▼"
    delta_color = "#b91c1c" if delta >= 0 else "#047857"
    st.markdown(
        f"""
        <div class="prediction-card">
          <div class="label">Projected Food Price Index · {region} · {district}</div>
          <div class="value">{scenario_pred:.2f}</div>
          <div style="margin-top:8px;color:{delta_color};font-weight:600;">
            {arrow} {abs(delta):.3f} ({delta_pct:+.1f}%) vs baseline ({baseline_pred:.2f})
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Driver contribution ---
    _driver_contribution_card(
        baselines, region, district, basket_mult, cpi_mult, critical_level, baseline_pred
    )

    # --- Scenario comparison ---
    _scenario_comparison_card(baselines, region, district)

    # --- Sensitivity sweep ---
    _sensitivity_card(baselines, region, district, basket_mult, cpi_mult, critical_level)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _baseline_medians(df: pd.DataFrame) -> dict:
    return {
        "maize": float(df["market_price_maize"].median()),
        "rice": float(df["market_price_rice"].median()),
        "sorghum": float(df["market_price_sorghum"].median()),
        "oil": float(df["market_price_oil"].median()),
        "population": float(df["population"].median()),
        "exchange_rate": float(df["exchange_rate_typical"].median()),
        "cpi_comm": float(df["cpi_communication"].median()),
        "cpi_housing": float(df["cpi_housing_utilities"].median()),
        "rolling_mean_3": float(df["food_price_index"].tail(3).mean()),
    }


def _build_input(
    baselines: dict, region: str, district: str,
    basket_mult: float, cpi_mult: float, critical: float,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "region": region,
        "district": district,
        "market_price_maize": baselines["maize"] * basket_mult,
        "market_price_rice": baselines["rice"] * basket_mult,
        "market_price_sorghum": baselines["sorghum"] * basket_mult,
        "market_price_oil": baselines["oil"] * basket_mult,
        "population": baselines["population"],
        "exchange_rate_typical": baselines["exchange_rate"],
        "food_price_critical": critical,
        "cpi_communication": baselines["cpi_comm"] * cpi_mult,
        "cpi_housing_utilities": baselines["cpi_housing"] * cpi_mult,
        "food_price_index_rolling_mean_3": baselines["rolling_mean_3"],
    }])


# ---------------------------------------------------------------------------
# Driver contribution
# ---------------------------------------------------------------------------
def _driver_contribution_card(
    baselines, region, district,
    basket_mult, cpi_mult, critical_level, baseline_pred,
):
    """Toggle one driver at a time from baseline → current, measure delta."""
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">2 · Driver Contribution to the Scenario</div>',
        unsafe_allow_html=True,
    )

    drivers = [
        ("Market Basket", basket_mult, 1.0, 1.0, 2),
        ("CPI", 1.0, cpi_mult, 1.0, 2),
        ("Critical Stress", 1.0, 1.0, 1.0, critical_level),
    ]

    rows = []
    # First driver toggles basket, second toggles cpi, third toggles critical
    for label, b_mult, c_mult, _placeholder, crit in drivers:
        test = _build_input(baselines, region, district, b_mult, c_mult, crit)
        pred = predict_hybrid_residual(test, seed=f"{region}|{district}|driver|{label}")
        rows.append({"Driver": label, "Δ FPI vs Baseline": pred - baseline_pred})

    contrib_df = pd.DataFrame(rows).sort_values("Δ FPI vs Baseline")

    colors = [
        "#10b981" if v <= 0 else "#ef4444"
        for v in contrib_df["Δ FPI vs Baseline"].tolist()
    ]
    fig = go.Figure(go.Bar(
        x=contrib_df["Δ FPI vs Baseline"],
        y=contrib_df["Driver"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in contrib_df["Δ FPI vs Baseline"]],
        textposition="outside",
        textfont=dict(color="#0f172a"),
        hovertemplate="<b>%{y}</b><br>Δ FPI = %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(**{**PLOT_LAYOUT, "height": 260, "showlegend": False})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='eyebrow'>Each bar isolates one driver: FPI when only that driver "
        "is set to the scenario level and the others stay at baseline.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Scenario comparison
# ---------------------------------------------------------------------------
def _scenario_comparison_card(baselines, region, district):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">3 · Scenario Comparison</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for name, params in PRESETS.items():
        inp = _build_input(
            baselines, region, district,
            params["basket_mult"], params["cpi_mult"], params["critical"],
        )
        pred = predict_hybrid_residual(inp, seed=f"{region}|{district}|cmp|{name}")
        rows.append({"Scenario": name, "FPI": pred})

    cmp_df = pd.DataFrame(rows)
    palette = {"Baseline": "#0ea5e9", "Moderate Shock": "#f59e0b", "Severe Shock": "#ef4444"}

    fig = px.bar(
        cmp_df, x="Scenario", y="FPI",
        color="Scenario", color_discrete_map=palette,
        text=cmp_df["FPI"].round(3),
    )
    fig.update_traces(textposition="outside", textfont=dict(color="#0f172a", size=12))
    fig.update_layout(**{**PLOT_LAYOUT, "height": 320, "showlegend": False})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='eyebrow'>All three preset scenarios evaluated for the current "
        "region + district selection.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sensitivity sweep
# ---------------------------------------------------------------------------
def _sensitivity_card(baselines, region, district, basket_mult, cpi_mult, critical_level):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">4 · Sensitivity Sweep</div>',
        unsafe_allow_html=True,
    )

    driver = st.selectbox(
        "Sweep a driver",
        options=["Market Basket", "CPI", "Critical Stress"],
        key="xai_sweep_driver",
    )

    if driver == "Market Basket":
        xs = np.linspace(0.5, 2.0, 25)
        label = "Basket multiplier"
        make_input = lambda v: _build_input(baselines, region, district, v, cpi_mult, critical_level)
        current = basket_mult
    elif driver == "CPI":
        xs = np.linspace(0.8, 1.5, 25)
        label = "CPI multiplier"
        make_input = lambda v: _build_input(baselines, region, district, basket_mult, v, critical_level)
        current = cpi_mult
    else:
        xs = np.linspace(0, 70, 25)
        label = "Critical stress level"
        make_input = lambda v: _build_input(baselines, region, district, basket_mult, cpi_mult, v)
        current = float(critical_level)

    preds = [
        predict_hybrid_residual(make_input(v), seed=f"{region}|{district}|sweep|{driver}|{v:.3f}")
        for v in xs
    ]

    fig = px.line(
        x=xs, y=preds, markers=True,
        labels={"x": label, "y": "Predicted FPI"},
    )
    fig.update_traces(line=dict(width=2.8, color="#10b981"), marker=dict(size=6, color="#059669"))
    fig.add_vline(
        x=current, line_dash="dash", line_color="#0ea5e9",
        annotation_text="Current", annotation_font_color="#0ea5e9",
    )
    fig.update_layout(**{**PLOT_LAYOUT, "height": 340})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='eyebrow'>Sweeps one driver across its full range while the "
        "others stay at their current scenario values.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
