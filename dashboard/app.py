"""
app.py — Options Pricing Library Dashboard
-------------------------------------------
Entry point for the Streamlit multi-page dashboard.

Run with:
    cd OptionsPricingLibrary
    streamlit run dashboard/app.py

Pages
-----
1. Home           — Live pricer + model comparison
2. Vol Surface    — SVI calibration, 3D surface, RND
3. Greek Surface  — Heatmaps for all 8 Greeks
4. Vega Matrix    — Pillar-by-pillar vol surface sensitivity
5. Model Risk     — Local vol vs BS for barriers, Heston vs BS for Asians
6. P&L Attribution — Taylor decomposition + backtest
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from options_lib import (BlackScholes, EuropeanOption, AmericanOption, 
                          MarketData, OptionType, Heston, HestonParams)
from options_lib.numerics import CrankNicolson, LongstaffSchwartz, HestonSimulator
from options_lib.market_data.option_chain import fetch_option_chain
from options_lib.market_data.vol_surface import calibrate_vol_surface, SVIParams, VolSurface

st.set_page_config(
    page_title="Options Pricer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;500;700;800&display=swap');

:root {
    --bg:        #0a0b0d;
    --surface:   #111318;
    --surface2:  #181c23;
    --border:    #1e2530;
    --accent:    #00d4b8;
    --accent2:   #7c6af7;
    --warn:      #f59e0b;
    --danger:    #ef4444;
    --text:      #e8eaf0;
    --muted:     #5a6278;
    --mono:      'DM Mono', monospace;
    --display:   'Syne', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--display);
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
header[data-testid="stHeader"] { display: none; }

[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 11px !important; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--mono) !important; font-size: 22px !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-family: var(--mono) !important; font-size: 12px !important; }

.stSelectbox > div > div, .stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
.stSlider > div > div { background: var(--border) !important; }
.stSlider > div > div > div > div { background: var(--accent) !important; }

.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-family: var(--mono) !important;
    letter-spacing: 0.05em !important;
    padding: 8px 20px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; font-family: var(--mono) !important; font-size: 12px !important; letter-spacing: 0.06em !important; padding: 12px 20px !important; border-radius: 0 !important; border-bottom: 2px solid transparent !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; background: transparent !important; }

code { font-family: var(--mono) !important; background: var(--surface2) !important; color: var(--accent) !important; padding: 2px 6px !important; border-radius: 4px !important; }
hr { border-color: var(--border) !important; }
.dataframe { background: var(--surface2) !important; color: var(--text) !important; font-family: var(--mono) !important; font-size: 12px !important; }
.dataframe th { background: var(--surface) !important; color: var(--accent) !important; }
.streamlit-expanderHeader { background: var(--surface2) !important; border: 1px solid var(--border) !important; }
.stAlert { border-radius: 6px !important; font-family: var(--mono) !important; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,19,24,1)',
    font=dict(family='DM Mono', color='#e8eaf0', size=11),
    xaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    yaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ── Sidebar navigation & Data Setup ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 28px;">
        <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:800;
                    color:#00d4b8; letter-spacing:-0.02em;">◈ OPTIONS</div>
        <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:800;
                    color:#e8eaf0; letter-spacing:-0.02em; margin-top:-4px;">PRICER</div>
        <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5a6278;
                    letter-spacing:0.12em; margin-top:8px;">QUANT RESEARCH SUITE</div>
    </div>
    <hr style="border-color:#1e2530; margin-bottom:20px;">
    """, unsafe_allow_html=True)

    st.markdown("**Navigate**")
    st.page_link("app.py",              label="⌂  Home — Live Pricer",       )
    st.page_link("pages/1_Vol_Surface.py",     label="◎  Vol Surface",       )
    st.page_link("pages/2_Greek_Surface.py",   label="△  Greek Surface",      )
    st.page_link("pages/3_Vega_Matrix.py",     label="⊞  Vega Matrix",       )
    st.page_link("pages/4_Model_Risk.py",      label="⚡  Model Risk",       )
    st.page_link("pages/5_PnL_Attribution.py", label="◑  P&L Attribution",  )

    st.markdown("<hr style='border-color:#1e2530;margin:20px 0;'>", unsafe_allow_html=True)
    
    st.markdown("### Market Data Source")
    data_mode = st.radio("Source", ["Synthetic (Demo)", "Live (yfinance)"])
    
    if data_mode == "Live (yfinance)":
        ticker = st.text_input("Ticker", value="SPY")
        live_r = st.number_input("Risk-free rate", value=0.05, step=0.005)
        live_q = st.number_input("Dividend yield", value=0.013, step=0.001)
        fetch_btn = st.button("⟳ Fetch & Calibrate")
    else:
        syn_spot = st.number_input("Spot (S₀)", value=100.0)
        syn_r    = st.number_input("Rate (r)", value=0.05)
        syn_q    = st.number_input("Div yield (q)", value=0.0)
        syn_vol  = st.number_input("ATM Volatility (σ)", value=0.20)
        syn_skew = st.slider("Vol Skew (ρ)", min_value=-0.9, max_value=0.1, value=-0.5)

# ── Data Pipelines ────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_live_surface(ticker, r, q):
    chain = fetch_option_chain(ticker, rate=r, div_yield=q, n_expiries=4)
    surf = calibrate_vol_surface(chain, verbose=False)
    return surf

def get_synthetic_surface(S, r, q, sigma, skew):
    slices = {
        '3M': SVIParams(a=sigma**2*0.9, b=0.12, rho=skew*0.9, m=0.0, sigma=0.10, expiry=0.25),
        '6M': SVIParams(a=sigma**2,     b=0.14, rho=skew,     m=0.0, sigma=0.12, expiry=0.50),
        '1Y': SVIParams(a=sigma**2*1.1, b=0.16, rho=skew*1.1, m=0.0, sigma=0.15, expiry=1.00),
    }
    forwards = {d: S * np.exp((r-q) * s.expiry) for d, s in slices.items()}
    return VolSurface(svi_slices=slices, forwards=forwards, spot=S, rate=r, ticker='DEMO')

@st.cache_data(show_spinner=False)
def calibrate_heston(_surface, S, r, q):
    initial = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.5)
    mkt_strikes, mkt_expiries, mkt_ivs = [], [], []
    for T_val in _surface.expiries:
        for K_val in np.linspace(S * 0.8, S * 1.2, 7):
            mkt_strikes.append(K_val)
            mkt_expiries.append(T_val)
            mkt_ivs.append(_surface.implied_vol(K_val, T_val))
            
    mkt_data = MarketData(spot=S, rate=r, div_yield=q)
    calibrated = Heston(initial).calibrate(
        np.array(mkt_strikes), np.array(mkt_expiries), np.array(mkt_ivs), mkt_data, initial, verbose=False
    )
    return calibrated.params

# Setup global market variables
surface = None
if data_mode == "Live (yfinance)":
    if 'fetch_btn' in locals() and fetch_btn:
        with st.spinner(f"Fetching {ticker} and calibrating models..."):
            surface = get_live_surface(ticker, live_r, live_q)
            S, r, q = surface.spot, live_r, live_q
            sigma = float(surface.svi_slices[list(surface.svi_slices.keys())[0]].implied_vol(np.array([0.0]))[0])
            st.session_state['live_data'] = (surface, S, r, q, sigma)
    elif 'live_data' in st.session_state:
        surface, S, r, q, sigma = st.session_state['live_data']
    else:
        st.info("Click 'Fetch & Calibrate' in the sidebar to load live market data.")
        st.stop()
else:
    S, r, q, sigma = syn_spot, syn_r, syn_q, syn_vol
    surface = get_synthetic_surface(S, r, q, sigma, syn_skew)

heston_params = calibrate_heston(surface, S, r, q)

# ── Home page UI ──────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800;
           letter-spacing:-0.03em; margin-bottom:4px;">
    Live Option Pricer
</h1>
<p style="font-family:'DM Mono',monospace; font-size:12px; color:#5a6278; margin-bottom:32px;">
    Multi-model pricing · Fully Calibrated Volatility Dynamics
</p>
""", unsafe_allow_html=True)

col_params, col_model, col_results = st.columns([1.2, 1, 1.8])

with col_params:
    st.markdown("**Instrument inputs**")
    S_disp = st.number_input("Spot (S₀) - From Data", value=float(S), disabled=True)
    K     = st.number_input("Strike (K)", value=float(S), step=1.0, format="%.1f")
    T     = st.number_input("Expiry (years)", value=1.0, step=0.25, min_value=0.01)
    opt_type = st.selectbox("Option type", ["Call", "Put"])

with col_model:
    st.markdown("**Pricing Models**")
    instrument_type = st.selectbox("Instrument", ["European", "American"])
    models_selected = st.multiselect(
        "Price with",
        ["Black-Scholes", "Heston", "LSMC", "Crank-Nicolson"],
        default=["Black-Scholes", "Heston"]
    )

    st.markdown("""<div style="margin-top:10px; font-size:12px; color:#00d4b8;"><b>✓ Heston Calibrated</b></div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px; color:#5a6278; font-family:'DM Mono';">
    v₀: {heston_params.v0:.4f} | κ: {heston_params.kappa:.2f}<br>
    v̄: {heston_params.v_bar:.4f} | ξ: {heston_params.xi:.2f}<br>
    ρ: {heston_params.rho:.2f}
    </div>
    """, unsafe_allow_html=True)

# ── Compute prices ────────────────────────────────────────────────────────────
otype = OptionType.CALL if opt_type == "Call" else OptionType.PUT
mkt   = MarketData(spot=S, rate=r, div_yield=q)
prices = {}

if "Black-Scholes" in models_selected:
    if instrument_type == "European":
        inst = EuropeanOption(K, T, otype)
        prices["Black-Scholes"] = BlackScholes(sigma=sigma).price(inst, mkt)
    else:
        fd = CrankNicolson(sigma=sigma, M=200, N=200)
        inst = AmericanOption(K, T, otype)
        prices["Black-Scholes (FD)"] = fd.price(inst, spot=S, r=r, q=q)

if "Heston" in models_selected:
    try:
        if instrument_type == "European":
            inst_eu = EuropeanOption(K, T, otype)
            prices["Heston (FFT)"] = Heston(heston_params).price(inst_eu, mkt)
        else:
            # The Magic Hook: Heston paths into Longstaff-Schwartz!
            sim = HestonSimulator(params=heston_params, n_paths=10000, n_steps=50, scheme='qe', seed=42)
            lsmc = LongstaffSchwartz(sigma=sigma, path_simulator=sim.as_lsmc_simulator())
            inst_am = AmericanOption(K, T, otype)
            res = lsmc.price(inst_am, spot=S, rate=r, div_yield=q)
            prices["Heston (LSMC)"] = res.price
    except Exception as e:
        st.warning(f"Heston error: {e}")

if "LSMC" in models_selected:
    try:
        # Standard GBM LSMC
        lsmc = LongstaffSchwartz(sigma=sigma, n_paths=10000, n_steps=50, seed=42)
        inst_am = AmericanOption(K, T, otype)
        result = lsmc.price(inst_am, spot=S, rate=r, div_yield=q)
        prices["BS LSMC"] = result.price
    except Exception as e:
        st.warning(f"LSMC error: {e}")

if "Crank-Nicolson" in models_selected:
    try:
        fd = CrankNicolson(sigma=sigma, M=200, N=200)
        inst_am = AmericanOption(K, T, otype)
        prices["Crank-Nicolson"] = fd.price(inst_am, spot=S, r=r, q=q)
    except Exception as e:
        st.warning(f"FD error: {e}")

bs_model = BlackScholes(sigma=sigma)
inst_ref  = EuropeanOption(K, T, otype)
bs_ref    = bs_model.price(inst_ref, mkt)

with col_results:
    st.markdown("**Prices**")
    if prices:
        pcols = st.columns(min(len(prices), 2))
        for idx, (model_name, price) in enumerate(prices.items()):
            with pcols[idx % 2]:
                st.metric(model_name, f"{price:.4f}")
    else:
        st.info("Select at least one model")

    st.markdown("**BS Greeks Reference**")
    g_cols = st.columns(4)
    with g_cols[0]: st.metric("Delta", f"{bs_model.delta(inst_ref, mkt):.4f}")
    with g_cols[1]: st.metric("Gamma", f"{bs_model.gamma(inst_ref, mkt):.4f}")
    with g_cols[2]: st.metric("Vega",  f"{bs_model.vega(inst_ref, mkt):.2f}")
    with g_cols[3]: st.metric("Theta", f"{bs_model.theta(inst_ref, mkt):.4f}/d")

st.markdown("---")

# ── Charts row ────────────────────────────────────────────────────────────────
chart1, chart2, chart3 = st.columns(3)

with chart1:
    spots = np.linspace(max(S * 0.5, 1), S * 1.5, 120)
    bs_prices  = [BlackScholes(sigma=sigma).price(EuropeanOption(K, T, otype), MarketData(s, r, q)) for s in spots]
    intrinsics = [max(s - K, 0) if opt_type == "Call" else max(K - s, 0) for s in spots]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=bs_prices, name="BS price", line=dict(color='#00d4b8', width=2.5)))
    fig.add_trace(go.Scatter(x=spots, y=intrinsics, name="Intrinsic", line=dict(color='#5a6278', width=1.5, dash='dot')))
    fig.add_vline(x=S, line_color='#7c6af7', line_dash='dash', line_width=1.5, annotation_text=f"S={S:.1f}", annotation_font_color='#7c6af7')
    fig.update_layout(**PLOTLY_LAYOUT, title="Price vs Spot", height=280, legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

with chart2:
    deltas = [BlackScholes(sigma=sigma).delta(EuropeanOption(K, T, otype), MarketData(s, r, q)) for s in spots]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=spots, y=deltas, name="Delta", line=dict(color='#00d4b8', width=2.5)))
    fig2.add_vline(x=S, line_color='#5a6278', line_dash='dash', line_width=1)
    fig2.update_layout(**PLOTLY_LAYOUT, title="Delta vs Spot", height=280, legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig2, use_container_width=True)

with chart3:
    sigmas = np.linspace(0.05, 0.80, 100)
    prices_vs_vol = [BlackScholes(sv).price(inst_ref, mkt) for sv in sigmas]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sigmas * 100, y=prices_vs_vol, name="Price", line=dict(color='#00d4b8', width=2.5)))
    fig3.add_vline(x=sigma * 100, line_color='#7c6af7', line_dash='dash', line_width=1.5)
    fig3.update_layout(**PLOTLY_LAYOUT, title="Price vs Implied Vol (%)", height=280, xaxis_title="Implied Vol (%)", legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig3, use_container_width=True)
