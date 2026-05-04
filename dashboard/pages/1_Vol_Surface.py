"""
pages/1_Vol_Surface.py  —  Updated: adds Heston calibration

KEY CHANGE: After SVI calibration, a "Calibrate Heston to surface" button
fits v0 kappa v_bar xi rho to the SVI-implied IV grid and stores them in
st.session_state['heston_params']. All other pages read from that key.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from options_lib.market_data.option_chain import fetch_option_chain
from options_lib.market_data.vol_surface import calibrate_vol_surface, SVIParams, VolSurface
from options_lib.models.heston import Heston, HestonParams
from options_lib.instruments.base import MarketData

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,19,24,1)',
    font=dict(family='DM Mono', color='#e8eaf0', size=11),
    xaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    yaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    margin=dict(l=40, r=20, t=50, b=40),
)
COLORS = ['#00d4b8', '#7c6af7', '#f59e0b', '#ef4444', '#10b981', '#3b82f6']

st.markdown("""
<h1 style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
           letter-spacing:-0.02em;margin-bottom:4px;">Volatility Surface</h1>
<p style="font-family:'DM Mono',monospace;font-size:12px;color:#5a6278;margin-bottom:28px;">
    SVI calibration · Heston calibration · Arbitrage checks · Risk-neutral density
</p>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Data source")
    mode = st.radio("Source", ["Live (yfinance)", "Synthetic (demo)"])
    if mode == "Live (yfinance)":
        ticker    = st.text_input("Ticker", value="SPY")
        rate      = st.number_input("Rate", value=0.05, step=0.005, format="%.3f")
        div_q     = st.number_input("Div yield", value=0.013, step=0.001, format="%.3f")
        n_exp     = st.slider("Max expiries", 2, 8, 4)
        fetch_btn = st.button("⟳  Fetch & Calibrate SVI")
    else:
        spot_s    = st.number_input("Spot", value=100.0)
        rate      = st.number_input("Rate", value=0.05, step=0.005, format="%.3f")
        fetch_btn = True

    st.markdown("---")
    st.markdown("### Heston calibration")
    st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:11px;color:#5a6278;margin-bottom:8px;">
    Fits v₀ κ v̄ ξ ρ to the SVI surface above.
    Calibrated params are shared with Home · Greek Surface · Model Risk pages.
    </div>""", unsafe_allow_html=True)
    calibrate_heston_btn  = st.button("⟳  Calibrate Heston to surface")
    show_heston_overlay   = st.checkbox("Show Heston smile overlay", value=True)

    st.markdown("---")
    show_3d   = st.checkbox("3D Surface",           value=True)
    show_smile = st.checkbox("IV Smiles",            value=True)
    show_rnd  = st.checkbox("Risk-Neutral Density",  value=True)
    show_arb  = st.checkbox("Arbitrage Checks",      value=True)

# ── Session state init ────────────────────────────────────────────────────────
if 'heston_params' not in st.session_state:
    st.session_state['heston_params'] = None
if 'vol_surface' not in st.session_state:
    st.session_state['vol_surface'] = None

# ── Build / fetch surface ─────────────────────────────────────────────────────
def make_synthetic(spot=100.0, r=0.05):
    slices = {
        '1M': SVIParams(0.012, 0.10, -0.45, 0.0, 0.08, 1/12),
        '3M': SVIParams(0.016, 0.12, -0.50, 0.0, 0.10, 0.25),
        '6M': SVIParams(0.019, 0.14, -0.55, 0.0, 0.12, 0.50),
        '1Y': SVIParams(0.022, 0.16, -0.60, 0.0, 0.15, 1.00),
        '2Y': SVIParams(0.026, 0.18, -0.62, 0.0, 0.18, 2.00),
    }
    fwd = {d: spot * np.exp(r * s.expiry) for d, s in slices.items()}
    return VolSurface(slices, fwd, spot, r, 'DEMO')

@st.cache_data(ttl=300)
def fetch_and_calibrate(ticker, rate, div_q, n_exp):
    chain   = fetch_option_chain(ticker, rate=rate, div_yield=div_q,
                                  n_expiries=n_exp, min_volume=5, max_spread_pct=0.40)
    surface = calibrate_vol_surface(chain, verbose=False)
    return chain, surface

surface = None
chain   = None

if mode == "Synthetic (demo)":
    surface = make_synthetic(spot_s, rate)
    st.session_state['vol_surface'] = surface
elif mode == "Live (yfinance)" and fetch_btn:
    with st.spinner(f"Fetching {ticker} and calibrating SVI..."):
        try:
            chain, surface = fetch_and_calibrate(ticker, rate, div_q, n_exp)
            st.session_state['vol_surface'] = surface
            st.success(f"SVI calibrated: {len(surface.svi_slices)} slices from {len(chain.quotes)} quotes.")
        except Exception as e:
            st.error(f"Error: {e}")

if surface is None and st.session_state['vol_surface'] is not None:
    surface = st.session_state['vol_surface']

if surface is None:
    st.info("Click **Fetch & Calibrate SVI** or select Synthetic demo.")
    st.stop()

spot     = surface.spot
dates    = surface.expiry_dates
expiries = surface.expiries

# ── Heston calibration ────────────────────────────────────────────────────────
def calibrate_heston_to_surface(surface):
    """
    Correct calibration flow:
      1. SVI surface gives smooth, arb-free IVs at any (K, T)
      2. Sample a dense grid from the SVI surface
      3. Fit Heston to that IV grid via L-BFGS-B

    This avoids fitting Heston directly to noisy raw quotes.
    """
    spot = surface.spot
    rate = surface.rate
    mkt  = MarketData(spot=spot, rate=rate)
    strikes_all, expiries_all, ivs_all = [], [], []

    for date, svi in surface.svi_slices.items():
        T   = svi.expiry
        F   = surface.forwards.get(date, spot * np.exp(rate * T))
        k_g = np.linspace(-0.25, 0.25, 10)
        K_g = F * np.exp(k_g)
        iv_g = svi.implied_vol(k_g)
        for K, iv in zip(K_g, iv_g):
            if 0.01 < iv < 3.0:
                strikes_all.append(float(K))
                expiries_all.append(float(T))
                ivs_all.append(float(iv))

    atm_iv = float(surface.svi_slices[dates[-1]].implied_vol(np.array([0.0]))[0])
    init   = HestonParams(v0=atm_iv**2, kappa=2.0, v_bar=atm_iv**2, xi=0.30, rho=-0.50)
    tpl    = Heston(init)
    cal    = tpl.calibrate(
        market_strikes  = np.array(strikes_all),
        market_expiries = np.array(expiries_all),
        market_ivs      = np.array(ivs_all),
        market_data     = mkt,
        initial_params  = init,
        verbose         = False,
    )
    return cal.params

if calibrate_heston_btn:
    with st.spinner("Calibrating Heston (~30s)..."):
        try:
            p = calibrate_heston_to_surface(surface)
            st.session_state['heston_params'] = p
            st.success(
                f"✓ Heston calibrated — v₀={p.v0:.4f}  κ={p.kappa:.3f}  "
                f"v̄={p.v_bar:.4f}  ξ={p.xi:.3f}  ρ={p.rho:.3f}  "
                f"Feller={'✓' if p.feller_satisfied else '✗'}"
            )
        except Exception as e:
            st.error(f"Heston calibration failed: {e}")

# Status banner
if st.session_state['heston_params'] is not None:
    p = st.session_state['heston_params']
    st.markdown(f"""
    <div style="background:#0f2a1e;border:1px solid #10b981;border-radius:6px;
                padding:10px 14px;margin-bottom:12px;font-family:'DM Mono',monospace;font-size:11px;">
        <span style="color:#10b981;font-weight:500;">◈ Heston calibrated to this surface</span>
        <span style="color:#5a6278"> — </span>
        v₀=<span style="color:#00d4b8">{p.v0:.4f}</span>  
        κ=<span style="color:#00d4b8">{p.kappa:.3f}</span>  
        v̄=<span style="color:#00d4b8">{p.v_bar:.4f}</span>  
        ξ=<span style="color:#00d4b8">{p.xi:.3f}</span>  
        ρ=<span style="color:#00d4b8">{p.rho:.3f}</span>
        <span style="color:#5a6278"> — shared with Home · Greek Surface · Model Risk</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Heston not calibrated. Click **Calibrate Heston to surface** in the sidebar.")

# ── Summary metrics ────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
cal_check  = surface.check_calendar_arbitrage()
but_check  = surface.check_butterfly_arbitrage()
atm_short  = float(surface.svi_slices[dates[0]].implied_vol(np.array([0.0]))[0])
atm_long   = float(surface.svi_slices[dates[-1]].implied_vol(np.array([0.0]))[0])
m1.metric("Spot",            f"{spot:.2f}")
m2.metric("ATM Vol (near)",  f"{atm_short:.1%}")
m3.metric("ATM Vol (far)",   f"{atm_long:.1%}")
m4.metric("Skew ρ (near)",   f"{surface.svi_slices[dates[0]].rho:.3f}")
m5.metric("Slices",          f"{len(dates)}")
st.markdown("---")

# ── 3D Surface ────────────────────────────────────────────────────────────────
if show_3d:
    st.markdown("### 3D Implied Volatility Surface")
    k_g = np.linspace(-0.4, 0.4, 60)
    T_g = np.linspace(expiries.min(), expiries.max(), 50)
    IV  = np.zeros((len(T_g), len(k_g)))
    for i, Tv in enumerate(T_g):
        F = spot * np.exp(rate * Tv)
        for j, kv in enumerate(k_g):
            try: IV[i,j] = surface.implied_vol(F * np.exp(kv), Tv) * 100
            except: IV[i,j] = np.nan
    fig3d = go.Figure(go.Surface(
        x=(np.exp(k_g)-1)*100, y=T_g, z=IV,
        colorscale=[[0,'#0f172a'],[0.2,'#1e3a5f'],[0.4,'#0f6e56'],
                    [0.6,'#00d4b8'],[0.8,'#7c6af7'],[1,'#f59e0b']],
        colorbar=dict(title=dict(text='IV %',font=dict(color='#e8eaf0',size=11)),
                      tickfont=dict(color='#e8eaf0',size=10), thickness=12, len=0.6),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=False)), opacity=0.92,
    ))
    fig3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(bgcolor='rgba(10,11,13,1)',
                   xaxis=dict(title='Moneyness (%)', gridcolor='#1e2530', color='#5a6278'),
                   yaxis=dict(title='Expiry (yr)',   gridcolor='#1e2530', color='#5a6278'),
                   zaxis=dict(title='IV (%)',         gridcolor='#1e2530', color='#5a6278'),
                   camera=dict(eye=dict(x=1.6,y=-1.6,z=0.9))),
        margin=dict(l=0,r=0,t=10,b=0), height=500,
        font=dict(family='DM Mono', color='#e8eaf0', size=11),
    )
    st.plotly_chart(fig3d, use_container_width=True)

# ── IV Smiles ─────────────────────────────────────────────────────────────────
if show_smile:
    st.markdown("### Implied Volatility Smile")
    k_f = np.linspace(-0.45, 0.45, 200)
    fig_sm = go.Figure()
    for idx, (date, svi) in enumerate(surface.svi_slices.items()):
        F     = surface.forwards.get(date, spot * np.exp(rate * svi.expiry))
        K_pct = (np.exp(k_f) - 1) * 100
        color = COLORS[idx % len(COLORS)]
        fig_sm.add_trace(go.Scatter(
            x=K_pct, y=svi.implied_vol(k_f)*100, name=f"{date} SVI",
            line=dict(color=color, width=2.5),
        ))
        # Market quotes overlay
        if chain:
            sq = chain.get_slice(date)
            if sq:
                fig_sm.add_trace(go.Scatter(
                    x=[(np.exp(np.log(q.strike/F))-1)*100 for q in sq],
                    y=[q.iv*100 for q in sq], mode='markers',
                    marker=dict(color=color, size=5), showlegend=False,
                ))
        # Heston overlay
        if show_heston_overlay and st.session_state['heston_params'] is not None:
            try:
                h = Heston(st.session_state['heston_params'])
                K_arr = F * np.exp(k_f)
                h_iv  = h.implied_vol_smile(K_arr, svi.expiry, MarketData(spot, rate)) * 100
                valid = ~np.isnan(h_iv)
                if valid.sum() > 5:
                    fig_sm.add_trace(go.Scatter(
                        x=K_pct[valid], y=h_iv[valid], name=f"{date} Heston",
                        line=dict(color=color, width=1.5, dash='dot'), showlegend=True,
                    ))
            except: pass
    fig_sm.add_vline(x=0, line_color='#5a6278', line_dash='dash', line_width=1,
                     annotation_text='ATM', annotation_font_color='#5a6278')
    fig_sm.update_layout(**PLOTLY_LAYOUT, height=380,
                          title="IV Smile — SVI (solid) · Heston (dotted) · market quotes (dots)",
                          xaxis_title="Moneyness (% from ATM)", yaxis_title="IV (%)",
                          legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_sm, use_container_width=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        atm_v = [float(s.implied_vol(np.array([0.0]))[0])*100 for s in surface.svi_slices.values()]
        h_atm = []
        if st.session_state['heston_params'] is not None:
            hm = Heston(st.session_state['heston_params'])
            for dt2, svi2 in surface.svi_slices.items():
                try:
                    iv2 = hm.implied_vol_smile(np.array([spot]), svi2.expiry, MarketData(spot, rate))
                    h_atm.append(float(iv2[0])*100 if not np.isnan(iv2[0]) else np.nan)
                except: h_atm.append(np.nan)
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=list(expiries), y=atm_v, name="SVI",
                                     line=dict(color='#00d4b8', width=2.5), mode='lines+markers',
                                     fill='tozeroy', fillcolor='rgba(0,212,184,0.08)'))
        if h_atm:
            fig_ts.add_trace(go.Scatter(x=list(expiries), y=h_atm, name="Heston",
                                         line=dict(color='#7c6af7', width=2, dash='dot'),
                                         mode='lines+markers'))
        fig_ts.update_layout(**PLOTLY_LAYOUT, title="ATM Vol Term Structure",
                              xaxis_title="Expiry", yaxis_title="ATM IV (%)", height=240,
                              legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_ts, use_container_width=True)
    with sc2:
        rhos = [s.rho for s in surface.svi_slices.values()]
        fig_r = go.Figure(go.Bar(x=dates, y=rhos,
                                  marker_color=['#ef4444' if r<0 else '#10b981' for r in rhos],
                                  text=[f"{r:.3f}" for r in rhos], textposition='outside',
                                  textfont=dict(size=10, color='#e8eaf0')))
        fig_r.update_layout(**PLOTLY_LAYOUT, title="Skew ρ by Expiry",
                             yaxis_title="ρ", height=240)
        st.plotly_chart(fig_r, use_container_width=True)

# ── RND ───────────────────────────────────────────────────────────────────────
if show_rnd:
    st.markdown("### Risk-Neutral Density")
    rnd_date = st.selectbox("Expiry", dates, index=min(2, len(dates)-1))
    K_rnd, dens = surface.risk_neutral_density(rnd_date)
    svi_T = surface.svi_slices[rnd_date]
    fwd   = surface.forwards.get(rnd_date, spot * np.exp(rate * svi_T.expiry))
    from scipy.stats import lognorm
    atm_iv2 = float(svi_T.implied_vol(np.array([0.0]))[0])
    s_ln = atm_iv2 * np.sqrt(svi_T.expiry)
    ln_d = lognorm.pdf(K_rnd, s=s_ln, scale=np.exp(np.log(fwd) - 0.5*s_ln**2))
    nf = np.trapezoid(ln_d, K_rnd)
    if nf > 0: ln_d /= nf
    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Scatter(x=K_rnd, y=dens, name="Market RND",
                                 line=dict(color='#00d4b8', width=2.5),
                                 fill='tozeroy', fillcolor='rgba(0,212,184,0.12)'))
    fig_r2.add_trace(go.Scatter(x=K_rnd, y=ln_d, name="Lognormal (BS)",
                                 line=dict(color='#5a6278', width=1.5, dash='dash')))
    fig_r2.add_vline(x=spot, line_color='#7c6af7', line_dash='dash', line_width=1.5,
                     annotation_text='Spot', annotation_font_color='#7c6af7')
    fig_r2.add_vline(x=fwd, line_color='#f59e0b', line_dash='dash', line_width=1.5,
                     annotation_text='Fwd', annotation_font_color='#f59e0b')
    fig_r2.update_layout(**PLOTLY_LAYOUT, title=f"RND — {rnd_date}",
                          xaxis_title="Strike", yaxis_title="Density", height=320,
                          legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_r2, use_container_width=True)

# ── Arb checks ────────────────────────────────────────────────────────────────
if show_arb:
    st.markdown("### Arbitrage Checks")
    c1, c2 = st.columns(2)
    for col, ok, label, detail in [
        (c1, cal_check['is_arbitrage_free'],
         f"{'✓' if cal_check['is_arbitrage_free'] else '✗'} Calendar spread arb-free",
         "Total variance non-decreasing in T"),
        (c2, but_check['is_arbitrage_free'],
         f"{'✓' if but_check['is_arbitrage_free'] else '✗'} Butterfly arb-free",
         "Risk-neutral density g(k) ≥ 0"),
    ]:
        clr = "#10b981" if ok else "#ef4444"
        col.markdown(f"""
        <div style="background:#111318;border:1px solid {clr};border-radius:8px;padding:14px;">
            <div style="font-family:'DM Mono',monospace;font-size:13px;color:{clr};
                        font-weight:500;">{label}</div>
            <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5a6278;
                        margin-top:4px;">{detail}</div>
        </div>""", unsafe_allow_html=True)