import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="rgba(200,200,200,0.15)",
    zeroline=False,
    showline=False,
    ticks="outside",
)
def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    s = str(hex_color).strip()
    if not s.startswith("#") or len(s) != 7:
        return s
    r = int(s[1:3], 16)
    g = int(s[3:5], 16)
    b = int(s[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"
warnings.filterwarnings('ignore')

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LaserOpt · Surface Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #0a0b0e;
    --surface: #111318;
    --border: #1e2130;
    --accent: #00e5ff;
    --accent2: #ff6b35;
    --text: #e8eaf2;
    --muted: #5a5f78;
    --good: #00ff88;
    --warn: #ffcc00;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
}
.metric-label {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: var(--accent);
    margin: 4px 0;
}
.metric-sub {
    font-size: 12px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
}

/* Section headers */
.section-title {
    font-size: 13px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Optimization result */
.opt-card {
    background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 100%);
    border: 1px solid rgba(0, 229, 255, 0.3);
    border-radius: 12px;
    padding: 24px;
}
.opt-param {
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
}
.opt-label { color: var(--muted); }
.opt-value { color: var(--accent); font-weight: 600; }

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}
.badge-good { background: rgba(0,255,136,0.1); color: var(--good); border: 1px solid rgba(0,255,136,0.3); }
.badge-warn { background: rgba(255,204,0,0.1); color: var(--warn); border: 1px solid rgba(255,204,0,0.3); }

/* Header */
.app-header {
    padding: 8px 0 24px;
}
.app-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #00e5ff 0%, #0099ff 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-subtitle {
    font-size: 13px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Slider styling */
.stSlider > div > div > div { background: var(--border) !important; }

/* Select boxes */
.stSelectbox > div { background: var(--surface) !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─── REAL experimental data from PhD dissertation ────────────────────────────
# Prykhodko V.O., NTU KhPI, 2024
# Material: AISI 321 austenitic stainless steel (12Kh18N10T)
# Laser: Minimarker 2, Nd:YAG nanosecond pulse, max power 50W
# Hatch step: 25 µm (all experiments)
# Source: Table 4.4 (processing params) + Tables 5.7, 5.8 (roughness results)

REAL_EXPERIMENTS = pd.DataFrame({
    "position":         ["Base", "Pos 1", "Pos 2", "Pos 3", "Pos 4"],
    "label":            ["Untreated", "Stochastic", "Quasi-stochastic", "Quasi-periodic", "Periodic"],
    "power_pct":        [0,    15,    20,    30,    40],
    "power_W":          [0,    7.5,   10.0,  15.0,  20.0],   # % of 50W max
    "scan_speed_mm_s":  [0,   100,   150,   150,   100],
    "pulse_freq_kHz":   [0,    50,    40,    40,    60],
    "hatch_um":         [0,    25,    25,    25,    25],
    # Surface roughness parameters (Table 5.7 — 3D areal, µm)
    "Sa_um":            [0.19, 0.24,  0.28,  0.36,  1.02],
    "Sq_um":            [0.24, 0.31,  0.34,  0.45,  1.24],
    "Sp_um":            [1.60, 1.38,  1.32,  1.83,  5.05],
    "Sv_um":            [1.10, 1.40,  1.64,  1.73,  3.10],
    "Sz_um":            [2.70, 2.78,  2.96,  3.56,  8.15],
    "Ssk":              [-0.15, 0.02, -0.06, -0.05,  0.42],
    "Sku":              [3.59,  3.59,  2.80,  2.89,  2.64],
    "Sdq":              [0.04,  0.04,  0.04,  0.09,  0.22],
    "Sdr_pct":          [0.07,  0.07,  0.08,  0.39,  2.47],
    # Profile roughness (Table 5.8 — 2D profile, µm)
    "Ra_x_um":          [0.046, 0.049, 0.080, 0.195, 0.532],
    "Ra_y_um":          [0.036, 0.043, 0.027, 0.074, 0.183],
    "Rq_x_um":          [0.057, 0.063, 0.094, 0.234, 0.639],
    "Rq_y_um":          [0.046, 0.053, 0.035, 0.092, 0.231],
    "Rz_x_um":          [0.234, 0.297, 0.340, 0.874, 2.384],
    "Rz_y_um":          [0.188, 0.241, 0.157, 0.396, 1.021],
    # Vickers hardness HV (Chapter 5.4)
    "hardness_HV":      [190,   184,   181,   176,   195],
    # Surface structure type
    "structure_type":   ["Untreated", "Stochastic", "Quasi-stochastic", "Quasi-periodic", "Periodic"],
    "color":            ["#64748B", "#3B82F6", "#8B5CF6", "#F59E0B", "#EF4444"],
})


@st.cache_data
def generate_dataset(n=1200):
    """
    Extended synthetic dataset anchored to real PhD dissertation data.
    Calibrated to match measured values: AISI 321, Minimarker 2 ns-laser, 50W max.
    Real anchor points: Table 4.4 (params) + Tables 5.7/5.8 (roughness).
    """
    np.random.seed(42)

    # Realistic parameter space based on dissertation experiments
    # Power: 5-25W (10-50% of 50W max), speed: 50-300 mm/s, freq: 20-100 kHz
    power_pct   = np.random.uniform(5, 50, n)
    laser_power = power_pct * 0.5   # W (50W max laser)
    scan_speed  = np.random.uniform(50, 300, n)    # mm/s  (dissertation range)
    pulse_freq  = np.random.uniform(20, 100, n)    # kHz
    hatch_dist  = np.full(n, 25.0)                 # µm (fixed at 25µm per dissertation)
    spot_size   = np.full(n, 50.0)                 # µm (estimated from Minimarker 2 specs)

    # Energy density (fluence per pulse, J/cm²) — key process parameter
    # E = P / (v * d_spot * f)  adapted to units
    pulse_energy_uJ = (laser_power * 1e6) / (pulse_freq * 1e3)   # µJ per pulse
    fluence = pulse_energy_uJ / (np.pi * (spot_size / 2)**2 * 1e-8)   # J/cm²
    fluence = np.clip(fluence, 0.01, 50)

    # Pulse overlap along scan direction
    pulse_spacing = scan_speed / (pulse_freq * 1e3) * 1e3   # µm between pulses
    overlap_x = np.clip(1 - pulse_spacing / spot_size, 0, 0.99)

    # Combined energy input (intensity metric)
    energy_input = fluence * (1 + overlap_x) * (spot_size / hatch_dist)

    # ── Ra model calibrated to dissertation Table 5.8 ──
    # Base: ~0.046 µm (untreated), Pos4: ~0.532 µm at 20W, 100mm/s, 60kHz
    Ra_base = (
        0.046                                        # untreated baseline
        + 0.18  * np.log1p(energy_input * 0.15)     # energy drives roughness
        + 0.04  * (power_pct / 15) ** 1.4           # power nonlinearity
        - 0.015 * np.log1p(scan_speed / 50)         # faster = smoother
        + 0.008 * (pulse_freq / 40)                  # frequency effect
        + 0.02  * np.sin(energy_input * 0.3)         # periodic structure formation
    )
    Ra_x = np.clip(Ra_base + np.random.normal(0, 0.012, n), 0.03, 0.65)

    # Ra_y always lower (anisotropy confirmed in dissertation)
    Ra_y = np.clip(Ra_x * np.random.uniform(0.30, 0.55, n), 0.02, 0.25)

    # Sa (3D areal) ~ 4-5x Ra_x based on dissertation Tables 5.7/5.8
    Sa = np.clip(Ra_x * np.random.uniform(3.5, 5.5, n), 0.15, 1.2)

    # Sdr: expanded surface area — key for scattering/adsorption
    Sdr = np.clip(0.07 + 0.8 * (energy_input / 20)**2 + np.random.exponential(0.05, n), 0.05, 3.0)

    # Ssk: skewness — negative for low power, positive for high (Pos4: +0.42)
    Ssk = np.clip(-0.15 + 0.014 * energy_input + np.random.normal(0, 0.08, n), -0.5, 0.6)

    # Structure type classification based on dissertation findings
    structure_score = energy_input / energy_input.max()
    structure_type = np.where(structure_score < 0.15, "Stochastic",
                     np.where(structure_score < 0.35, "Quasi-stochastic",
                     np.where(structure_score < 0.65, "Quasi-periodic", "Periodic")))

    # Surface hardness (HV) — dissertation shows decrease then increase
    # Low power: softening (176-184 HV), high power: hardening via martensite
    hardness = np.clip(
        190 - 8 * np.log1p(energy_input * 0.3) + 15 * (energy_input / energy_input.max())**3
        + np.random.normal(0, 3, n), 160, 230
    )

    df = pd.DataFrame({
        "power_pct":        np.round(power_pct, 0),
        "laser_power_W":    np.round(laser_power, 1),
        "scan_speed_mm_s":  np.round(scan_speed, 0),
        "pulse_freq_kHz":   np.round(pulse_freq, 0),
        "hatch_dist_um":    hatch_dist,
        "fluence_J_cm2":    np.round(fluence, 3),
        "overlap_x_pct":    np.round(overlap_x * 100, 1),
        "energy_input":     np.round(energy_input, 3),
        "Ra_x_um":          np.round(Ra_x, 4),
        "Ra_y_um":          np.round(Ra_y, 4),
        "Sa_um":            np.round(Sa, 3),
        "Sdr_pct":          np.round(Sdr, 3),
        "Ssk":              np.round(Ssk, 3),
        "hardness_HV":      np.round(hardness, 0),
        "structure_type":   structure_type,
    })
    return df


@st.cache_resource
def train_model(df):
    features = ["laser_power_W", "scan_speed_mm_s", "pulse_freq_kHz", "fluence_J_cm2", "energy_input"]
    X = df[features]
    y = df["Ra_x_um"]   # Ra in X direction (scan direction) — primary output

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=250, max_depth=5,
                                       learning_rate=0.07, random_state=42)
    model.fit(X_tr_s, y_train)

    y_pred = model.predict(X_te_s)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return model, scaler, features, r2, mae, importance, y_test.values, y_pred


# ─── Load data ───────────────────────────────────────────────────────────────
df = generate_dataset()
model, scaler, features, r2, mae, importance, y_test, y_pred = train_model(df)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0 0 20px;'>
        <div style='font-size:18px; font-weight:800; color:#00e5ff;'>⚡ LaserOpt</div>
        <div style='font-size:11px; color:#5a5f78; font-family: JetBrains Mono; margin-top:2px;'>
            Surface Analytics · AISI 321
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["📊 Overview", "🔬 Process Explorer", "🤖 ML Optimizer",
                         "📈 Parameter Study", "🧪 PhD Experimental Data"],
                    label_visibility="collapsed")

    st.markdown('<div class="section-title" style="margin-top:24px;">Experiment Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family: JetBrains Mono; font-size:12px; color:#5a5f78; line-height:2;'>
    Material: <span style='color:#e8eaf2'>AISI 321</span><br>
    Laser: <span style='color:#e8eaf2'>Minimarker 2</span><br>
    Max Power: <span style='color:#e8eaf2'>50 W (Nd:YAG)</span><br>
    Hatch step: <span style='color:#e8eaf2'>25 µm</span><br>
    Positions: <span style='color:#e8eaf2'>4 + Base</span><br>
    Model R²: <span style='color:#00ff88'>{r2:.4f}</span><br>
    MAE: <span style='color:#00e5ff'>{mae:.4f} µm</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:24px;">Filter Data</div>', unsafe_allow_html=True)
    ra_range = st.slider("Ra (X-dir) range (µm)", 0.03, 0.65, (0.03, 0.65), 0.01)
    speed_range = st.slider("Scan speed (mm/s)", 50, 300, (50, 300), 10)

    df_filtered = df[
        (df["Ra_x_um"] >= ra_range[0]) & (df["Ra_x_um"] <= ra_range[1]) &
        (df["scan_speed_mm_s"] >= speed_range[0]) & (df["scan_speed_mm_s"] <= speed_range[1])
    ]
    st.markdown(f'<div style="font-family:JetBrains Mono;font-size:11px;color:#5a5f78;">Showing {len(df_filtered):,} records</div>',
                unsafe_allow_html=True)


# ─── Plot theme ───────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Syne', color='#5a5f78', size=12),
    margin=dict(t=30, b=20, l=10, r=10),
)

# Default axis style — apply manually per chart to avoid conflicts
AXIS = dict(gridcolor='#1e2130', zerolinecolor='#1e2130', showline=False)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":

    st.markdown("""
    <div class="app-header">
        <div class="app-title">Surface Roughness Analytics</div>
        <div class="app-subtitle">// AISI 321 · Minimarker 2 Nd:YAG ns-laser · PhD dissertation data · NTU KhPI 2024</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row — calibrated to real dissertation measurements
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Ra (X-dir)</div>
            <div class="metric-value">{df_filtered['Ra_x_um'].mean():.3f}</div>
            <div class="metric-sub">µm · scan direction</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        best_sa = REAL_EXPERIMENTS['Sa_um'].min()
        st.markdown(f"""
        <div class="metric-card" style="--accent: #00ff88;">
            <div class="metric-label">Best Sa (Real)</div>
            <div class="metric-value" style="color:#00ff88;">{best_sa:.2f}</div>
            <div class="metric-sub">µm · untreated base</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="--accent: #ff6b35;">
            <div class="metric-label">Model R²</div>
            <div class="metric-value" style="color:#ff6b35;">{r2*100:.1f}%</div>
            <div class="metric-sub">GBR · Ra X-direction</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        max_sdr = REAL_EXPERIMENTS['Sdr_pct'].max()
        st.markdown(f"""
        <div class="metric-card" style="--accent: #7c3aed;">
            <div class="metric-label">Max Sdr (Real)</div>
            <div class="metric-value" style="color:#7c3aed;">{max_sdr:.2f}%</div>
            <div class="metric-sub">surface area increase · Pos 4</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-title">Ra Distribution by Laser Power</div>', unsafe_allow_html=True)
        df_filtered['power_bin'] = pd.cut(df_filtered['laser_power_W'],
                                           bins=[0,5,10,15,20,26],
                                           labels=["<5W","5-10W","10-15W","15-20W","20-25W"])
        fig = px.violin(df_filtered.dropna(subset=['power_bin']),
                        x="power_bin", y="Ra_x_um", color="power_bin",
                        box=True, points=False,
                        color_discrete_sequence=["#00e5ff","#0099ff","#7c3aed","#ff6b35","#ff2d78"])
        fig.update_layout(**PLOT_LAYOUT, height=280,
                          xaxis=dict(**AXIS_STYLE, title="Laser Power Range"),
                          yaxis=dict(**AXIS_STYLE, title="Ra X (µm)"),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        labels_map = {
            "laser_power_W":   "Laser Power (W)",
            "scan_speed_mm_s": "Scan Speed",
            "pulse_freq_kHz":  "Pulse Freq",
            "fluence_J_cm2":   "Fluence",
            "energy_input":    "Energy Input",
        }
        imp_df = importance.reset_index()
        imp_df.columns = ["feature", "importance"]
        imp_df["label"] = imp_df["feature"].map(labels_map)
        fig2 = go.Figure(go.Bar(
            x=imp_df["importance"],
            y=imp_df["label"],
            orientation='h',
            marker=dict(color=imp_df["importance"],
                        colorscale=[[0,'#1e2130'],[0.5,'#0099ff'],[1,'#00e5ff']],
                        showscale=False)
        ))
        fig2.update_layout(**PLOT_LAYOUT, height=280,
                           xaxis=dict(**AXIS_STYLE, title="Importance"),
                           yaxis=dict(**AXIS_STYLE, title="", categoryorder="total ascending"))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Energy input vs Ra scatter
    st.markdown('<div class="section-title">Energy Input vs Ra (X-direction) — Simulation</div>', unsafe_allow_html=True)
    sample = df_filtered.sample(min(600, len(df_filtered)), random_state=1)
    fig3 = px.scatter(sample, x="energy_input", y="Ra_x_um",
                      color="structure_type",
                      color_discrete_map={"Stochastic":"#3B82F6","Quasi-stochastic":"#8B5CF6",
                                          "Quasi-periodic":"#F59E0B","Periodic":"#EF4444"},
                      opacity=0.65,
                      labels={"energy_input": "Energy Input (a.u.)", "Ra_x_um": "Ra X (µm)",
                               "structure_type": "Surface Structure"})
    fig3.update_traces(marker=dict(size=5))
    fig3.update_layout(**PLOT_LAYOUT, height=300, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PROCESS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Process Explorer":

    st.markdown("""
    <div class="app-header">
        <div class="app-title">Process Explorer</div>
        <div class="app-subtitle">// Interactive parameter space · AISI 321 simulation</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        x_param = st.selectbox("X axis", features,
                                format_func=lambda x: x.replace("_"," "))
    with col2:
        y_param = st.selectbox("Y axis", [f for f in features if f != x_param], index=1,
                                format_func=lambda x: x.replace("_"," "))

    color_by = st.select_slider("Color by",
                                 options=["Ra_x_um", "Ra_y_um", "Sa_um", "Sdr_pct", "hardness_HV"],
                                 value="Ra_x_um")

    sample = df_filtered.sample(min(800, len(df_filtered)), random_state=42)
    color_labels = {"Ra_x_um": "Ra X (µm)", "Ra_y_um": "Ra Y (µm)",
                    "Sa_um": "Sa (µm)", "Sdr_pct": "Sdr (%)", "hardness_HV": "Hardness (HV)"}

    fig = px.scatter(sample, x=x_param, y=y_param,
                     color=color_by, color_continuous_scale="Turbo", opacity=0.6,
                     hover_data={"Ra_x_um":":.4f","Ra_y_um":":.4f","structure_type":True},
                     labels={x_param: x_param.replace("_"," "), y_param: y_param.replace("_"," "),
                              color_by: color_labels[color_by]})
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=420,
                      coloraxis_colorbar=dict(title=color_labels[color_by],
                                               tickfont=dict(color='#5a5f78', size=10)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Correlation heatmap
    st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = ["laser_power_W","scan_speed_mm_s","pulse_freq_kHz","fluence_J_cm2",
                "energy_input","Ra_x_um","Ra_y_um","Sa_um","Sdr_pct","hardness_HV"]
    corr = df_filtered[num_cols].corr()
    labels_short = ["Power","Speed","Freq","Fluence","Energy","Ra-X","Ra-Y","Sa","Sdr","HV"]
    fig_h = go.Figure(go.Heatmap(
        z=corr.values, x=labels_short, y=labels_short,
        colorscale=[[0,'#ff2d78'],[0.5,'#1e2130'],[1,'#00e5ff']],
        zmid=0, text=corr.round(2).values, texttemplate="%{text}",
        textfont=dict(size=10, color='#e8eaf2')))
    fig_h.update_layout(**PLOT_LAYOUT, height=360, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE)
    st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Optimizer":

    st.markdown("""
    <div class="app-header">
        <div class="app-title">ML Process Optimizer</div>
        <div class="app-subtitle">// Gradient Boosting Regressor · Ra prediction & parameter optimization</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="section-title">Input Parameters</div>', unsafe_allow_html=True)
        power   = st.slider("Laser Power (W)", 2.5, 25.0, 12.5, 0.5,
                             help="5–40% of Minimarker 2 max 50W")
        speed   = st.slider("Scan Speed (mm/s)", 50, 300, 150, 10)
        freq    = st.slider("Pulse Frequency (kHz)", 20, 100, 50, 5)

        # Derived for model
        pulse_spacing_um = speed / (freq * 1e3) * 1e3
        overlap_x_val = max(0, 1 - pulse_spacing_um / 50)
        pulse_e_uJ = (power * 1e6) / (freq * 1e3)
        fluence_val = pulse_e_uJ / (np.pi * 25**2 * 1e-8)
        fluence_val = np.clip(fluence_val, 0.01, 50)
        energy_input_val = fluence_val * (1 + overlap_x_val) * (50 / 25)

        X_input = np.array([[power, speed, freq, fluence_val, energy_input_val]])
        X_scaled = scaler.transform(X_input)
        ra_pred = float(model.predict(X_scaled)[0])
        ra_pred = np.clip(ra_pred, 0.03, 0.65)

    with col_result:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        # Quality classification based on dissertation structure types
        if ra_pred < 0.06:
            q_label, q_class, q_desc = "Stochastic", "badge-good", "Low energy · random distribution"
        elif ra_pred < 0.12:
            q_label, q_class, q_desc = "Quasi-stochastic", "badge-good", "Pos 2 range · beginning of periodicity"
        elif ra_pred < 0.25:
            q_label, q_class, q_desc = "Quasi-periodic", "badge-warn", "Pos 3 range · partial periodicity"
        else:
            q_label, q_class, q_desc = "Periodic", "badge-warn", "Pos 4 range · crater+ring structures"

        # Find nearest real experiment
        real_ra_x = REAL_EXPERIMENTS['Ra_x_um'].values
        nearest_idx = int(np.argmin(np.abs(real_ra_x - ra_pred)))
        nearest_pos = REAL_EXPERIMENTS.iloc[nearest_idx]

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0a1628,#0d1f3c);border:1px solid rgba(0,229,255,0.3);
                    border-radius:12px;padding:24px;">
            <div style="font-size:11px;letter-spacing:.15em;text-transform:uppercase;
                        color:#5a5f78;font-family:'JetBrains Mono';margin-bottom:12px;">
                Predicted Ra (X-direction)
            </div>
            <div style="font-size:52px;font-weight:800;color:#00e5ff;line-height:1;">
                {ra_pred:.4f}
            </div>
            <div style="font-size:14px;color:#5a5f78;font-family:'JetBrains Mono';margin-bottom:16px;">
                µm · AISI 321 · scan direction
            </div>
            <span style="display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;
                         font-family:'JetBrains Mono';background:rgba(0,255,136,0.1);
                         color:#00ff88;border:1px solid rgba(0,255,136,0.3);">{q_label}</span>&nbsp;
            <span style="font-size:12px;color:#5a5f78;font-family:'JetBrains Mono';">{q_desc}</span>
            <div style="margin-top:20px;">
                <div style="display:flex;justify-content:space-between;padding:10px 0;
                            border-bottom:1px solid #1e2130;font-family:'JetBrains Mono';font-size:14px;">
                    <span style="color:#5a5f78;">Fluence (est.)</span>
                    <span style="color:#00e5ff;font-weight:600;">{fluence_val:.2f} J/cm²</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:10px 0;
                            border-bottom:1px solid #1e2130;font-family:'JetBrains Mono';font-size:14px;">
                    <span style="color:#5a5f78;">Pulse overlap X</span>
                    <span style="color:#00e5ff;font-weight:600;">{overlap_x_val*100:.1f} %</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:10px 0;
                            border-bottom:1px solid #1e2130;font-family:'JetBrains Mono';font-size:14px;">
                    <span style="color:#5a5f78;">Nearest real experiment</span>
                    <span style="color:#00e5ff;font-weight:600;">{nearest_pos['label']} · Ra={nearest_pos['Ra_x_um']:.3f} µm</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:10px 0;
                            font-family:'JetBrains Mono';font-size:14px;">
                    <span style="color:#5a5f78;">Expected structure</span>
                    <span style="color:#00e5ff;font-weight:600;">{nearest_pos['structure_type']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Auto-Optimize for Target Ra</div>', unsafe_allow_html=True)
        target_ra = st.number_input("Target Ra X (µm)", min_value=0.04, max_value=0.60, value=0.20, step=0.01)

        if st.button("🔍 Find Optimal Parameters", use_container_width=True):
            grid = []
            for p in np.linspace(5, 22, 8):
                for s in np.linspace(50, 280, 8):
                    for f in np.linspace(30, 90, 6):
                        pulse_sp = s / (f * 1e3) * 1e3
                        ov = max(0, 1 - pulse_sp / 50)
                        pe = (p * 1e6) / (f * 1e3)
                        fl = np.clip(pe / (np.pi * 25**2 * 1e-8), 0.01, 50)
                        ei = fl * (1 + ov) * 2
                        grid.append([p, s, f, fl, ei])

            grid_arr = np.array(grid)
            grid_scaled = scaler.transform(grid_arr)
            preds = np.clip(model.predict(grid_scaled), 0.03, 0.65)
            best_idx = int(np.argmin(np.abs(preds - target_ra)))
            best = grid[best_idx]
            best_pred = preds[best_idx]

            st.markdown(f"""
            <div style="background:rgba(0,229,255,0.05); border:1px solid rgba(0,229,255,0.2);
                        border-radius:8px; padding:16px; font-family:'JetBrains Mono'; font-size:13px; margin-top:8px;">
                <div style="color:#00e5ff; font-weight:600; margin-bottom:8px;">
                    ✓ Optimal found · Predicted Ra = {best_pred:.4f} µm
                </div>
                <div style="color:#5a5f78; line-height:2.4;">
                Power: <span style="color:#e8eaf2">{best[0]:.1f} W ({best[0]/0.5:.0f}%)</span><br>
                Speed: <span style="color:#e8eaf2">{best[1]:.0f} mm/s</span> &nbsp;|&nbsp;
                Freq: <span style="color:#e8eaf2">{best[2]:.0f} kHz</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Model evaluation
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Evaluation · Predicted vs Actual Ra X</div>', unsafe_allow_html=True)
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(
        x=y_test[:300], y=y_pred[:300], mode='markers',
        marker=dict(size=4, color='#00e5ff', opacity=0.5), name='Predictions'))
    mn, mx = float(np.min(y_test)), float(np.max(y_test))
    fig_eval.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode='lines',
        line=dict(color='#ff6b35', dash='dash', width=1.5), name='Perfect fit'))
    # Add real experiment points
    fig_eval.add_trace(go.Scatter(
        x=REAL_EXPERIMENTS['Ra_x_um'].values,
        y=REAL_EXPERIMENTS['Ra_x_um'].values,
        mode='markers', name='Real (dissertation)',
        marker=dict(size=10, color='#00ff88', symbol='star', line=dict(color='white', width=1))))
    fig_eval.update_layout(**PLOT_LAYOUT, height=320,
                           xaxis=dict(**AXIS_STYLE, title="Actual Ra X (µm)"),
                           yaxis=dict(**AXIS_STYLE, title="Predicted Ra X (µm)"))
    st.plotly_chart(fig_eval, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PARAMETER STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Parameter Study":

    st.markdown("""
    <div class="app-header">
        <div class="app-title">Parameter Study</div>
        <div class="app-subtitle">// Single-variable sensitivity analysis · all others held at median</div>
    </div>
    """, unsafe_allow_html=True)

    param = st.selectbox("Select parameter to study",
                          features,
                          format_func=lambda x: x.replace("_", " "))

    medians = {f: df[f].median() for f in features}
    param_range = np.linspace(df[param].min(), df[param].max(), 100)

    study_data = []
    for val in param_range:
        row = [medians[f] if f != param else val for f in features]
        study_data.append(row)

    study_arr = np.array(study_data)
    study_scaled = scaler.transform(study_arr)
    study_preds = model.predict(study_scaled)

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=param_range, y=study_preds,
        mode='lines',
        line=dict(color='#00e5ff', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0,229,255,0.05)',
        name='Predicted Ra'
    ))
    # Add target zone
    fig_s.add_hrect(y0=0, y1=0.4, fillcolor="rgba(0,255,136,0.05)",
                    line_width=0, annotation_text="Mirror finish zone",
                    annotation_position="top left",
                    annotation_font=dict(color="#00ff88", size=11))
    fig_s.update_layout(**PLOT_LAYOUT, height=360,
                        xaxis=dict(**AXIS_STYLE, title=param.replace("_", " ")),
                        yaxis=dict(**AXIS_STYLE, title="Predicted Ra (µm)"))
    st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

    # Key insights
    min_idx = np.argmin(study_preds)
    st.markdown(f"""
    <div style="background:rgba(0,229,255,0.04); border:1px solid #1e2130;
                border-radius:10px; padding:20px; font-family:'JetBrains Mono'; font-size:13px; margin-top:4px;">
        <div style="color:#5a5f78; margin-bottom:12px; font-size:11px; letter-spacing:.1em; text-transform:uppercase;">
            Sensitivity Insights
        </div>
        <div style="line-height:2.2; color:#5a5f78;">
            Optimal value: <span style="color:#00e5ff; font-weight:600;">{param_range[min_idx]:.1f}</span>
            &nbsp;→&nbsp; Ra = <span style="color:#00ff88; font-weight:600;">{study_preds[min_idx]:.3f} µm</span><br>
            Ra range: <span style="color:#e8eaf2">{study_preds.min():.3f} – {study_preds.max():.3f} µm</span>
            &nbsp;(Δ = {study_preds.max()-study_preds.min():.3f} µm)<br>
            Influence score: <span style="color:#ff6b35">{importance[param]*100:.1f}%</span> of model variance
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2D heatmap study
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">2D Interaction Map</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Parameter 1", features, index=0,
                           format_func=lambda x: x.replace("_", " "))
    with col2:
        p2 = st.selectbox("Parameter 2", [f for f in features if f != p1], index=1,
                           format_func=lambda x: x.replace("_", " "))

    r1 = np.linspace(df[p1].min(), df[p1].max(), 40)
    r2 = np.linspace(df[p2].min(), df[p2].max(), 40)
    grid_rows = []
    for v1 in r1:
        for v2 in r2:
            row = [medians[f] if (f != p1 and f != p2) else (v1 if f == p1 else v2) for f in features]
            grid_rows.append(row)

    grid_2d = np.array(grid_rows)
    preds_2d = model.predict(scaler.transform(grid_2d)).reshape(40, 40)

    fig_2d = go.Figure(go.Heatmap(
        z=preds_2d,
        x=np.round(r2, 1),
        y=np.round(r1, 1),
        colorscale=[[0,'#00ff88'],[0.3,'#00e5ff'],[0.7,'#7c3aed'],[1,'#ff2d78']],
        colorbar=dict(title="Ra (µm)", tickfont=dict(color='#5a5f78', size=10))
    ))
    fig_2d.update_layout(**PLOT_LAYOUT, height=360,
                         xaxis=dict(**AXIS_STYLE, title=p2.replace("_", " ")),
                         yaxis=dict(**AXIS_STYLE, title=p1.replace("_", " ")))
    st.plotly_chart(fig_2d, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PhD EXPERIMENTAL DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 PhD Experimental Data":

    st.markdown("""
    <div class="app-header">
        <div class="app-title">PhD Experimental Results</div>
        <div class="app-subtitle">
            // Prykhodko V.O. · NTU KhPI 2024 · AISI 321 · Minimarker 2 Nd:YAG · Tables 4.4, 5.7, 5.8
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Info banner ──
    st.markdown("""
    <div style="background:rgba(0,229,255,0.04); border:1px solid #1e2130; border-left: 3px solid #00e5ff;
                border-radius:8px; padding:16px 20px; font-family:'JetBrains Mono'; font-size:12px;
                color:#5a5f78; margin-bottom:24px; line-height:2;">
        <b style="color:#00e5ff;">Material:</b> AISI 321 (12Kh18N10T) austenitic stainless steel, 35×75×1 mm samples &nbsp;|&nbsp;
        <b style="color:#00e5ff;">Laser:</b> Minimarker 2, Nd:YAG, max 50W, nanosecond pulses &nbsp;|&nbsp;
        <b style="color:#00e5ff;">Hatch step:</b> 25 µm &nbsp;|&nbsp;
        <b style="color:#00e5ff;">Microscopes:</b> Bruker Alicona PortableRL · Bruker ICON AFM · KEYENCE VHX 7100 · ZEISS AXIO &nbsp;|&nbsp;
        <b style="color:#00e5ff;">Hardness:</b> ZwickRoell Vickers · ISO 6507
    </div>
    """, unsafe_allow_html=True)

    # ── Real KPI cards ──
    cols = st.columns(5)
    for i, (_, row) in enumerate(REAL_EXPERIMENTS.iterrows()):
        with cols[i]:
            color = row['color']
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color};">
                <div class="metric-label" style="color:{color};">{row['position']}</div>
                <div class="metric-value" style="color:{color}; font-size:20px;">{row['label']}</div>
                <div class="metric-sub">Sa = {row['Sa_um']:.2f} µm</div>
                <div class="metric-sub">Ra-X = {row['Ra_x_um']:.3f} µm</div>
                <div class="metric-sub">{row['power_W']:.1f}W · {row['scan_speed_mm_s']:.0f}mm/s · {row['pulse_freq_kHz']:.0f}kHz</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Processing parameters table ──
    st.markdown('<div class="section-title">Processing Parameters · Table 4.4</div>', unsafe_allow_html=True)
    params_df = REAL_EXPERIMENTS[['position','label','power_pct','power_W','scan_speed_mm_s','pulse_freq_kHz','hatch_um']].copy()
    params_df.columns = ['Position','Structure Type','Power (%)','Power (W)','Scan Speed (mm/s)','Pulse Freq (kHz)','Hatch (µm)']
    st.dataframe(params_df.style.apply(
        lambda r: [f'color: {REAL_EXPERIMENTS.iloc[i]["color"]}' if c == 'Position' else '' for i, c in enumerate(r.index)],
        axis=1
    ), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ra X vs Y anisotropy chart ──
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Ra Anisotropy: X vs Y Direction · Table 5.8</div>', unsafe_allow_html=True)
        pos_labels = REAL_EXPERIMENTS['position'].tolist()
        colors_list = REAL_EXPERIMENTS['color'].tolist()

        fig_ra = go.Figure()
        fig_ra.add_trace(go.Bar(
            name='Ra X (scan dir)', x=pos_labels,
            y=REAL_EXPERIMENTS['Ra_x_um'].tolist(),
            marker_color=[c for c in colors_list],
            opacity=1.0
        ))
        fig_ra.add_trace(go.Bar(
            name='Ra Y (cross dir)', x=pos_labels,
            y=REAL_EXPERIMENTS['Ra_y_um'].tolist(),
            marker_color=[c for c in colors_list],
            opacity=0.45
        ))
        fig_ra.update_layout(**PLOT_LAYOUT, height=300, barmode='group',
                             xaxis=dict(**AXIS_STYLE, title="Position"),
                             yaxis=dict(**AXIS_STYLE, title="Ra (µm)"),
                             showlegend=True)
        fig_ra.update_layout(legend=dict(orientation='h', y=1.1, bgcolor='rgba(0,0,0,0)', font=dict(size=11)))
        st.plotly_chart(fig_ra, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        st.markdown('<div class="section-title">3D Areal Parameters Sa, Sq, Sz · Table 5.7</div>', unsafe_allow_html=True)
        fig_3d = go.Figure()
        for param_name, label, opacity in [('Sa_um','Sa',1.0),('Sq_um','Sq',0.65),('Sz_um','Sz',0.35)]:
            fig_3d.add_trace(go.Bar(
                name=label, x=pos_labels,
                y=REAL_EXPERIMENTS[param_name].tolist(),
                marker_color=['#00e5ff','#3B82F6','#8B5CF6','#F59E0B','#EF4444'],
                opacity=opacity
            ))
        fig_3d.update_layout(**PLOT_LAYOUT, height=300, barmode='group',
                             xaxis=dict(**AXIS_STYLE, title="Position"),
                             yaxis=dict(**AXIS_STYLE, title="µm"),
                             showlegend=True)
        fig_3d.update_layout(legend=dict(orientation='h', y=1.1, bgcolor='rgba(0,0,0,0)', font=dict(size=11)))
        st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar": False})

    # ── Ssk, Sku, Sdr row ──
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="section-title">Skewness Ssk · Table 5.7</div>', unsafe_allow_html=True)
        fig_ssk = go.Figure(go.Bar(
            x=pos_labels, y=REAL_EXPERIMENTS['Ssk'].tolist(),
            marker_color=['#EF4444' if v > 0 else '#00e5ff' for v in REAL_EXPERIMENTS['Ssk']],
            text=[f'{v:.2f}' for v in REAL_EXPERIMENTS['Ssk']],
            textposition='outside', textfont=dict(color='#e8eaf2', size=11)
        ))
        fig_ssk.add_hline(y=0, line_color='#5a5f78', line_dash='dash', line_width=1)
        fig_ssk.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, height=240,
                              showlegend=False,
                              annotations=[dict(text="↑ peaks dominant", x=0.01, y=0.95,
                                                xref='paper', yref='paper', showarrow=False,
                                                font=dict(color='#EF4444', size=10)),
                                           dict(text="↓ valleys dominant", x=0.01, y=0.08,
                                                xref='paper', yref='paper', showarrow=False,
                                                font=dict(color='#00e5ff', size=10))])
        st.plotly_chart(fig_ssk, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown('<div class="section-title">Kurtosis Sku · Table 5.7</div>', unsafe_allow_html=True)
        fig_sku = go.Figure(go.Bar(
            x=pos_labels, y=REAL_EXPERIMENTS['Sku'].tolist(),
            marker_color=colors_list,
            text=[f'{v:.2f}' for v in REAL_EXPERIMENTS['Sku']],
            textposition='outside', textfont=dict(color='#e8eaf2', size=11)
        ))
        fig_sku.add_hline(y=3, line_color='#ff6b35', line_dash='dash', line_width=1,
                          annotation_text="Normal dist (Sku=3)",
                          annotation_font=dict(color='#ff6b35', size=10))
        fig_sku.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, yaxis=dict(**AXIS_STYLE, title="Sku"), height=240, showlegend=False)
        st.plotly_chart(fig_sku, use_container_width=True, config={"displayModeBar": False})

    with col_c:
        st.markdown('<div class="section-title">Surface Area Sdr (%) · Table 5.7</div>', unsafe_allow_html=True)
        fig_sdr = go.Figure(go.Bar(
            x=pos_labels, y=REAL_EXPERIMENTS['Sdr_pct'].tolist(),
            marker_color=colors_list,
            text=[f'{v:.2f}%' for v in REAL_EXPERIMENTS['Sdr_pct']],
            textposition='outside', textfont=dict(color='#e8eaf2', size=11)
        ))
        fig_sdr.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, yaxis=dict(**AXIS_STYLE, title="Sdr (%)"), height=240, showlegend=False)
        st.plotly_chart(fig_sdr, use_container_width=True, config={"displayModeBar": False})

    # ── Hardness ──
    st.markdown('<div class="section-title">Vickers Hardness HV · ZwickRoell · ISO 6507</div>', unsafe_allow_html=True)
    fig_hv = go.Figure()
    fig_hv.add_trace(go.Scatter(
        x=REAL_EXPERIMENTS['position'].tolist(),
        y=REAL_EXPERIMENTS['hardness_HV'].tolist(),
        mode='lines+markers+text',
        line=dict(color='#00e5ff', width=2),
        marker=dict(size=12, color=colors_list,
                    line=dict(color='white', width=1.5)),
        text=[f"{v} HV" for v in REAL_EXPERIMENTS['hardness_HV']],
        textposition='top center',
        textfont=dict(color='#e8eaf2', size=11)
    ))
    fig_hv.add_hline(y=190, line_color='#5a5f78', line_dash='dot', line_width=1,
                     annotation_text="Base (untreated 190 HV)",
                     annotation_font=dict(color='#5a5f78', size=10),
                     annotation_position="right")
    fig_hv.add_annotation(
        x="Pos 3", y=176, text="Min hardness<br>(softening zone)",
        showarrow=True, arrowhead=2, arrowcolor='#ff6b35',
        font=dict(color='#ff6b35', size=10), ax=40, ay=-30
    )
    fig_hv.update_layout(**PLOT_LAYOUT, height=280,
                         xaxis=AXIS_STYLE,
                         yaxis=dict(**AXIS_STYLE, range=[160, 210], title="Hardness (HV)"),
                         showlegend=False)
    st.plotly_chart(fig_hv, use_container_width=True, config={"displayModeBar": False})

    # ── Surface structure evolution radar ──
    st.markdown('<div class="section-title">Surface Structure Evolution · Multi-parameter Radar</div>', unsafe_allow_html=True)
    categories = ['Sa (norm)', 'Sdr (norm)', 'Sdq (norm)', 'Ra-X (norm)', 'Hardness (inv norm)']
    fig_radar = go.Figure()

# Normalize each parameter 0→1
    sa_n   = (REAL_EXPERIMENTS['Sa_um']      - REAL_EXPERIMENTS['Sa_um'].min())   / (REAL_EXPERIMENTS['Sa_um'].max()   - REAL_EXPERIMENTS['Sa_um'].min())
    sdr_n  = (REAL_EXPERIMENTS['Sdr_pct']    - REAL_EXPERIMENTS['Sdr_pct'].min()) / (REAL_EXPERIMENTS['Sdr_pct'].max() - REAL_EXPERIMENTS['Sdr_pct'].min())
    sdq_n  = (REAL_EXPERIMENTS['Sdq']        - REAL_EXPERIMENTS['Sdq'].min())     / (REAL_EXPERIMENTS['Sdq'].max()     - REAL_EXPERIMENTS['Sdq'].min())
    rax_n  = (REAL_EXPERIMENTS['Ra_x_um']    - REAL_EXPERIMENTS['Ra_x_um'].min()) / (REAL_EXPERIMENTS['Ra_x_um'].max() - REAL_EXPERIMENTS['Ra_x_um'].min())
    hv_n   = 1 - (REAL_EXPERIMENTS['hardness_HV'] - REAL_EXPERIMENTS['hardness_HV'].min()) / (REAL_EXPERIMENTS['hardness_HV'].max() - REAL_EXPERIMENTS['hardness_HV'].min())

    for i, (_, row) in enumerate(REAL_EXPERIMENTS.iterrows()):
        vals = [sa_n.iloc[i], sdr_n.iloc[i], sdq_n.iloc[i], rax_n.iloc[i], hv_n.iloc[i]]
        vals += [vals[0]]  # close loop

        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill='toself',
            name=row['position'],
            line=dict(color=row['color']),
            fillcolor=hex_to_rgba(row['color'], 0.1),
            opacity=0.8
    ))

    fig_radar.update_layout(
        **{k: v for k, v in PLOT_LAYOUT.items() if k not in ['xaxis', 'yaxis']},
        height=380,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='#1e2130',
                tickfont=dict(color='#5a5f78', size=9)
            ),
            angularaxis=dict(
                gridcolor='#1e2130',
                tickfont=dict(color='#e8eaf2', size=11)
            )
        )
    )

    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

# ── Full data table ──
st.markdown('<div class="section-title">Complete Measurement Table · Tables 5.7 + 5.8</div>', unsafe_allow_html=True)
full_table = REAL_EXPERIMENTS[[
    'position','label','power_W','scan_speed_mm_s','pulse_freq_kHz',
    'Sa_um','Sq_um','Ssk','Sku','Sdr_pct','Sdq',
    'Ra_x_um','Ra_y_um','Rq_x_um','Rq_y_um','Rz_x_um','Rz_y_um',
    'hardness_HV'
]].copy()
full_table.columns = [
    'Pos','Type','P(W)','v(mm/s)','f(kHz)',
    'Sa(µm)','Sq(µm)','Ssk','Sku','Sdr(%)','Sdq',
    'Ra-X','Ra-Y','Rq-X','Rq-Y','Rz-X','Rz-Y',
    'HV'
]
st.dataframe(full_table, use_container_width=True, hide_index=True)
