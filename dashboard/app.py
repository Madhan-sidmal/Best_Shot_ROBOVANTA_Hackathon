"""
KrishiDrishti Dashboard — Streamlit Application
==================================================
Interactive dashboard for crop classification, moisture stress
visualization, and irrigation advisory display.

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DASHBOARD, CROP_CLASSES, STRESS_THRESHOLDS, IRRIGATION_ADVISORY

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="KrishiDrishti — AI Crop Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Dark theme enhancement */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(120deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(64, 145, 108, 0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #b7e4c7;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #52b788;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #adb5bd;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status badges */
    .status-adequate { background: #00AA00; color: white; padding: 4px 12px; border-radius: 20px; }
    .status-watch { background: #FFDD00; color: black; padding: 4px 12px; border-radius: 20px; }
    .status-urgent { background: #FF8800; color: white; padding: 4px 12px; border-radius: 20px; }
    .status-critical { background: #FF0000; color: white; padding: 4px 12px; border-radius: 20px; }
    
    /* Sidebar */
    .css-1d391kg { background: #1a1a2e; }
    
    /* Remove default padding */
    .block-container { padding-top: 1rem; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# GENERATE DEMO DATA
# ============================================================
@st.cache_data
def generate_demo_data():
    """Generate realistic demo data for dashboard display."""
    np.random.seed(42)
    
    # Crop distribution
    crops = {
        'Rice (Paddy)': {'area_ha': 12500, 'color': '#228B22', 'pct': 35},
        'Cotton': {'area_ha': 7800, 'color': '#FFD700', 'pct': 22},
        'Maize': {'area_ha': 5200, 'color': '#FF8C00', 'pct': 15},
        'Soybean': {'area_ha': 4100, 'color': '#32CD32', 'pct': 11},
        'Sugarcane': {'area_ha': 3200, 'color': '#9370DB', 'pct': 9},
        'Groundnut': {'area_ha': 1800, 'color': '#CD853F', 'pct': 5},
        'Pulses': {'area_ha': 1100, 'color': '#FF6347', 'pct': 3},
    }
    
    # Time series (24 8-day composites ≈ 6 months)
    n_steps = 24
    dates = pd.date_range('2025-06-01', periods=n_steps, freq='8D')
    t = np.linspace(0, 2 * np.pi, n_steps)
    
    ndvi_rice = 0.2 + 0.5 * np.sin(t - np.pi/6) + np.random.normal(0, 0.02, n_steps)
    ndvi_cotton = 0.15 + 0.45 * np.sin(t - np.pi/4) + np.random.normal(0, 0.02, n_steps)
    evi = 0.15 + 0.4 * np.sin(t - np.pi/6) + np.random.normal(0, 0.015, n_steps)
    ndwi = 0.1 + 0.35 * np.sin(t - np.pi/6) + np.random.normal(0, 0.02, n_steps)
    vh = -18 + 6 * np.sin(t - np.pi/6) + np.random.normal(0, 0.3, n_steps)
    vv = -12 + 4 * np.sin(t - np.pi/6) + np.random.normal(0, 0.3, n_steps)
    
    ts_df = pd.DataFrame({
        'Date': dates,
        'NDVI_Rice': np.clip(ndvi_rice, 0, 1),
        'NDVI_Cotton': np.clip(ndvi_cotton, 0, 1),
        'EVI': np.clip(evi, 0, 1),
        'NDWI': np.clip(ndwi, -0.5, 0.5),
        'VH': vh,
        'VV': vv
    })
    
    # Stress distribution
    stress_data = {
        'No Stress': 45,
        'Mild Stress': 25,
        'Moderate Stress': 20,
        'Severe Stress': 10
    }
    
    # Advisory distribution
    advisory_data = {
        'Adequate': 42,
        'Watch': 28,
        'Urgent': 20,
        'Critical': 10
    }
    
    # Pixel-level advisory table
    n_fields = 50
    advisory_table = pd.DataFrame({
        'Field_ID': [f"F-{i:03d}" for i in range(n_fields)],
        'Crop': np.random.choice(list(crops.keys()), n_fields),
        'Stage': np.random.choice(['Germination', 'Vegetative', 'Reproductive', 'Maturity'], n_fields),
        'Kc': np.random.uniform(0.3, 1.25, n_fields).round(2),
        'ETc (mm/8d)': np.random.uniform(15, 55, n_fields).round(1),
        'Rainfall (mm)': np.random.exponential(12, n_fields).round(1),
        'Deficit (mm)': np.random.uniform(0, 40, n_fields).round(1),
        'VCI': np.random.uniform(0.1, 0.9, n_fields).round(2),
    })
    advisory_table['Status'] = advisory_table['Deficit (mm)'].apply(
        lambda d: '🟢 Adequate' if d < 5 else ('🟡 Watch' if d < 15 else ('🟠 Urgent' if d < 30 else '🔴 Critical'))
    )
    
    return crops, ts_df, stress_data, advisory_data, advisory_table


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown("### 🛰️ KrishiDrishti")
        st.markdown("**AI Crop Intelligence Platform**")
        st.markdown("---")
        
        # Study area selector
        st.markdown("#### 📍 Study Area")
        study_area = st.selectbox(
            "Canal Command Area",
            ["Pilot Area — Canal Command", "Extended Region", "Full District"],
            index=0
        )
        
        # Season selector
        st.markdown("#### 📅 Season")
        season = st.selectbox(
            "Cropping Season",
            ["Kharif 2025", "Rabi 2024-25", "Kharif 2024"],
            index=0
        )
        
        # Layer controls
        st.markdown("#### 🗺️ Map Layers")
        show_crop_map = st.checkbox("Crop Classification", value=True)
        show_stress_map = st.checkbox("Moisture Stress", value=True)
        show_advisory_map = st.checkbox("Irrigation Advisory", value=True)
        show_canal = st.checkbox("Canal Boundary", value=True)
        show_sar = st.checkbox("SAR Backscatter", value=False)
        
        # Date slider
        st.markdown("#### ⏱️ Time Step")
        date_idx = st.slider("Composite Date", 0, 23, 12, 
                              help="Select an 8-day composite (June → November)")
        
        st.markdown("---")
        
        # Model info
        st.markdown("#### 🤖 Model Performance")
        st.metric("Classification Accuracy", "87.3%", "+2.3%")
        st.metric("Cohen's Kappa", "0.84", "+0.05")
        st.metric("F1 Score (Macro)", "0.85", "+0.03")
        
        st.markdown("---")
        st.markdown("#### 📥 Export")
        col1, col2 = st.columns(2)
        with col1:
            st.button("📄 PDF Report", use_container_width=True)
        with col2:
            st.button("🗺️ GeoTIFF", use_container_width=True)
        
        st.markdown("---")
        st.caption("Team BEST SHOT")
    
    return {
        'study_area': study_area,
        'season': season,
        'show_crop_map': show_crop_map,
        'show_stress_map': show_stress_map,
        'show_advisory_map': show_advisory_map,
        'show_canal': show_canal,
        'date_idx': date_idx
    }


# ============================================================
# HEADER
# ============================================================
def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ KrishiDrishti — AI Crop Intelligence Dashboard</h1>
        <p>PS 06 — AI-Driven Crop Type, Moisture Stress & Irrigation Advisory</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# METRIC CARDS
# ============================================================
def render_metrics(crops, stress_data, advisory_data):
    """Render top-level KPI metric cards."""
    total_area = sum(c['area_ha'] for c in crops.values())
    stressed_pct = stress_data['Moderate Stress'] + stress_data['Severe Stress']
    irrigation_needed = advisory_data['Urgent'] + advisory_data['Critical']
    
    cols = st.columns(6)
    
    metrics = [
        ("🌾", f"{total_area:,} ha", "Total Crop Area"),
        ("🗂️", f"{len(crops)}", "Crop Types"),
        ("📊", "87.3%", "Classification OA"),
        ("💧", f"{stressed_pct}%", "Under Stress"),
        ("🚿", f"{irrigation_needed}%", "Irrigation Needed"),
        ("🛰️", "NISAR Ready", "Architecture"),
    ]
    
    for col, (emoji, value, label) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.5rem;">{emoji}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# MAP VISUALIZATION
# ============================================================
def render_map(controls):
    """Render the interactive Folium map."""
    # Create base map
    m = folium.Map(
        location=DASHBOARD['map_center'],
        zoom_start=DASHBOARD['map_zoom'],
        tiles='CartoDB dark_matter'
    )
    
    # Add layer control
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('Stamen Terrain', name='Terrain').add_to(m)
    
    # Canal command area boundary (demo polygon)
    canal_boundary = [
        [17.1, 78.1], [17.1, 78.4], [17.4, 78.4], [17.4, 78.1], [17.1, 78.1]
    ]
    
    if controls.get('show_canal', True):
        folium.Polygon(
            locations=canal_boundary,
            color='#00CED1',
            weight=3,
            fill=True,
            fill_opacity=0.05,
            popup='Canal Command Area Boundary',
            tooltip='Pilot Canal Command Area'
        ).add_to(m)
    
    # Demo markers for fields
    np.random.seed(42)
    n_markers = 25
    lats = np.random.uniform(17.15, 17.35, n_markers)
    lons = np.random.uniform(78.15, 78.35, n_markers)
    crops_list = ['Rice', 'Cotton', 'Maize', 'Soybean', 'Sugarcane']
    statuses = ['🟢 Adequate', '🟡 Watch', '🟠 Urgent', '🔴 Critical']
    status_colors = ['green', 'orange', 'orange', 'red']
    
    for i in range(n_markers):
        crop = np.random.choice(crops_list)
        status_idx = np.random.choice([0, 0, 0, 1, 1, 2, 3])
        status = statuses[status_idx]
        color = status_colors[status_idx]
        
        popup_html = f"""
        <div style='font-family: Arial; width: 200px;'>
            <b>Field F-{i:03d}</b><br>
            <b>Crop:</b> {crop}<br>
            <b>Stage:</b> Reproductive<br>
            <b>VCI:</b> {np.random.uniform(0.2, 0.8):.2f}<br>
            <b>Deficit:</b> {np.random.uniform(0, 35):.1f} mm<br>
            <b>Status:</b> {status}<br>
        </div>
        """
        
        folium.CircleMarker(
            location=[lats[i], lons[i]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Field F-{i:03d}: {crop} — {status}"
        ).add_to(m)
    
    # Layer control
    folium.LayerControl().add_to(m)
    
    return m


# ============================================================
# CHARTS
# ============================================================
def render_time_series(ts_df, date_idx):
    """Render interactive NDVI/EVI/SAR time series charts."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Vegetation Indices (NDVI, EVI, NDWI)', 'SAR Backscatter (VH, VV)'),
        vertical_spacing=0.12
    )
    
    # NDVI
    fig.add_trace(
        go.Scatter(x=ts_df['Date'], y=ts_df['NDVI_Rice'], name='NDVI (Rice)',
                   line=dict(color='#52b788', width=2.5), fill='tozeroy',
                   fillcolor='rgba(82, 183, 136, 0.1)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ts_df['Date'], y=ts_df['EVI'], name='EVI',
                   line=dict(color='#74c69d', width=2, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ts_df['Date'], y=ts_df['NDWI'], name='NDWI',
                   line=dict(color='#48bfe3', width=2)),
        row=1, col=1
    )
    
    # Current date marker
    current_date = ts_df['Date'].iloc[date_idx]
    fig.add_vline(x=current_date, line_width=2, line_dash="dot", 
                  line_color="#ff6b6b", row=1, col=1)
    
    # SAR
    fig.add_trace(
        go.Scatter(x=ts_df['Date'], y=ts_df['VH'], name='VH',
                   line=dict(color='#7b2cbf', width=2.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=ts_df['Date'], y=ts_df['VV'], name='VV',
                   line=dict(color='#c77dff', width=2, dash='dash')),
        row=2, col=1
    )
    fig.add_vline(x=current_date, line_width=2, line_dash="dot", 
                  line_color="#ff6b6b", row=2, col=1)
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#adb5bd'),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    
    return fig


def render_crop_distribution(crops):
    """Render crop distribution charts."""
    df = pd.DataFrame([
        {'Crop': name, 'Area (ha)': info['area_ha'], 'Percentage': info['pct']}
        for name, info in crops.items()
    ])
    colors = [info['color'] for info in crops.values()]
    
    fig = px.pie(
        df, values='Area (ha)', names='Crop',
        color_discrete_sequence=colors,
        hole=0.45
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#adb5bd'),
        height=350,
        showlegend=True,
        legend=dict(font=dict(size=11))
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def render_stress_chart(stress_data):
    """Render stress distribution."""
    colors_map = {'No Stress': '#00AA00', 'Mild Stress': '#FFDD00', 
                  'Moderate Stress': '#FF8800', 'Severe Stress': '#FF0000'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(stress_data.keys()),
            y=list(stress_data.values()),
            marker_color=[colors_map[k] for k in stress_data.keys()],
            text=[f"{v}%" for v in stress_data.values()],
            textposition='auto',
            textfont=dict(size=14, color='white')
        )
    ])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        yaxis_title='Area (%)',
        font=dict(color='#adb5bd')
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    
    return fig


def render_advisory_chart(advisory_data):
    """Render advisory distribution."""
    colors_map = {'Adequate': '#00AA00', 'Watch': '#FFDD00', 
                  'Urgent': '#FF8800', 'Critical': '#FF0000'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(advisory_data.keys()),
            y=list(advisory_data.values()),
            marker_color=[colors_map[k] for k in advisory_data.keys()],
            text=[f"{v}%" for v in advisory_data.values()],
            textposition='auto',
            textfont=dict(size=14, color='white')
        )
    ])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        yaxis_title='Area (%)',
        font=dict(color='#adb5bd')
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    
    return fig


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Load demo data
    crops, ts_df, stress_data, advisory_data, advisory_table = generate_demo_data()
    
    # Sidebar
    controls = render_sidebar()
    
    # Header
    render_header()
    
    # Top Metrics
    render_metrics(crops, stress_data, advisory_data)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ---- Main Content Tabs ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Map View", "📈 Time Series", "🌾 Crop Analysis", 
        "💧 Stress & Advisory", "📱 Alerts & Export"
    ])
    
    # TAB 1: MAP VIEW
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🗺️ Interactive Map")
            m = render_map(controls)
            st_folium(m, width=None, height=550)
        
        with col2:
            st.markdown("### 📊 Quick Stats")
            st.plotly_chart(render_crop_distribution(crops), use_container_width=True)
            
            st.markdown("### 💧 Stress Overview")
            st.plotly_chart(render_stress_chart(stress_data), use_container_width=True)
    
    # TAB 2: TIME SERIES
    with tab2:
        st.markdown("### 📈 Multi-temporal Vegetation & SAR Analysis")
        st.plotly_chart(
            render_time_series(ts_df, controls['date_idx']),
            use_container_width=True
        )
        
        st.markdown("### 📊 Phenological Growth Stage")
        stages = ['Pre-season', 'Germination', 'Vegetative', 'Vegetative',
                   'Reproductive', 'Reproductive', 'Reproductive', 'Maturity']
        
        # Growth stage timeline
        stage_colors = {
            'Pre-season': '#808080', 'Germination': '#FFF9C4',
            'Vegetative': '#C8E6C9', 'Reproductive': '#BBDEFB',
            'Maturity': '#FFE0B2'
        }
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("🌱 **Germination** (Jun-Jul)\nNDVI: 0.15 → 0.30")
        with col2:
            st.success("🌿 **Vegetative** (Jul-Aug)\nNDVI: 0.30 → 0.65")
        with col3:
            st.warning("🌾 **Reproductive** (Aug-Oct)\nNDVI: 0.65 → 0.80 (Peak)")
        with col4:
            st.error("🍂 **Maturity** (Oct-Nov)\nNDVI: 0.80 → 0.25")
    
    # TAB 3: CROP ANALYSIS
    with tab3:
        st.markdown("### 🌾 Crop Type Classification Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Crop Distribution")
            st.plotly_chart(render_crop_distribution(crops), use_container_width=True)
        
        with col2:
            st.markdown("#### Classification Performance")
            
            # Simulated confusion matrix data
            cm_data = np.array([
                [45, 2, 1, 0, 1],
                [1, 38, 2, 1, 0],
                [0, 1, 35, 2, 0],
                [1, 0, 1, 30, 1],
                [0, 0, 0, 1, 28]
            ])
            labels = ['Rice', 'Cotton', 'Maize', 'Soybean', 'Sugarcane']
            
            fig = px.imshow(
                cm_data, x=labels, y=labels,
                color_continuous_scale='Blues',
                labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'},
                text_auto=True
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400,
                font=dict(color='#adb5bd')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Per-class metrics
        st.markdown("#### Per-Class Accuracy Metrics")
        metrics_df = pd.DataFrame({
            'Crop': ['Rice', 'Cotton', 'Maize', 'Soybean', 'Sugarcane'],
            'Precision': [0.96, 0.93, 0.90, 0.88, 0.93],
            'Recall': [0.92, 0.90, 0.92, 0.91, 0.93],
            'F1-Score': [0.94, 0.91, 0.91, 0.89, 0.93],
            'Support': [49, 42, 38, 33, 30]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # TAB 4: STRESS & ADVISORY
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💧 Moisture Stress Distribution")
            st.plotly_chart(render_stress_chart(stress_data), use_container_width=True)
        
        with col2:
            st.markdown("### 🚿 Irrigation Advisory")
            st.plotly_chart(render_advisory_chart(advisory_data), use_container_width=True)
        
        st.markdown("### 📋 Pixel-Level Irrigation Advisory Table")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            crop_filter = st.multiselect("Filter by Crop", options=advisory_table['Crop'].unique())
        with col2:
            stage_filter = st.multiselect("Filter by Stage", options=advisory_table['Stage'].unique())
        with col3:
            status_filter = st.multiselect("Filter by Status", options=advisory_table['Status'].unique())
        
        # Apply filters
        filtered = advisory_table.copy()
        if crop_filter:
            filtered = filtered[filtered['Crop'].isin(crop_filter)]
        if stage_filter:
            filtered = filtered[filtered['Stage'].isin(stage_filter)]
        if status_filter:
            filtered = filtered[filtered['Status'].isin(status_filter)]
        
        st.dataframe(
            filtered.style.background_gradient(subset=['Deficit (mm)'], cmap='YlOrRd'),
            use_container_width=True,
            height=400,
            hide_index=True
        )
    
    # TAB 5: ALERTS & EXPORT
    with tab5:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📱 SMS/WhatsApp Alert Preview")
            st.markdown("---")
            
            # Sample alerts
            alerts = [
                {"emoji": "🔴", "crop": "Rice", "stage": "Reproductive", 
                 "deficit": 32.5, "status": "Critical",
                 "msg": "IMMEDIATE irrigation required. Apply 35mm water depth."},
                {"emoji": "🟠", "crop": "Cotton", "stage": "Vegetative", 
                 "deficit": 22.1, "status": "Urgent",
                 "msg": "Irrigate within 1-2 days. Apply ~25mm water depth."},
                {"emoji": "🟡", "crop": "Maize", "stage": "Maturity", 
                 "deficit": 11.3, "status": "Watch",
                 "msg": "Schedule irrigation within 3-5 days. Monitor closely."},
            ]
            
            for alert in alerts:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-radius: 10px; 
                            padding: 1rem; margin-bottom: 0.8rem; border-left: 4px solid 
                            {'#FF0000' if alert['status']=='Critical' else '#FF8800' if alert['status']=='Urgent' else '#FFDD00'};">
                    <strong>{alert['emoji']} KrishiDrishti Alert</strong><br>
                    <strong>Crop:</strong> {alert['crop']} | <strong>Stage:</strong> {alert['stage']}<br>
                    <strong>Water Deficit:</strong> {alert['deficit']} mm/8-day<br>
                    <strong>Status:</strong> {alert['status']}<br>
                    <em>{alert['msg']}</em>
                </div>
                """, unsafe_allow_html=True)
            
            st.button("📤 Send Test SMS Alert", type="primary", use_container_width=True)
        
        with col2:
            st.markdown("### 📥 Export Options")
            st.markdown("---")
            
            st.markdown("#### 🗺️ Map Outputs")
            st.button("Download Crop Map (GeoTIFF)", use_container_width=True)
            st.button("Download Stress Map (GeoTIFF)", use_container_width=True)
            st.button("Download Advisory Map (GeoTIFF)", use_container_width=True)
            
            st.markdown("#### 📊 Reports")
            st.button("Download Full Report (PDF)", use_container_width=True)
            st.button("Download Advisory Table (CSV)", use_container_width=True)
            st.button("Download KML (Google Earth)", use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🏛️ Policy Alignment")
            st.info("**PMKSY** — Crop water demand monitoring")
            st.info("**PMFBY** — Crop insurance verification via stress maps")
            st.info("**Digital Agriculture Mission** — Automated crop mapping")
            st.info("**NMSA** — Climate-resilient farm management")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.85rem;">
        🛰️ <strong>KrishiDrishti</strong> — AI Crop Intelligence Platform | 
        Team BEST SHOT | PS 06<br>
        Powered by Sentinel-1/2, MODIS, NISAR | Google Earth Engine + Python + Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
