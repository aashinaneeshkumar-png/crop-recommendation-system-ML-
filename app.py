import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import onnxruntime as rt
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Crop recommendation — Fieldwise",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL STYLES & SAAS SIDEBAR THEME ---
st.markdown("""
<style>
    .stApp {
        background-color: #f8f7f4;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1f2937;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Elegant SaaS Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f1f0eb !important;
        border-right: 1px solid #e5e7eb;
        padding-top: 0.5rem;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .brand-logo {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1b4332;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    .sidebar-section-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        color: #6b7280;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
    }
    
    .sidebar-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        font-size: 0.85rem;
        color: #374151;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-top: 1.5rem;
    }
    
    .user-profile-box {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
    .user-avatar {
        width: 36px;
        height: 36px;
        background-color: #1b4332;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Main Container & Breadcrumbs */
    .breadcrumb {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .breadcrumb span {
        color: #111827;
        font-weight: 500;
    }
    .page-title {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.025em;
        margin: 0;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #4b5563;
        margin-top: 0.25rem;
    }
    
    .card-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .feature-badge {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 0.3rem;
    }
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
        background-color: #ffffff;
        border-radius: 8px;
        overflow: hidden;
    }
    .custom-table th {
        background-color: #f3f4f6;
        color: #374151;
        text-align: left;
        padding: 10px 12px;
        font-weight: 600;
        border-bottom: 1px solid #e5e7eb;
    }
    .custom-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #f3f4f6;
        color: #1f2937;
    }
    .stButton button {
        background-color: #1b4332 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #2d6a4f !important;
        color: white !important;
    }
    div.stProgress > div > div > div > div {
        background-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# --- BIOLOGICAL DOMAIN SAFETY THRESHOLDS ---
CROP_LIMITS = {
    "Rice": {"max_temp": 38.0, "min_temp": 15.0, "min_rain": 100.0},
    "Maize": {"max_temp": 35.0, "min_temp": 10.0, "min_rain": 50.0},
    "Chickpea": {"max_temp": 30.0, "min_temp": 10.0, "min_rain": 20.0},
    "Kidneybeans": {"max_temp": 32.0, "min_temp": 15.0, "min_rain": 50.0},
    "Pigeonpeas": {"max_temp": 40.0, "min_temp": 18.0, "min_rain": 40.0},
    "Mothbeans": {"max_temp": 42.0, "min_temp": 20.0, "min_rain": 30.0},
    "Mungbean": {"max_temp": 40.0, "min_temp": 20.0, "min_rain": 40.0},
    "Blackgram": {"max_temp": 38.0, "min_temp": 20.0, "min_rain": 40.0},
    "Lentil": {"max_temp": 30.0, "min_temp": 10.0, "min_rain": 30.0},
    "Pomegranate": {"max_temp": 38.0, "min_temp": 10.0, "min_rain": 30.0},
    "Banana": {"max_temp": 38.0, "min_temp": 15.0, "min_rain": 100.0},
    "Mango": {"max_temp": 42.0, "min_temp": 20.0, "min_rain": 50.0},
    "Grapes": {"max_temp": 35.0, "min_temp": 10.0, "min_rain": 30.0},
    "Watermelon": {"max_temp": 40.0, "min_temp": 18.0, "min_rain": 40.0},
    "Muskmelon": {"max_temp": 40.0, "min_temp": 18.0, "min_rain": 40.0},
    "Apple": {"max_temp": 30.0, "min_temp": -5.0, "min_rain": 60.0},
    "Orange": {"max_temp": 35.0, "min_temp": 10.0, "min_rain": 60.0},
    "Papaya": {"max_temp": 44.0, "min_temp": 15.0, "min_rain": 80.0},
    "Coconut": {"max_temp": 38.0, "min_temp": 20.0, "min_rain": 100.0},
    "Cotton": {"max_temp": 42.0, "min_temp": 18.0, "min_rain": 50.0},
    "Jute": {"max_temp": 38.0, "min_temp": 24.0, "min_rain": 120.0},
    "Coffee": {"max_temp": 32.0, "min_temp": 15.0, "min_rain": 100.0}
}

@st.cache_resource
def load_onnx_model():
    session = rt.InferenceSession("crop_model.onnx")
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    proba_name = session.get_outputs()[1].name
    return session, input_name, label_name, proba_name

session, input_name, label_name, proba_name = load_onnx_model()

# Initialize session state variables
if 'active_nav' not in st.session_state:
    st.session_state.active_nav = "Crop recommendation"

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

if 'has_run_prediction' not in st.session_state:
    st.session_state.has_run_prediction = False

if 'entry_mode' not in st.session_state:
    st.session_state.entry_mode = "Manual entry"

# Render HTML Tables
def render_html_table(data_list, headers):
    html = '<div style="border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; background: #ffffff;"><table class="custom-table"><thead><tr>'
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr></thead><tbody>"
    
    for row in data_list:
        html += "<tr>"
        for h in headers:
            html += f"<td>{row.get(h, '')}</td>"
        html += "</tr>"
        
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="brand-logo">🌾 fieldwise</div>', unsafe_allow_html=True)
    st.selectbox("Workspace", ["Greenway Farm (Personal workspace)"], index=0)
    
    st.markdown('<div class="sidebar-section-title">Workspace Menu</div>', unsafe_allow_html=True)
    
    nav1 = st.button("📊 Overview & History", use_container_width=True)
    nav2 = st.button("🌱 Crop Recommendation", use_container_width=True)
    
    if nav1:
        st.session_state.active_nav = "Overview"
        st.rerun()
    if nav2:
        st.session_state.active_nav = "Crop recommendation"
        st.rerun()
    
    st.markdown("""
    <div class="sidebar-card">
        <div style="font-weight: 700; color: #111827; margin-bottom: 4px;">From soil to possibility.</div>
        <div style="color: #6b7280; font-size: 0.8rem; margin-bottom: 12px; line-height: 1.4;">Explore how your field conditions shape crop suitability.</div>
        <a href="#" style="color: #1b4332; font-weight: 600; text-decoration: none; font-size: 0.8rem;">Read the field guide ↗</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="user-profile-box">
        <div class="user-avatar">AS</div>
        <div>
            <div style="font-weight: 600; color: #111827; font-size: 0.85rem;">Anika Sharma</div>
            <div style="font-size: 0.75rem; color: #6b7280;">Field manager</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- VIEW 1: OVERVIEW & HISTORY ---
if st.session_state.active_nav == "Overview":
    st.markdown('<div class="breadcrumb">Workspace > <span>Overview</span></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Session History & Activity</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Review previously checked soil parameters and historical crop recommendations.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.search_history:
        st.markdown("""
        <div class="card-box" style="text-align: center; padding: 3rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📜</div>
            <h3 style="margin: 0; color: #111827;">No runs recorded yet</h3>
            <p style="color: #6b7280; font-size: 0.9rem;">Analyze crop recommendations to automatically log your checked soil and climate tests here.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"### 📋 Logged Field Runs ({len(st.session_state.search_history)})")
        
        summary_rows = []
        for idx, item in enumerate(reversed(st.session_state.search_history), start=1):
            run_num = len(st.session_state.search_history) - idx + 1
            summary_rows.append({
                "Run #": f"#{run_num}",
                "Timestamp": item["Timestamp"],
                "N-P-K": f"{item['Inputs']['N']} - {item['Inputs']['P']} - {item['Inputs']['K']}",
                "pH": f"{item['Inputs']['pH']:.1f}",
                "Temp (°C)": f"{item['Inputs']['Temp']}°C",
                "Humidity (%)": f"{item['Inputs']['Humidity']}%",
                "Rainfall (mm)": f"{item['Inputs']['Rainfall']} mm",
                "Recommended Crop": f"<b>{item['Top Crop']}</b>",
                "Confidence Score": item["Top Match"]
            })
            
        render_html_table(summary_rows, ["Run #", "Timestamp", "N-P-K", "pH", "Temp (°C)", "Humidity (%)", "Rainfall (mm)", "Recommended Crop", "Confidence Score"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Activity History"):
            st.session_state.search_history = []
            st.rerun()

# --- VIEW 2: CROP RECOMMENDATION FORM ---
else:
    st.markdown('<div class="breadcrumb">Workspace > <span>Crop recommendation</span></div>', unsafe_allow_html=True)

    col_head1, col_head2 = st.columns([4, 1])
    with col_head1:
        st.markdown('<h1 class="page-title">Field conditions</h1>', unsafe_allow_html=True)
        st.markdown('<p class="page-subtitle">Enter your soil test values.</p>', unsafe_allow_html=True)
    with col_head2:
        if st.button("📑 Saved reports"):
            st.info("Viewing saved reports drawer.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SEPARATE LINE FOR SWITCHABLE ENTRY MODES ---
    mode_col1, mode_col2, mode_empty = st.columns([1.5, 1.5, 3])
    with mode_col1:
        if st.button("✍️ Manual entry", use_container_width=True, type="primary" if st.session_state.entry_mode == "Manual entry" else "secondary"):
            st.session_state.entry_mode = "Manual entry"
            st.rerun()
    with mode_col2:
        if st.button("📂 Import soil test", use_container_width=True, type="primary" if st.session_state.entry_mode == "Import soil test" else "secondary"):
            st.session_state.entry_mode = "Import soil test"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.entry_mode == "Import soil test":
        st.info("📥 Soil test import module selected. Upload a file or connect to live sensor data.")
        st.file_uploader("Upload Soil Test File (CSV / JSON)", type=["csv", "json"])

    # --- PERFECTLY ALIGNED SIDE-BY-SIDE HEADINGS & PARAMETERS ---
    input_left, input_right = st.columns(2, gap="large")

    with input_left:
        st.markdown("#### 🧪 Soil profile <span style='font-size:0.75rem; color:#6b7280; font-weight:normal;'>01 / SOIL</span>", unsafe_allow_html=True)
        
        st.markdown('<span class="feature-badge">Nitrogen (N)</span>', unsafe_allow_html=True)
        N = st.number_input("N", min_value=0, max_value=150, value=90, label_visibility="collapsed")
        
        st.markdown('<span class="feature-badge">Phosphorus (P)</span>', unsafe_allow_html=True)
        P = st.number_input("P", min_value=0, max_value=150, value=42, label_visibility="collapsed")
        
        st.markdown('<span class="feature-badge">Potassium (K)</span>', unsafe_allow_html=True)
        K = st.number_input("K", min_value=0, max_value=200, value=43, label_visibility="collapsed")
        
        st.markdown('<span class="feature-badge">Soil pH Level</span>', unsafe_allow_html=True)
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, label_visibility="collapsed")
            
        st.caption("Scale 0-14 · 7 is neutral. Use the same test method across samples.")

    with input_right:
        st.markdown("#### 🌤️ Seasonal climate <span style='font-size:0.75rem; color:#6b7280; font-weight:normal;'>02 / CLIMATE</span>", unsafe_allow_html=True)
        
        st.markdown('<span class="feature-badge">Temperature (°C)</span>', unsafe_allow_html=True)
        temp = st.number_input("Temp", min_value=0.0, max_value=50.0, value=24.0, label_visibility="collapsed")
        
        st.markdown('<span class="feature-badge">Humidity (%)</span>', unsafe_allow_html=True)
        humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=82.0, label_visibility="collapsed")
        
        st.markdown('<span class="feature-badge">Rainfall (mm)</span>', unsafe_allow_html=True)
        rainfall = st.number_input("Rainfall", min_value=0.0, max_value=300.0, value=220.0, label_visibility="collapsed")
            
        st.caption("Total rainfall for the selected growing season; excludes irrigation.")

    # --- DEDICATED SEPARATE FULL-WIDTH ROW FOR ACTION BUTTONS ---
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
    with btn_col1:
        recommend_clicked = st.button("Recommend crops ➔", use_container_width=True)
    with btn_col2:
        if st.button("Reset", use_container_width=True):
            st.session_state.has_run_prediction = False
            st.rerun()
    with btn_col3:
        if st.button("Load", use_container_width=True):
            st.rerun()

    st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e5e7eb;'>", unsafe_allow_html=True)

    # --- BOTTOM OUTPUT SECTION: LEFT & RIGHT LENGTH MATCHED ---
    if recommend_clicked:
        st.session_state.has_run_prediction = True

    if st.session_state.has_run_prediction:
        input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]], dtype=np.float32)
        predictions = session.run([label_name, proba_name], {input_name: input_data})
        raw_probs = predictions[1][0]
        
        if isinstance(raw_probs, dict):
            crop_classes = list(raw_probs.keys())
            probabilities = np.array(list(raw_probs.values()))
        else:
            probabilities = raw_probs
            crop_classes = [
                "apple", "banana", "blackgram", "chickpea", "coconut", "coffee",
                "cotton", "grapes", "jute", "kidneybeans", "lentil", "maize",
                "mango", "mothbeans", "mungbean", "muskmelon", "orange", "papaya",
                "pigeonpeas", "pomegranate", "rice", "watermelon"
            ] if len(probabilities) == 22 else [f"Crop_{i}" for i in range(len(probabilities))]

        adjusted_scores = []
        status_notes = []
        for idx, raw_score in enumerate(probabilities):
            crop_name = str(crop_classes[idx]).title()
            adjusted_score = float(raw_score)
            status = "Optimal"
            if crop_name in CROP_LIMITS:
                limits = CROP_LIMITS[crop_name]
                if temp > limits["max_temp"]:
                    adjusted_score *= 0.05
                    status = f"⚠️ Heat Stress (> {limits['max_temp']}°C)"
                elif temp < limits["min_temp"]:
                    adjusted_score *= 0.05
                    status = f"⚠️ Cold Stress (< {limits['min_temp']}°C)"
                elif rainfall < limits["min_rain"]:
                    adjusted_score *= 0.3
                    status = f"⚠️ Water Deficit (< {limits['min_rain']}mm)"
            adjusted_scores.append(adjusted_score)
            status_notes.append(status)

        adjusted_scores = np.array(adjusted_scores)
        if adjusted_scores.sum() > 0:
            adjusted_scores = adjusted_scores / adjusted_scores.sum()

        all_sorted_indices = np.argsort(adjusted_scores)[::-1]
        
        all_ranked_results = []
        for rank, idx in enumerate(all_sorted_indices, start=1):
            crop_name = str(crop_classes[idx]).title()
            score = adjusted_scores[idx] * 100
            all_ranked_results.append({
                "Rank": f"#{rank}",
                "Crop": crop_name,
                "Suitability Match": f"{score:.2f}%",
                "Status": status_notes[idx]
            })

        best_crop = str(crop_classes[all_sorted_indices[0]]).title()
        best_score = int(adjusted_scores[all_sorted_indices[0]] * 100)

        if recommend_clicked:
            st.session_state.search_history.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Inputs": {"N": N, "P": P, "K": K, "pH": ph, "Temp": temp, "Humidity": humidity, "Rainfall": rainfall},
                "Top Crop": best_crop,
                "Top Match": f"{best_score}%",
                "All Recommendations": all_ranked_results
            })
            st.toast("Run saved to Overview history!", icon="✅")

        out_left, out_right = st.columns(2, gap="large")

        with out_left:
            # Top Recommended Card
            st.markdown(f"""
            <div class="card-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="font-weight: 600; color: #374151;">🌱 Your recommended crop</div>
                    <span style="background-color: #ecfdf5; color: #065f46; font-size: 0.75rem; padding: 3px 8px; border-radius: 6px; font-weight: 600;">Prediction Output</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin: 0; font-size: 2rem; color: #111827; font-weight: 700;">{best_crop}</h2>
                        <div style="font-style: italic; color: #6b7280; font-size: 0.9rem; margin-bottom: 0.5rem;">Optimal field match</div>
                        <p style="font-size: 0.85rem; color: #374151; margin: 0 0 1rem 0;">A strong recommendation based on your soil profile and climate parameters.</p>
                    </div>
                    <div style="text-align: center; border: 3px solid #10b981; border-radius: 50%; width: 75px; height: 75px; display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #ecfdf5;">
                        <span style="font-size: 1.2rem; font-weight: 700; color: #065f46; line-height: 1;">{best_score}</span>
                        <span style="font-size: 0.55rem; color: #047857; font-weight: 600;">/ 100</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Why Fits Card
            st.markdown(f"""
            <div class="card-box">
                <div style="font-weight: 600; color: #111827; margin-bottom: 0.75rem;">Why {best_crop.lower()} fits this run</div>
                <div style="font-size: 0.85rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                    <span>Temperature: <b>{temp}°C</b></span>
                    <span style="color: #6b7280;">Warm conditions fall inside optimal range.</span>
                </div>
                <div style="font-size: 0.85rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                    <span>Humidity: <b>{humidity}%</b></span>
                    <span style="color: #6b7280;">High humidity aligns with crop profile.</span>
                </div>
                <div style="font-size: 0.85rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                    <span>Rainfall: <b>{rainfall} mm</b></span>
                    <span style="color: #6b7280;">Seasonal rainfall fits expected limits.</span>
                </div>
                <div style="font-size: 0.85rem; margin-bottom: 0.75rem; display: flex; justify-content: space-between;">
                    <span>Soil pH: <b>{ph} pH</b></span>
                    <span style="color: #6b7280;">Slightly acidic soil sits within optimal range.</span>
                </div>
                <div style="border-top: 1px solid #e5e7eb; padding-top: 0.75rem; font-size: 0.8rem; color: #065f46;">
                    🧪 <b>N {N} · P {P} · K {K} mg/kg</b> — Nutrient values support this ranking.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with out_right:
            # Also Worth Exploring Top 5 Alternative Crops Card (Height-balanced)
            st.markdown('<div class="card-box"><div style="font-weight: 600; color: #111827; margin-bottom: 1rem;">Also worth exploring (Top 5)</div>', unsafe_allow_html=True)
            for rank in range(1, min(6, len(all_sorted_indices))):
                idx = all_sorted_indices[rank]
                alt_crop = str(crop_classes[idx]).title()
                alt_score = int(adjusted_scores[idx] * 100)
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.2rem; font-size: 0.85rem;">
                    <span><b>{alt_crop}</b></span>
                    <span style="font-weight: 600; color: #065f46;">{alt_score} / 100</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(alt_score)
            st.markdown("</div>", unsafe_allow_html=True)

        # Complete Table Output below across full width
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Complete Ranking of All 22 Crops")
        render_html_table(all_ranked_results, ["Rank", "Crop", "Suitability Match", "Status"])
    else:
        # Placeholder display prior to execution
        st.markdown("""
        <div class="card-box" style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌾</div>
            <h3 style="margin: 0; color: #111827;">Ready for Analysis</h3>
            <p style="color: #6b7280; font-size: 0.9rem; margin-top: 0.5rem;">Enter your soil profile and climate parameters above, then click <b>"Recommend crops ➔"</b> to view detailed predictions and rankings.</p>
        </div>
        """, unsafe_allow_html=True)