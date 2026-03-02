import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="AI Manufacturing Intelligence",
    layout="wide"
)

# -----------------------
# Dark Theme Styling
# -----------------------
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0E1117;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Metric Card Background */
.stMetric {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
}

/* Metric Label (Hardness, Dissolution etc.) */
[data-testid="stMetricLabel"] {
    color: #BBBBBB !important;
    font-size: 16px !important;
}

/* Metric VALUE (96.02 etc.) */
[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 32px !important;
    font-weight: bold !important;
}

/* Metric Delta (green/red circle) */
[data-testid="stMetricDelta"] {
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Load Model
# -----------------------
model = joblib.load("C:/Users/bodap/OneDrive/Desktop/AIML_hackathon/xgboost_quality_model.pkl")

st.title("🏭 AI-Driven Pharmaceutical Manufacturing Intelligence")

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.title("⚙ Process Parameters")

gran_time = st.sidebar.slider("Granulation Time", 5.0, 30.0, 15.0)
binder = st.sidebar.slider("Binder Amount", 4.0, 12.0, 8.0)
dry_temp = st.sidebar.slider("Drying Temp", 50.0, 80.0, 60.0)
dry_time = st.sidebar.slider("Drying Time", 15.0, 40.0, 25.0)
comp_force = st.sidebar.slider("Compression Force", 5.0, 20.0, 12.0)
machine_speed = st.sidebar.slider("Machine Speed", 80.0, 250.0, 150.0)
lubricant = st.sidebar.slider("Lubricant Conc", 0.5, 2.0, 1.0)
moisture = st.sidebar.slider("Moisture Content", 0.5, 3.5, 2.0)

# -----------------------
# Prepare Input
# -----------------------
input_df = pd.DataFrame([[gran_time, binder, dry_temp, dry_time,
                          comp_force, machine_speed,
                          lubricant, moisture]],
                        columns=[
                            "Granulation_Time",
                            "Binder_Amount",
                            "Drying_Temp",
                            "Drying_Time",
                            "Compression_Force",
                            "Machine_Speed",
                            "Lubricant_Conc",
                            "Moisture_Content"
                        ])

prediction = model.predict(input_df)[0]

# -----------------------
# Create Tabs
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Quality Dashboard",
    "🏆 Golden Batch",
    "🧠 Explainability",
    "⚡ Energy Analytics"
])

# =====================================================
# TAB 1 — QUALITY DASHBOARD
# =====================================================
with tab1:

    st.header("📊 Predicted Quality Metrics")

    col1, col2, col3 = st.columns(3)

    def traffic_light(value, low, high):
        return "🟢" if low <= value <= high else "🔴"

    with col1:
        st.metric("Hardness",
                  round(prediction[0],2),
                  traffic_light(prediction[0],95,105))

        st.metric("Friability",
                  round(prediction[1],3),
                  traffic_light(prediction[1],0.3,1.0))

    with col2:
        st.metric("Dissolution",
                  round(prediction[2],2),
                  traffic_light(prediction[2],85,100))

        st.metric("Uniformity",
                  round(prediction[3],2),
                  traffic_light(prediction[3],95,105))

    with col3:
        st.metric("Disintegration",
                  round(prediction[4],2),
                  traffic_light(prediction[4],5,15))

# =====================================================
# TAB 2 — GOLDEN BATCH
# =====================================================
with tab2:

    st.header("🏆 Golden Batch Optimization")

    baseline_force = 11.5966
    golden_force = 9.123

    reduction = (baseline_force - comp_force) / baseline_force * 100

    st.metric("Compression Reduction (%)",
              round(reduction,2))

    st.write("Golden Batch Optimal Compression Force:", golden_force)

# =====================================================
# TAB 3 — SHAP EXPLAINABILITY
# =====================================================
with tab3:

    st.header("🧠 SHAP Explainability")

    explainer = shap.TreeExplainer(model.estimators_[0])
    shap_values = explainer.shap_values(input_df)

    fig = plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=input_df.columns
    )
    st.pyplot(fig)

# =====================================================
# TAB 4 — ENERGY ANALYTICS
# =====================================================
with tab4:

    st.header("⚡ Energy & Carbon Impact")

    baseline_energy = 76.3
    carbon_factor = 0.82

    estimated_energy = baseline_energy * (comp_force / baseline_force)
    estimated_carbon = estimated_energy * carbon_factor

    st.metric("Estimated Energy (kWh)",
              round(estimated_energy,2))

    st.metric("Estimated CO₂ (kg)",
              round(estimated_carbon,2))

    energy_data = pd.DataFrame({
        "Type":["Baseline","Current"],
        "Energy":[baseline_energy, estimated_energy]
    })

    st.bar_chart(energy_data.set_index("Type"))