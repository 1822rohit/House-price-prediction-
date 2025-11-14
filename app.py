import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# LOAD CUSTOM CSS
# --------------------------------------------------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------------------------------------------------------------
# LOAD DATA & MODEL
# --------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Pune_House_Data.csv")  # your dataset

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

df = load_data()
model = load_model()

load_css()  # apply premium styling

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="🏡 Premium House Price Dashboard",
    layout="wide",
    page_icon="🏡"
)

# --------------------------------------------------------------
# HEADER
# --------------------------------------------------------------
st.markdown("""
<div class="header">
    <h1>🏡 Premium House Price Prediction Dashboard</h1>
    <p>Advanced ML-powered dashboard with real-time predictions</p>
</div>
""", unsafe_allow_html=True)

# MAIN TABS
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard", 
    "📈 Analytics", 
    "🤖 Prediction"
])

# =============================================================
# 📊 TAB 1 — PREMIUM DASHBOARD
# =============================================================
with tab1:
    st.markdown('<div class="section-title">📊 Overview Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # KPI CARDS (Animated Glass Cards)
    col1.markdown(f"""
        <div class="glass-card kpi-card">
            <h3>🏠 Total Properties</h3>
            <p>{df.shape[0]}</p>
        </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
        <div class="glass-card kpi-card">
            <h3>📏 Avg Area (sqft)</h3>
            <p>{round(df['area'].mean(), 2)}</p>
        </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
        <div class="glass-card kpi-card">
            <h3>💰 Avg Price</h3>
            <p>₹ {round(df['price'].mean(), 2)}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    st.markdown('<div class="section-subtitle">🔍 Filters</div>', unsafe_allow_html=True)
    colF1, colF2 = st.columns(2)

    bedrooms = colF1.selectbox("Bedrooms", ["All"] + sorted(df["bedrooms"].unique()))
    furnish = colF2.selectbox("Furnishing", ["All"] + df["furnishingstatus"].unique().tolist())

    df_filtered = df.copy()
    if bedrooms != "All":
        df_filtered = df_filtered[df_filtered["bedrooms"] == bedrooms]
    if furnish != "All":
        df_filtered = df_filtered[df_filtered["furnishingstatus"] == furnish]

    st.markdown('<div class="table-container">', unsafe_allow_html=True)
    st.dataframe(df_filtered, use_container_width=True, height=350)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    colG, colH = st.columns(2)

    with colG:
        st.markdown('<div class="chart-title">💰 Price Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.hist(df_filtered["price"], bins=25, alpha=0.7)
        st.pyplot(fig)

    with colH:
        st.markdown('<div class="chart-title">📐 Area Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.hist(df_filtered["area"], bins=25, alpha=0.7)
        st.pyplot(fig)


# =============================================================
# 📈 TAB 2 — ANALYTICS
# =============================================================
with tab2:
    st.markdown('<div class="section-title">📈 Data Analytics</div>', unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.markdown('<div class="chart-title">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        cax = ax.matshow(corr, cmap="coolwarm")
        fig.colorbar(cax)
        st.pyplot(fig)

    with colB:
        st.markdown('<div class="chart-title">📊 Area vs Price</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.scatter(df["area"], df["price"], alpha=0.5)
        ax.set_xlabel("Area")
        ax.set_ylabel("Price")
        st.pyplot(fig)

# =============================================================
# 🤖 TAB 3 — PREMIUM PREDICTION
# =============================================================
with tab3:
    st.markdown('<div class="section-title">🤖 Predict House Price</div>', unsafe_allow_html=True)

    st.markdown('<p class="predict-subtext">Enter property details below</p>', unsafe_allow_html=True)

    X = df.drop("price", axis=1)
    inputs = {}

    col1, col2 = st.columns(2)

    for i, col in enumerate(X.columns):
        if df[col].dtype == "object":
            ui_col = col1 if i % 2 == 0 else col2
            inputs[col] = ui_col.selectbox(col, df[col].unique())
        else:
            ui_col = col1 if i % 2 == 0 else col2
            inputs[col] = ui_col.number_input(
                col, 
                float(df[col].min()), 
                float(df[col].max()), 
                float(df[col].mean())
            )

    input_df = pd.DataFrame([inputs])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Price", use_container_width=True):
        pred = model.predict(input_df)[0]
        st.markdown(f"""
            <div class="result-box">
                <h2>🏠 Estimated Price</h2>
                <p>₹ {round(pred, 2)}</p>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
