# ============================================================
# PREMIUM HOUSE PRICE DASHBOARD — FULL FINAL FIXED VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# ---------------------------------------------------------
# 1. LOAD CUSTOM CSS
# ---------------------------------------------------------
def load_css(path="style.css"):
    p = Path(path)
    if p.exists():
        with open(path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. FIX: CLEAN total_sqft COLUMN
# ---------------------------------------------------------
def convert_sqft(x):
    try:
        x = str(x).lower().strip()

        # Case 1: "1000 - 1200"
        if "-" in x:
            parts = x.split("-")
            if len(parts) == 2:
                p1 = parts[0].strip().replace(",", "")
                p2 = parts[1].strip().replace(",", "")
                if p1.replace(".", "").isdigit() and p2.replace(".", "").isdigit():
                    return (float(p1) + float(p2)) / 2

        # Case 2: "34.46sq. meter"
        if "meter" in x:
            num = float(x.split("sq")[0].strip())
            return num * 10.7639   # m² → sqft

        # Case 3: "2.5 Acres"
        if "acre" in x:
            num = float(x.split("acre")[0].strip())
            return num * 43560   # acres → sqft

        # Case 4: pure numeric
        clean = x.replace(",", "")
        if clean.replace(".", "").isdigit():
            return float(clean)

    except:
        return None

    return None



# ---------------------------------------------------------
# 3. LOAD DATA SAFELY
# ---------------------------------------------------------
@st.cache_data
def load_data(path="Pune_House_Data.csv"):
    p = Path(path)
    if not p.exists():
        st.error("❌ Dataset 'Pune_House_Data.csv' not found in root folder.")
        st.stop()
    df = pd.read_csv(path)
    return df



# ---------------------------------------------------------
# 4. LOAD MODEL SAFELY
# ---------------------------------------------------------
@st.cache_resource
def load_model(path="model.pkl"):
    p = Path(path)
    if not p.exists():
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        st.error("❌ model.pkl found but could not be loaded.")
        st.stop()



# ---------------------------------------------------------
# 5. PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="🏡 Premium House Price Dashboard",
    layout="wide",
    page_icon="🏡"
)

load_css()



# ---------------------------------------------------------
# 6. HEADER UI
# ---------------------------------------------------------
st.markdown("""
<div class="header">
    <h1>🏡 Premium House Price Prediction Dashboard</h1>
    <p>Advanced ML-powered analytics + live predictions</p>
</div>
""", unsafe_allow_html=True)



# ---------------------------------------------------------
# 7. LOAD DATA + MODEL
# ---------------------------------------------------------
df = load_data()
model = load_model()

df_clean = df.copy()
df_clean.columns = [c.strip() for c in df_clean.columns]



# ---------------------------------------------------------
# 8. APPLY CLEANING TO total_sqft
# ---------------------------------------------------------
if "total_sqft" in df_clean.columns:
    df_clean["total_sqft"] = df_clean["total_sqft"].apply(convert_sqft)
    df_clean = df_clean.dropna(subset=["total_sqft"])



# ---------------------------------------------------------
# 9. DETECT AREA COLUMN
# ---------------------------------------------------------
if "total_sqft" in df_clean.columns:
    area_col = "total_sqft"
else:
    numerics = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    numerics = [c for c in numerics if c != "price"]
    area_col = numerics[0] if numerics else None



# ---------------------------------------------------------
# 10. SHOW KPIs
# ---------------------------------------------------------
total_props = len(df_clean)
avg_price = df_clean["price"].mean()
avg_area = df_clean[area_col].mean() if area_col else np.nan
median_price = df_clean["price"].median()

col1, col2, col3, col4 = st.columns(4)

col1.metric("🏠 Total Properties", total_props)
col2.metric("💰 Avg Price", f"₹ {round(avg_price, 2)}")
col3.metric(f"📏 Avg {area_col}", round(avg_area, 2))
col4.metric("⚖ Median Price", f"₹ {round(median_price, 2)}")

st.markdown("---")



# ---------------------------------------------------------
# 11. TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📈 Analytics",
    "📉 Model Performance",
    "🤖 Predict Price"
])



# ---------------------------------------------------------
# 12. TAB 1 — DASHBOARD
# ---------------------------------------------------------
with tab1:
    st.subheader("📊 Interactive Dashboard")

    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

    # Filters (only top 3 categorical)
    for col in cat_cols[:3]:
        options = ["All"] + sorted(df_clean[col].dropna().unique().tolist())
        selected = st.selectbox(f"Filter by {col}", options=options)
        if selected != "All":
            df_clean = df_clean[df_clean[col] == selected]

    st.dataframe(df_clean, use_container_width=True, height=350)

    c1, c2 = st.columns(2)

    with c1:
        st.write("### 💰 Price Distribution")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.histplot(df_clean["price"], bins=30, ax=ax)
        st.pyplot(fig)

    with c2:
        st.write(f"### 📐 {area_col} vs Price")
        fig, ax = plt.subplots(figsize=(5,3))
        ax.scatter(df_clean[area_col], df_clean["price"], alpha=0.5)
        st.pyplot(fig)



# ---------------------------------------------------------
# 13. TAB 2 — ANALYTICS
# ---------------------------------------------------------
with tab2:
    st.subheader("📈 Correlation Heatmap")

    num_df = df_clean.select_dtypes(include=[np.number])

    if num_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation analysis")



# ---------------------------------------------------------
# 14. TAB 3 — MODEL PERFORMANCE (SAFE VERSION)
# ---------------------------------------------------------
with tab3:
    st.subheader("📉 Model Performance")

    if model is None:
        st.warning("⚠ Cannot evaluate — model.pkl not found.")
    else:
        try:
            X = df_clean.drop("price", axis=1)
            y = df_clean["price"]

            # If dataset too small, avoid train-test crash
            if len(X) < 10:
                st.warning("⚠ Not enough data to evaluate model (needs at least 10 samples).")
                st.info(f"Dataset has only {len(X)} rows.")

                # Show prediction preview
                preds = model.predict(X)
                preview = pd.DataFrame({
                    "Actual Price": y.values,
                    "Predicted Price": preds
                })
                st.write("### 🔍 Prediction Preview")
                st.dataframe(preview.head(10), use_container_width=True)

            else:
                # Normal evaluation
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

                preds = model.predict(X_te)

                r2 = r2_score(y_te, preds)
                rmse = mean_squared_error(y_te, preds, squared=False)

                st.metric("📊 R² Score", round(r2, 3))
                st.metric("📉 RMSE", round(rmse, 3))

                st.write("### Residual Plot")
                fig, ax = plt.subplots()
                ax.scatter(preds, y_te - preds)
                ax.axhline(0, color="red", linestyle="--")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠ Model evaluation failed: {e}")



# ---------------------------------------------------------
# 15. TAB 4 — PREDICT PRICE
# ---------------------------------------------------------
with tab4:
    st.subheader("🤖 Predict House Price")

    if model is None:
        st.warning("⚠ Cannot predict — model.pkl not found.")
    else:
        inputs = {}
        Xcols = df_clean.drop("price", axis=1).columns.tolist()

        colA, colB = st.columns(2)

        for i, col in enumerate(Xcols):
            ui = colA if i % 2 == 0 else colB

            if df_clean[col].dtype == object:
                inputs[col] = ui.selectbox(col, sorted(df_clean[col].dropna().unique()))
            else:
                mn = float(df_clean[col].min())
                mx = float(df_clean[col].max())
                mv = float(df_clean[col].mean())
                inputs[col] = ui.number_input(col, mn, mx, mv)

        if st.button("🔮 Predict Price"):
            try:
                df_in = pd.DataFrame([inputs])
                prediction = model.predict(df_in)[0]
                st.success(f"🏠 Predicted Price: ₹ {round(prediction, 2)}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")



# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ by Rohit | Premium ML Dashboard v1.0")
