# ============================================================
# PREMIUM HOUSE PRICE DASHBOARD — FINAL STABLE VERSION
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Load CSS
# ---------------------------------------------------------
def load_css(path="style.css"):
    if Path(path).exists():
        with open(path, "r") as css:
            st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Convert sqft safely
# ---------------------------------------------------------
def convert_sqft(x):
    try:
        x = str(x).lower().strip()

        if "-" in x:
            p = x.split("-")
            if len(p) == 2:
                p1, p2 = p[0].strip(), p[1].strip()
                if p1.replace('.', '', 1).isdigit() and p2.replace('.', '', 1).isdigit():
                    return (float(p1) + float(p2)) / 2

        if "meter" in x:
            num = float(x.split("sq")[0].strip())
            return num * 10.7639

        if "acre" in x:
            num = float(x.split("acre")[0].strip())
            return num * 43560

        clean = x.replace(",", "")
        if clean.replace(".", "", 1).isdigit():
            return float(clean)

    except:
        return None

    return None


# ---------------------------------------------------------
# Load model safely
# ---------------------------------------------------------
def load_model():
    if not Path("model.pkl").exists():
        st.error("❌ model.pkl not found in root folder.")
        st.stop()

    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        st.error("❌ model.pkl is corrupted.")
        st.stop()


# ---------------------------------------------------------
# Load dataset safely
# ---------------------------------------------------------
def load_data():
    if not Path("Pune_House_Data.csv").exists():
        st.error("❌ Pune_House_Data.csv not found.")
        st.stop()
    return pd.read_csv("Pune_House_Data.csv")


# ---------------------------------------------------------
# Streamlit Setup
# ---------------------------------------------------------
st.set_page_config(page_title="🏡 Premium Dashboard", layout="wide")
load_css()

st.markdown("""
<div class="header">
    <h1>🏡 Premium House Price Prediction Dashboard</h1>
    <p>Advanced, Reliable & Production-Ready</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Model + Data
# ---------------------------------------------------------
model = load_model()
df = load_data()


# ---------------------------------------------------------
# CLEANING EXACTLY LIKE TRAINING
# ---------------------------------------------------------
df.columns = [c.strip() for c in df.columns]

training_columns = [
    "area_type",
    "availability",
    "size",
    "society",
    "total_sqft",
    "bath",
    "balcony",
    "site_location"
]

# Clean sqft
df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna(subset=["total_sqft"])

# Enforce correct types
categorical_cols = ["area_type", "availability", "size", "society", "site_location"]
numeric_cols = ["total_sqft", "bath", "balcony"]

for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols)

# Keep only training schema + target
df_clean = df.copy()
df_clean = df_clean[training_columns + ["price"]]


# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------
st.markdown("## 📊 Overview")

total_rows = len(df_clean)
avg_price = df_clean["price"].mean()
avg_sqft = df_clean["total_sqft"].mean()

k1, k2, k3 = st.columns(3)
k1.metric("Total Properties", total_rows)
k2.metric("Avg Price", f"₹ {round(avg_price,2)}")
k3.metric("Avg Area (sqft)", round(avg_sqft, 2))

st.markdown("---")


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📈 Analytics",
    "📉 Model Evaluation",
    "🤖 Predict Price"
])


# ---------------------------------------------------------
# TAB 1 – Dashboard
# ---------------------------------------------------------
with tab1:
    st.subheader("Data Preview")
    st.dataframe(df_clean, height=350, use_container_width=True)

    cA, cB = st.columns(2)

    with cA:
        st.write("### Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df_clean["price"], bins=30, ax=ax)
        st.pyplot(fig)

    with cB:
        st.write("### Sqft vs Price")
        fig, ax = plt.subplots()
        ax.scatter(df_clean["total_sqft"], df_clean["price"], alpha=0.5)
        st.pyplot(fig)


# ---------------------------------------------------------
# TAB 2 – Analytics
# ---------------------------------------------------------
with tab2:
    st.subheader("Correlation Heatmap")

    numeric = df_clean.select_dtypes(include=[np.number])
    if len(numeric.columns) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns.")


# ---------------------------------------------------------
# TAB 3 – SAFE MODEL EVALUATION
# ---------------------------------------------------------
with tab3:
    st.subheader("Model Evaluation")

    if len(df_clean) < 10:
        st.warning("⚠ Not enough rows to evaluate model (need ≥ 10).")
        st.info(f"Available rows: {len(df_clean)}")

        try:
            preds = model.predict(df_clean[training_columns])
            preview = pd.DataFrame({
                "Actual Price": df_clean["price"],
                "Predicted Price": preds
            })
            st.write("### Prediction Preview")
            st.dataframe(preview.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Preview failed: {e}")

    else:
        X = df_clean[training_columns]
        y = df_clean["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        try:
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = mean_squared_error(y_test, preds, squared=False)

            st.metric("R² Score", round(r2, 3))
            st.metric("RMSE", round(rmse, 3))

            fig, ax = plt.subplots()
            ax.scatter(preds, y_test - preds)
            ax.axhline(0, color="red")
            ax.set_title("Residual Plot")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Model evaluation failed: {e}")


# ---------------------------------------------------------
# TAB 4 – Prediction
# ---------------------------------------------------------
with tab4:
    st.subheader("Predict House Price")

    user_input = {}

    c1, c2 = st.columns(2)

    for i, col in enumerate(training_columns):
        ui = c1 if i % 2 == 0 else c2

        if df_clean[col].dtype == object:
            user_input[col] = ui.selectbox(col, sorted(df_clean[col].unique()))
        else:
            mn, mx, mv = df_clean[col].min(), df_clean[col].max(), df_clean[col].mean()
            user_input[col] = ui.number_input(col, mn, mx, mv)

    if st.button("🔮 Predict Price"):
        try:
            df_in = pd.DataFrame([user_input])
            pred = model.predict(df_in)[0]
            st.success(f"🏠 Estimated Price: ₹ {round(pred, 2)}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ by Rohit | Premium ML Dashboard v1.0")
