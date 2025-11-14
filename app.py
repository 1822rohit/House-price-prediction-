# ============================================================
# PREMIUM HOUSE PRICE DASHBOARD — FINAL PRODUCTION VERSION
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# ============================================================
# Utility: Load CSS
# ============================================================
def load_css(path="style.css"):
    if Path(path).exists():
        with open(path, "r") as css:
            st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


# ============================================================
# Utility: Clean total_sqft input data
# ============================================================
def convert_sqft(x):
    try:
        x = str(x).lower().strip()

        # Case 1: Range "1000 - 1500"
        if "-" in x:
            p = x.split("-")
            if len(p) == 2:
                p1, p2 = p[0].strip(), p[1].strip()
                if p1.replace('.', '', 1).isdigit() and p2.replace('.', '', 1).isdigit():
                    return (float(p1) + float(p2)) / 2

        # Case 2: Square meter conversion
        if "meter" in x:
            num = float(x.split("sq")[0].strip())
            return num * 10.7639  # m² → sqft

        # Case 3: Acres conversion
        if "acre" in x:
            num = float(x.split("acre")[0].strip())
            return num * 43560  # acres → sqft

        # Case 4: Clean numeric
        clean = x.replace(",", "")
        if clean.replace(".", "", 1).isdigit():
            return float(clean)

    except:
        return None

    return None


# ============================================================
# Load Model Safely
# ============================================================
def load_model():
    model_path = Path("model.pkl")
    if not model_path.exists():
        st.error("❌ model.pkl missing. Upload it to the root directory.")
        st.stop()

    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except:
        st.error("❌ model.pkl could not be loaded. File is corrupted.")
        st.stop()


# ============================================================
# Load Dataset
# ============================================================
def load_data():
    csv_path = Path("Pune_House_Data.csv")
    if not csv_path.exists():
        st.error("❌ Pune_House_Data.csv missing in root directory.")
        st.stop()

    df = pd.read_csv(csv_path)
    return df


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(page_title="🏡 Premium House Price Dashboard",
                   layout="wide",
                   page_icon="🏡")

load_css()


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header">
    <h1>🏡 Premium House Price Prediction Dashboard</h1>
    <p>Cleaned, Optimized and Production Ready</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL + DATA
# ============================================================
model = load_model()
df = load_data()

# Strip column names
df.columns = [c.strip() for c in df.columns]

# Required training schema
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


# ============================================================
# KPI SECTION
# ============================================================
st.markdown("## 📊 Overview")

total_rows = len(df)
avg_price = df["price"].mean()
avg_sqft = df["total_sqft"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", total_rows)
c2.metric("Avg Price", f"₹ {round(avg_price,2)}")
c3.metric("Avg Area (sqft)", round(avg_sqft, 2))

st.markdown("---")


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📈 Analytics",
    "📉 Model Evaluation",
    "🤖 Predict Price"
])


# ============================================================
# TAB 1 — DASHBOARD
# ============================================================
with tab1:
    st.subheader("Interactive Data Preview")
    st.dataframe(df, use_container_width=True, height=350)

    colA, colB = st.columns(2)

    with colA:
        st.write("### Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["price"], bins=30, ax=ax)
        st.pyplot(fig)

    with colB:
        st.write("### Sqft vs Price")
        fig, ax = plt.subplots()
        ax.scatter(df["total_sqft"], df["price"], alpha=0.5)
        st.pyplot(fig)


# ============================================================
# TAB 2 — ANALYTICS
# ============================================================
with tab2:
    st.subheader("Correlation Heatmap")

    numeric = df.select_dtypes(include=[np.number])

    if len(numeric.columns) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for heatmap.")


# ============================================================
# TAB 3 — MODEL EVALUATION (SAFE)
# ============================================================
with tab3:
    st.subheader("Model Evaluation")

    # Check if good dataset
    if len(df) < 10:
        st.warning("⚠ Not enough rows for proper evaluation. Minimum required: 10.")
        st.info(f"Dataset contains only {len(df)} valid rows.")

        # Show preview predictions
        try:
            preds = model.predict(df[training_columns])
            preview = pd.DataFrame({
                "Actual Price": df["price"],
                "Predicted Price": preds
            })
            st.write("### Prediction Preview")
            st.dataframe(preview.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"⚠ Preview failed: {e}")

    else:
        # Normal evaluation
        X = df[training_columns]
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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


# ============================================================
# TAB 4 — PREDICTION FORM
# ============================================================
with tab4:
    st.subheader("Predict House Price")

    inputs = {}

    col1, col2 = st.columns(2)

    for i, col in enumerate(training_columns):
        ui = col1 if i % 2 == 0 else col2

        if df[col].dtype == object:
            inputs[col] = ui.selectbox(col, sorted(df[col].dropna().unique()))
        else:
            mn = float(df[col].min())
            mx = float(df[col].max())
            mv = float(df[col].mean())
            inputs[col] = ui.number_input(col, mn, mx, mv)

    if st.button("🔮 Predict Price"):
        try:
            df_in = pd.DataFrame([inputs])
            prediction = model.predict(df_in)[0]
            st.success(f"🏠 Predicted Price: ₹ {round(prediction, 2)}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Built with ❤️ by Rohit | Premium ML Dashboard v1.0")
