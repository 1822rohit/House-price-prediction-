# ============================================================
# PREMIUM HOUSE PRICE DASHBOARD — FINAL ERROR-FREE VERSION
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


# -----------------------------------------------------------
# Load CSS
# -----------------------------------------------------------
def load_css(path="style.css"):
    if Path(path).exists():
        with open(path, "r") as css:
            st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


# -----------------------------------------------------------
# Clean sqft data
# -----------------------------------------------------------
def convert_sqft(x):
    try:
        x = str(x).lower().strip()

        if "-" in x:
            p1, p2 = x.split("-")
            p1 = p1.strip().replace(",", "")
            p2 = p2.strip().replace(",", "")
            if p1.replace(".", "", 1).isdigit() and p2.replace(".", "", 1).isdigit():
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


# -----------------------------------------------------------
# Load model
# -----------------------------------------------------------
def load_model():
    if not Path("model.pkl").exists():
        st.error("❌ model.pkl missing.")
        st.stop()

    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        st.error("❌ model.pkl is corrupted.")
        st.stop()


# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
def load_data():
    if not Path("Pune_House_Data.csv").exists():
        st.error("❌ Pune_House_Data.csv not found.")
        st.stop()
    return pd.read_csv("Pune_House_Data.csv")


# -----------------------------------------------------------
# Streamlit Setup
# -----------------------------------------------------------
st.set_page_config(page_title="🏡 Premium House Dashboard", layout="wide")
load_css()

st.markdown("""
<div class="header">
    <h1>🏡 Premium House Price Prediction Dashboard</h1>
    <p>Stable, Clean & Production Ready</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# Load model + data
# -----------------------------------------------------------
model = load_model()
df = load_data()

df.columns = [c.strip() for c in df.columns]


# -----------------------------------------------------------
# TRAINING SCHEMA (IMPORTANT)
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# CLEAN DATA EXACTLY LIKE TRAINING
# -----------------------------------------------------------
df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna(subset=["total_sqft"])

categorical_cols = ["area_type", "availability", "size", "society", "site_location"]
numeric_cols = ["total_sqft", "bath", "balcony"]

for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df_clean = df.dropna(subset=numeric_cols)
df_clean = df_clean[training_columns + ["price"]]


# -----------------------------------------------------------
# KPI Section
# -----------------------------------------------------------
st.markdown("## 📊 Overview")

k1, k2, k3 = st.columns(3)
k1.metric("Total Properties", len(df_clean))
k2.metric("Avg Price", f"₹ {round(df_clean['price'].mean(),2)}")
k3.metric("Avg Area (sqft)", round(df_clean["total_sqft"].mean(), 2))

st.markdown("---")


# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📈 Analytics",
    "📉 Model Evaluation",
    "🤖 Predict Price"
])


# -----------------------------------------------------------
# TAB 1 — Dashboard
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# TAB 2 — Analytics
# -----------------------------------------------------------
with tab2:
    st.subheader("Correlation Heatmap")
    numeric = df_clean.select_dtypes(include=[np.number])

    if len(numeric.columns) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns.")


# -----------------------------------------------------------
# TAB 3 — MODEL EVALUATION (CRASH-PROOF)
# -----------------------------------------------------------
with tab3:
    st.subheader("Model Evaluation")

    if len(df_clean) < 10:
        st.warning("⚠ Not enough data to evaluate model (need ≥ 10 rows).")
        st.info(f"Rows available: {len(df_clean)}")

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

        try:
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
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Model evaluation failed: {e}")


# -----------------------------------------------------------
# TAB 4 — PREDICTION (DEBUG + SAFE)
# -----------------------------------------------------------
with tab4:
    st.subheader("Predict House Price (Safe & Debug)")

    # 1) Collect inputs from user
    user_inputs = {}
    col1, col2 = st.columns(2)
    for i, col in enumerate(training_columns):
        ui = col1 if i % 2 == 0 else col2
        if col in categorical_cols:
            opts = sorted(df_clean[col].dropna().unique())
            if len(opts) > 0:
                user_inputs[col] = ui.selectbox(col, opts)
            else:
                user_inputs[col] = ui.text_input(col)
        else:
            # numeric
            mn = float(df_clean[col].min())
            mx = float(df_clean[col].max())
            mv = float(df_clean[col].mean())
            user_inputs[col] = ui.number_input(col, value=mv, min_value=mn, max_value=mx)

    # 2) When user clicks predict: build df_in and align dtypes
    if st.button("🔮 Predict Price (Safe & Debug)"):
        # Build small DF from inputs
        df_in = pd.DataFrame([user_inputs])

        # --- Force types to match df_clean (your training-like cleaned df) ---
        # We inspect df_clean dtypes and cast df_in accordingly.
        dtype_map = df_clean.dtypes.to_dict()

        # Cast categorical -> str and numeric -> numeric
        for c in training_columns:
            if c in df_in.columns and c in dtype_map:
                target_dtype = dtype_map[c]
                if str(target_dtype).startswith("object") or c in categorical_cols:
                    # categorical: force string
                    df_in[c] = df_in[c].astype(str).str.strip()
                else:
                    # numeric: force numeric, coerce invalid -> NaN
                    df_in[c] = pd.to_numeric(df_in[c], errors="coerce")

        # Debug: show dtypes and values to user (very helpful)
        st.write("**Input dtypes (after casting):**")
        st.write(df_in.dtypes.astype(str).to_frame("dtype"))
        st.write("**Input values:**")
        st.dataframe(df_in)

        # Check numeric columns for NaN (invalid numeric input)
        bad_numeric = [c for c in numeric_cols if (c in df_in.columns and pd.isna(df_in.at[0, c]))]
        if bad_numeric:
            st.error(f"Please enter valid numeric values for: {bad_numeric}. Prediction aborted.")
            st.stop()

        # Ensure df_in has all training columns and in the same order
        missing_cols = [c for c in training_columns if c not in df_in.columns]
        if missing_cols:
            st.error(f"Missing columns required by model: {missing_cols}")
            st.stop()

        df_in = df_in[training_columns]

        # Attempt to show what the model thinks its input features are (best-effort)
        try:
            model_feature_info = None
            # try common attributes
            if hasattr(model, "feature_names_in_"):
                model_feature_info = list(model.feature_names_in_)
            elif hasattr(model, "named_steps") and "preprocess" in model.named_steps:
                pt = model.named_steps["preprocess"]
                if hasattr(pt, "feature_names_in_"):
                    model_feature_info = list(pt.feature_names_in_)
            st.write("**Model input feature names (if available):**")
            st.write(model_feature_info if model_feature_info is not None else "Not available from model object")
        except Exception as ex:
            st.write("Could not retrieve model feature names:", str(ex))

        # Final predict wrapped in try/except to surface full error
        try:
            prediction = model.predict(df_in)[0]
            st.success(f"🏠 Predicted Price: ₹ {round(prediction, 2)}")
        except Exception as e:
            st.error("Prediction failed — full exception shown below.")
            st.write(str(e))
            st.write("---- Debug snapshot ----")
            st.write("df_in.dtypes:")
            st.write(df_in.dtypes.astype(str).to_frame("dtype"))
            st.write("df_in values:")
            st.dataframe(df_in)
            # Optional: show small slice of df_clean to compare dtypes/values
            st.write("Sample rows from cleaned dataset (df_clean.dtypes):")
            st.write(df_clean.dtypes.astype(str).to_frame("dtype"))
            st.write("Sample rows from df_clean (first 5):")
            st.dataframe(df_clean.head(5))


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ by Rohit | Premium ML Dashboard v1.0")
