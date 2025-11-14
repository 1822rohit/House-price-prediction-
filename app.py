# app.py — Premium Advanced Dashboard + Prediction (robust & dynamic)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Helpers: Load CSS, Data, Model
# -----------------------------
def load_css(path="style.css"):
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # minimal inline styles if style.css missing
        st.markdown("""
        <style>
        .result-box {padding:16px;border-radius:10px;background:#f0f8ff;border-left:6px solid #1e88e5}
        .kpi {padding:12px;border-radius:12px;background:rgba(255,255,255,0.6);box-shadow:0 6px 18px rgba(0,0,0,0.04)}
        </style>
        """, unsafe_allow_html=True)

@st.cache_data
def load_data(path="Pune_House_Data.csv"):
    p = Path(path)
    if not p.exists():
        st.error(f"Dataset not found at `{path}`. Upload `Pune_House_Data.csv` to project root.")
        st.stop()
    df = pd.read_csv(p)
    return df

@st.cache_resource
def load_model(path="model.pkl"):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="🏡 Premium House Price Dashboard", layout="wide", page_icon="🏡")
load_css()

st.markdown("""
<div style="display:flex;align-items:center;gap:14px;">
  <div style="flex:1">
    <h1 style="margin:0">🏡 Premium House Price Prediction Dashboard</h1>
    <p style="margin:0;color:#5b6b7a">Interactive analytics, model metrics and real-time predictions</p>
  </div>
  <div style="text-align:right">
    <small style="color:#889aa8">Built by: Arihant Rathod</small>
  </div>
</div>
<hr>
""", unsafe_allow_html=True)

# -----------------------------
# Load resources
# -----------------------------
df = load_data()            # raw dataframe
model = load_model()        # may be None

# Work with a safe copy
df_clean = df.copy()

# Standardize column names (strip)
df_clean.columns = [c.strip() for c in df_clean.columns]

# Identify expected/available columns
all_columns = df_clean.columns.tolist()
target_col = "price"

# Try to detect the area column: prefer 'total_sqft' else 'area' else best numeric col
if "total_sqft" in all_columns:
    area_col = "total_sqft"
elif "area" in all_columns:
    area_col = "area"
else:
    # fallback: pick first numeric column that's not price
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    area_candidates = [c for c in numeric_cols if c != target_col]
    area_col = area_candidates[0] if area_candidates else None

# Make sure target exists
if target_col not in all_columns:
    st.error(f"The dataset does not contain required target column `{target_col}`. Please include it and re-run.")
    st.stop()

# Drop rows with missing target for metrics & modelling
df_model = df_clean.dropna(subset=[target_col]).copy()

# -----------------------------
# Sidebar: controls & uploads
# -----------------------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload dataset CSV (optional)", type=["csv"])
    if uploaded is not None:
        try:
            df_uploaded = pd.read_csv(uploaded)
            st.success("✅ Uploaded dataset loaded (will replace current view).")
            df_model = df_uploaded.dropna(subset=[target_col]).copy()
            df = df_uploaded.copy()
        except Exception as e:
            st.error("Failed to load uploaded CSV. Using existing dataset.")
    st.markdown("---")
    st.write("Model")
    if model is None:
        st.warning("No `model.pkl` found in project root. Upload or run training to create it.")
        model_upload = st.file_uploader("Upload model.pkl (optional)", type=["pkl"])
        if model_upload is not None:
            try:
                m = pickle.load(model_upload)
                # Save to disk for later use
                with open("model.pkl", "wb") as f:
                    pickle.dump(m, f)
                model = m
                st.success("model.pkl uploaded and saved.")
            except Exception as e:
                st.error("Uploaded file is not a valid pickle model.")
    else:
        st.success("✅ model.pkl loaded.")
        if st.button("🔁 Reload model"):
            # Force reload by clearing cache
            st.cache_resource.clear()
            model = load_model()

    st.markdown("---")
    st.write("Export")
    if st.button("📥 Download sample model.pkl"):
        # Offer download if model exists
        if model is None:
            st.info("No model to download.")
        else:
            with open("model.pkl", "rb") as f:
                st.download_button("Download model.pkl", data=f, file_name="model.pkl", mime="application/octet-stream")

    st.markdown("---")
    st.write("Notes")
    st.caption("This app dynamically adapts to dataset columns. Use filters on Overview page.")

# -----------------------------
# Calculate basic KPIs
# -----------------------------
st.markdown("## Overview")
col1, col2, col3, col4 = st.columns([1.2,1,1,1])

# KPI values computed safely with fallbacks
total_props = len(df_model)
avg_price = df_model[target_col].mean() if target_col in df_model.columns else np.nan
avg_area = df_model[area_col].mean() if area_col else np.nan
median_price = df_model[target_col].median()

col1.markdown(f"<div class='kpi'><h4>Total properties</h4><h2>{total_props}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='kpi'><h4>Avg Price</h4><h2>₹ {round(avg_price,2):,}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='kpi'><h4>Avg Area ({area_col})</h4><h2>{round(avg_area,2) if not np.isnan(avg_area) else 'N/A'}</h2></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='kpi'><h4>Median Price</h4><h2>₹ {round(median_price,2):,}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Tabs: Dashboard, EDA, Model, Predict
# -----------------------------
tab_dashboard, tab_eda, tab_model, tab_predict = st.tabs(["📊 Dashboard", "🔎 EDA", "📈 Model Performance", "🤖 Predict"])

# -----------------------------
# TAB: Dashboard (interactive filters + charts)
# -----------------------------
with tab_dashboard:
    st.subheader("Interactive Filters")
    # Identify a few common categorical columns if exist
    cat_cols = df_model.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target from numeric_cols if present
    numeric_cols = [c for c in numeric_cols if c != target_col]

    # create filters: up to 3 categorical filters
    filters = {}
    if cat_cols:
        for c in cat_cols[:3]:
            opts = ["All"] + sorted(df_model[c].dropna().unique().tolist())
            val = st.selectbox(f"Filter by {c}", options=opts, key=f"filter_{c}")
            filters[c] = val

    # Basic numeric range filter for area and price
    if area_col:
        min_area = float(df_model[area_col].min())
        max_area = float(df_model[area_col].max())
        rng = st.slider(f"{area_col} range", min_value=min_area, max_value=max_area, value=(min_area, max_area), step=(max_area-min_area)/100)
    else:
        rng = None

    # Apply filters
    df_filtered = df_model.copy()
    for c, v in filters.items():
        if v and v != "All":
            df_filtered = df_filtered[df_filtered[c] == v]

    if rng:
        df_filtered = df_filtered[(df_filtered[area_col] >= rng[0]) & (df_filtered[area_col] <= rng[1])]

    st.markdown(f"**Showing {len(df_filtered)} records** after filters.")
    st.dataframe(df_filtered.reset_index(drop=True), height=300, use_container_width=True)

    # Charts row
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Price distribution**")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df_filtered[target_col], bins=30, ax=ax, kde=True)
        ax.set_xlabel("Price")
        st.pyplot(fig)

    with c2:
        if area_col:
            st.markdown(f"**{area_col} vs Price**")
            fig, ax = plt.subplots(figsize=(6,3))
            ax.scatter(df_filtered[area_col], df_filtered[target_col], alpha=0.5)
            ax.set_xlabel(area_col)
            ax.set_ylabel("Price")
            st.pyplot(fig)
        else:
            st.info("No area-like numeric column found for scatter plot.")

# -----------------------------
# TAB: EDA (more plots & correlations)
# -----------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    st.write("Correlation heatmap (numeric features):")
    numeric = df_model.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

    st.markdown("### Top categorical value counts")
    for c in cat_cols[:4]:
        st.write(f"**{c}**")
        vc = df_model[c].value_counts().head(10)
        st.bar_chart(vc)

# -----------------------------
# TAB: Model Performance
# -----------------------------
with tab_model:
    st.subheader("Model Metrics & Diagnostics")
    if model is None:
        st.warning("No model loaded. Train locally using `train.py` and upload `model.pkl`, or upload model via sidebar.")
    else:
        # Evaluate on holdout split for quick metrics
        try:
            # Use rows without NaN for features used in model (we assume model can handle same columns)
            X = df_model.drop(columns=[target_col])
            y = df_model[target_col]
            # If model was trained with different columns, predictions may fail — handle gracefully
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)

            st.metric("R² (holdout)", f"{r2:.3f}")
            st.metric("RMSE (holdout)", f"{rmse:.3f}")

            # Residual plot
            fig, ax = plt.subplots(figsize=(6,3))
            ax.scatter(y_pred, y_test - y_pred, alpha=0.5)
            ax.axhline(0, color="red", linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            st.pyplot(fig)

            # Show a small sample of predictions vs actuals
            compare_df = pd.DataFrame({"actual": y_test.values, "predicted": y_pred})
            st.dataframe(compare_df.sample(min(10, len(compare_df))).reset_index(drop=True))
        except Exception as e:
            st.error("Model evaluation failed — the model may expect different feature columns or preprocessing. Error: " + str(e))

# -----------------------------
# TAB: Prediction (dynamic form)
# -----------------------------
with tab_predict:
    st.subheader("Make a Prediction")
    st.write("Fill the inputs and click **Predict**. Inputs are dynamically generated from dataset columns used for modelling.")

    if model is None:
        st.warning("No model loaded — prediction disabled.")
    else:
        # Build feature input form from the dataframe columns (excluding target)
        feature_cols = [c for c in df_model.columns if c != target_col]
        # We'll place inputs in two columns
        form_cols = st.columns(2)
        input_data = {}

        for i, col in enumerate(feature_cols):
            sample_val = df_model[col].dropna().sample(1).iloc[0] if df_model[col].dropna().shape[0] > 0 else None
            ui_col = form_cols[i % 2]

            if df_model[col].dtype == object or isinstance(sample_val, str):
                opts = sorted(df_model[col].dropna().unique().tolist())
                # limit options to first 200 to avoid huge selectbox
                if len(opts) > 200:
                    opts = opts[:200]
                input_data[col] = ui_col.selectbox(col, options=opts, index=0)
            else:
                # numeric input
                min_val = float(df_model[col].min())
                max_val = float(df_model[col].max())
                mean_val = float(df_model[col].mean())
                step = (max_val - min_val) / 100 if max_val > min_val else 1.0
                input_data[col] = ui_col.number_input(col, min_value=min_val, max_value=max_val, value=mean_val, step=step)

        if st.button("🔮 Predict", use_container_width=True):
            try:
                input_df = pd.DataFrame([input_data])
                pred = model.predict(input_df)[0]
                st.markdown(f"""
                <div class="result-box">
                    <h3 style="margin:0">🏠 Predicted Price</h3>
                    <p style="font-size:20px;margin:4px 0">₹ {round(pred,2):,}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error("Prediction failed. The model may expect different preprocessing or columns. Error: " + str(e))

# -----------------------------
# Footer
# -----------------------------
st.markdown("""---""")
st.caption("If you want automatic retrain when model missing, ask: 'auto train version banao' — I can provide an app variant that trains on startup if model.pkl absent.")
