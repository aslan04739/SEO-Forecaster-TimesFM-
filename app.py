import streamlit as st
import timesfm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io
import os
from huggingface_hub import snapshot_download

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="TimesFM Forecaster", layout="wide")

# --- CACHED MODEL LOADING (CORRECTED) ---
@st.cache_resource
def load_timesfm_model():
    print("Downloading/Loading TimesFM Model...")
    
    # 1. Ensure the PyTorch model files are downloaded
    # This fixes the "No such file" error by pulling the correct repo
    snapshot_download(repo_id="google/timesfm-1.0-200m-pytorch")

    # 2. Configure Model Parameters
    hparams = timesfm.TimesFmHparams(
        context_len=512,
        horizon_len=365,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu",
    )
    
    # 3. Load Checkpoint from the PyTorch Repo
    checkpoint = timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
    )
    
    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
    return tfm

# --- PREDICTION FUNCTION ---
def make_forecast(tfm, df, target_col_name, horizon_days):
    """
    Handles data prep and forecasting for a specific column.
    """
    # 1. Prepare Data
    df = df.copy()
    original_columns = df.columns.tolist()
    normalized = {c: c.strip() for c in original_columns}
    df = df.rename(columns=normalized)

    # Find Date Column
    date_col = next((c for c in df.columns if c.lower() == 'date'), None)
    if not date_col:
        return None, None, f"Error: No 'Date' column found."

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Find Target Column
    target_col = next((c for c in df.columns if c.lower() == target_col_name.lower()), None)
    if not target_col:
        return None, None, f"Skipped: Column '{target_col_name}' not found."

    # Clean Data (Numeric/Percentage handling)
    if 'ctr' in target_col.lower():
        df[target_col] = df[target_col].astype(str).str.rstrip('%').astype(float).fillna(0)
    else:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)

    series_historique = df[target_col].values

    # 2. Forecast
    try:
        # TimesFM expects list of arrays
        means, _ = tfm.forecast([series_historique], freq=[0])
        prediction = means[0]
        # Trim prediction if the model output is longer than requested horizon
        prediction = prediction[:horizon_days]
    except Exception as e:
        return None, None, f"TimesFM Error on {target_col_name}: {e}"

    # 3. Create Result DataFrame
    last_date = df[date_col].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)
    
    df_result = pd.DataFrame({
        'date': future_dates,
        f'prediction_{target_col_name}': prediction
    })

    # 4. Create Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[date_col], df[target_col], label='History', alpha=0.7)
    ax.plot(future_dates, prediction, label='Prediction', color='red', linestyle='--')
    ax.set_title(f"Forecast: {target_col_name}")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return df_result, fig, "Success"

# --- MAIN APP UI ---
st.title("📈 AI SEO Forecaster (TimesFM)")
st.markdown("Upload your GSC data to generate forecasts for **Clicks, Impressions, CTR, and Position**.")

# 1. Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    horizon = st.slider("Prediction Horizon (Days)", min_value=30, max_value=365, value=90)
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

# 2. Main Logic
if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.success(f"Loaded data: {len(df_input)} rows.")
        
        if st.button("Generate Forecasts"):
            # Load Model
            with st.spinner("Loading AI Model (this takes a minute the first time)..."):
                tfm_model = load_timesfm_model()

            # Targets to process
            targets = ['Clicks', 'Impressions', 'CTR', 'Position']
            
            # Storage for zip file
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                
                # Create tabs for display
                tabs = st.tabs(targets)
                
                # Loop through metrics
                for i, target in enumerate(targets):
                    with tabs[i]:
                        st.subheader(f"Analyzing {target}...")
                        
                        # Run the prediction function
                        result_df, fig, status = make_forecast(tfm_model, df_input, target, horizon)
                        
                        if status == "Success":
                            # Show Graph
                            st.pyplot(fig)
                            
                            # Show Data preview
                            st.dataframe(result_df.head())
                            
                            # Add CSV to Zip
                            csv_data = result_df.to_csv(index=False).encode('utf-8')
                            zip_file.writestr(f"prediction_{target}.csv", csv_data)
                            
                            # Add Image to Zip
                            img_data = io.BytesIO()
                            fig.savefig(img_data, format='png', dpi=150)
                            zip_file.writestr(f"chart_{target}.png", img_data.getvalue())
                            
                        else:
                            st.warning(status)

            # Final Download Button
            st.success("All forecasts complete!")
            st.download_button(
                label="📥 Download All Results (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="seo_forecasts.zip",
                mime="application/zip"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to begin.")