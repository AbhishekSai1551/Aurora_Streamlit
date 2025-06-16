import streamlit as st
import pandas as pd
import json
import requests
import plotly.graph_objects as go
import plotly.express as px
import datetime
import numpy as np

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:5000"

# --- User-specified island locations (N, W, S, E) ---
ISLAND_LOCATIONS = {
    "Caribbean Island 1 (12.8 N, 71.3 W)": {"N": 12.8, "W": 71.3, "S": 7.9, "E": 74.2},
    "Pacific Island 1 (12.4 N, 116.7 W)": {"N": 12.4, "W": 116.7, "S": 7.8, "E": 120.6},
    "Indian Ocean Island 1 (-7.4 N, 156.7 W)": {"N": -7.4, "W": 156.7, "S": -10.0, "E": 162.0},
    "Atlantic Island 1 (7.4 N, 72.3 W)": {"N": 7.4, "W": 72.3, "S": -0.9, "E": 74.3},
    "Maldives (7.0 N, 72.0 W)": {"N": 7.0, "W": 72.0, "S": -1.0, "E": 74.0},
    "Bermuda (32.5 N, 65.0 W)": {"N": 32.5, "W": -65.0, "S": 32.0, "E": -64.5},
    "Fiji (-15.0 N, 177.0 W)": {"N": -15.0, "W": 177.0, "S": -20.0, "E": 179.0},
    "Seychelles (-4.0 N, 55.0 W)": {"N": -4.0, "W": 55.0, "S": -5.0, "E": 56.0},
    "Hawaiian Islands (22.5 N, 160.0 W)": {"N": 22.5, "W": -160.0, "S": 18.0, "E": -154.5}, # Added
    "Galapagos Islands (-0.0 N, 90.0 W)": {"N": 0.0, "W": -90.0, "S": -1.5, "E": -89.0}, # Added
    "Custom Location": None
}

# Variables to display and analyze
TARGET_VARIABLES = ["swh1", "pp1d", "wind", "mwp", "shts"]
VARIABLE_NAMES = {
    "swh1": "Sig. Wave Height (1st Swell) [m]",
    "pp1d": "Peak Wave Period [s]",
    "wind": "Wind Speed [m/s]",
    "mwp": "Mean Wave Period [s]",
    "shts": "Sig. Height Total Swell [m]",
}

# --- Helper function to get prediction data from backend ---
@st.cache_data(ttl=3600) # Cache predictions for 1 hour
def get_predictions_from_backend(n_lat, w_lon, s_lat, e_lon, steps, prediction_date_str):
    params = {
        "n_lat": n_lat, "w_lon": w_lon, "s_lat": s_lat, "e_lon": e_lon,
        "steps": steps, "prediction_date": prediction_date_str
    }
    try:
        response = requests.get(f"{BACKEND_URL}/api/predict", params=params, timeout=300) # 5 min timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        return data
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the backend server at {BACKEND_URL}. Please ensure your Flask application is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Backend request timed out. The prediction might be taking too long. Consider reducing steps or selecting a smaller region.")
        return None
    except requests.exceptions.RequestException as e:
        error_message = f"Error getting predictions: {e}"
        if response is not None and response.text:
            try:
                error_detail = response.json().get('error', response.text)
                error_message = f"Error getting predictions: {error_detail}"
            except json.JSONDecodeError:
                error_message = f"Error getting predictions: {response.text}"
        st.error(error_message)
        return None

# --- Plotting Functions (adapted for new data structure) ---
def plot_variable_over_time(df_predictions, var_key, chart_title):
    fig = px.line(df_predictions,
                  x="forecast_time_iso",
                  y=f"{var_key}_mean",
                  title=chart_title,
                  labels={
                      "forecast_time_iso": "Forecast Time",
                      f"{var_key}_mean": VARIABLE_NAMES.get(var_key, var_key)
                  })
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text=VARIABLE_NAMES.get(var_key, var_key))
    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="AuroraWave Ocean Predictor")

st.title("AuroraWave Island Wave Prediction & Analysis")

st.markdown("""
    This application leverages the AuroraWave model to predict ocean wave conditions for specific island regions.
    Select a predefined island location or enter custom coordinates to get a statistical analysis of future wave parameters.
""")

# --- Sidebar for Inputs (Location Selector based on sketch) ---
st.sidebar.header("Location & Prediction Settings")

selected_location_name = st.sidebar.selectbox(
    "Select Predefined Island Location:",
    options=list(ISLAND_LOCATIONS.keys())
)

today = datetime.date.today()
# User requested "tomorrow", so default value is tomorrow
tomorrow = today + datetime.timedelta(days=1)
prediction_date = st.sidebar.date_input("Select Prediction Date:", value=tomorrow)


col_lat_n, col_lat_s = st.sidebar.columns(2)
col_lon_w, col_lon_e = st.sidebar.columns(2)

if selected_location_name and ISLAND_LOCATIONS[selected_location_name]:
    coords = ISLAND_LOCATIONS[selected_location_name]
    n_lat_default = coords["N"]
    w_lon_default = coords["W"]
    s_lat_default = coords["S"]
    e_lon_default = coords["E"]
    st.sidebar.markdown(f"**Selected Bounds:** N:{n_lat_default}, W:{w_lon_default}, S:{s_lat_default}, E:{e_lon_default}")
else: # Custom Location
    n_lat_default = 10.0 # sensible defaults for custom input
    w_lon_default = 70.0
    s_lat_default = 5.0
    e_lon_default = 75.0

n_lat = col_lat_n.number_input("North Latitude:", value=n_lat_default, format="%.1f", step=0.1)
s_lat = col_lat_s.number_input("South Latitude:", value=s_lat_default, format="%.1f", step=0.1)
w_lon = col_lon_w.number_input("West Longitude:", value=w_lon_default, format="%.1f", step=0.1)
e_lon = col_lon_e.number_input("East Longitude:", value=e_lon_default, format="%.1f", step=0.1)

if n_lat <= s_lat:
    st.sidebar.warning("North Latitude must be greater than South Latitude.")
if e_lon <= w_lon:
    st.sidebar.warning("East Longitude must be greater than West Longitude.")

selected_steps = st.sidebar.selectbox("Number of Prediction Steps (6-hour intervals):", options=list(range(1, 13)), index=1)

predict_button = st.sidebar.button("Get Predictions")

if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None
if 'prediction_region_info' not in st.session_state:
    st.session_state.prediction_region_info = None

if predict_button:
    if n_lat <= s_lat or e_lon <= w_lon:
        st.error("Invalid geographic coordinates: North Latitude must be > South Latitude, and East Longitude > West Longitude.")
    else:
        st.session_state.predictions_df = None
        st.session_state.prediction_region_info = None

        with st.spinner(f"Getting predictions for region ({n_lat},{w_lon},{s_lat},{e_lon}) on {prediction_date.strftime('%Y-%m-%d')} for {selected_steps} steps..."):
            raw_predictions = get_predictions_from_backend(
                n_lat, w_lon, s_lat, e_lon, selected_steps, prediction_date.strftime('%Y-%m-%d')
            )

            if raw_predictions:
                st.success("Predictions received successfully!")
                df = pd.DataFrame(raw_predictions)
                df['forecast_time_iso'] = pd.to_datetime(df['forecast_time_unix'], unit='s')
                st.session_state.predictions_df = df
                st.session_state.prediction_region_info = {
                    "lat_min": df['region_lat_min'].iloc[0],
                    "lat_max": df['region_lat_max'].iloc[0],
                    "lon_min": df['region_lon_min'].iloc[0],
                    "lon_max": df['region_lon_max'].iloc[0],
                    "spatial_resolution": df['spatial_resolution'].iloc[0],
                    "prediction_date": prediction_date.strftime('%Y-%m-%d')
                }
            else:
                st.warning("No predictions could be retrieved. Check backend logs for errors.")

if st.session_state.predictions_df is not None:
    st.header("Prediction Results")

    st.subheader("Region of Interest")
    if st.session_state.prediction_region_info:
        region = st.session_state.prediction_region_info
        st.write(f"**Analyzed Area:** Latitudes: {region['lat_min']:.2f}° to {region['lat_max']:.2f}°, "
                 f"Longitudes: {region['lon_min']:.2f}° to {region['lon_max']:.2f}° "
                 f"(Resolution: {region['spatial_resolution']:.2f}°)")
        st.write(f"**Prediction Date:** {region['prediction_date']}")
        
        # Simple map visualization
        center_lat = (region['lat_min'] + region['lat_max']) / 2
        center_lon = (region['lon_min'] + region['lon_max']) / 2
        st.map(pd.DataFrame([{'lat': center_lat, 'lon': center_lon}]))

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Time Series Charts", "Statistical Summary", "Raw Data"])

    with tab1:
        st.header("Time Series Charts")
        for var_key in TARGET_VARIABLES:
            plot_variable_over_time(st.session_state.predictions_df, var_key, f"{VARIABLE_NAMES.get(var_key, var_key)} Over Forecast")
    
    with tab2:
        st.header("Statistical Summary per Forecast Step")
        display_df = st.session_state.predictions_df.copy()
        display_df = display_df.drop(columns=[col for col in display_df.columns if col.startswith('region_') or col == 'spatial_resolution' or col == 'forecast_time_unix'])
        
        summary_cols = ['forecast_time_iso']
        for var_key in TARGET_VARIABLES:
            summary_cols.extend([f"{var_key}_mean", f"{var_key}_min", f"{var_key}_max", f"{var_key}_std"])
        
        st.dataframe(display_df[summary_cols].set_index('forecast_time_iso'), use_container_width=True)

        st.subheader("Broad Perspective on Small Islands")
        st.markdown("""
        **Applying Aurora Model to Small Islands:**
        
        The Aurora model, being a global foundation model, can provide predictions for any specific geographic region, including small islands, by extracting data relevant to their coordinates from its global output. The resolution of 0.25 degrees (approx 27km at equator) means that for very small islands, the predictions will represent the *average conditions within that 0.25x0.25 degree grid cell* covering the island, rather than highly localized effects right at the coastline or for tiny islets.
        
        **Statistical Analysis for Island Regions:**
        
        For small island regions, we are presenting the **mean, minimum, maximum, and standard deviation** of each predicted variable (Significant Wave Height of First Swell, Peak Wave Period, Wind Speed, Mean Wave Period, Significant Height of Total Swell) over the specified bounding box for each forecast step. This provides:
        * **Mean:** The overall expected value of the variable across the island region.
        * **Min/Max:** The range of values predicted within the region, indicating spatial variability.
        * **Standard Deviation:** A measure of the spread or dispersion of values, useful for understanding how uniform the conditions are across the region. A higher standard deviation indicates more variability within the small region.
        
        **Benefits for Island Analysis:**
        * **Early Warning:** Provides forecasts for critical wave parameters, crucial for coastal communities, shipping, and tourism.
        * **Resource Management:** Aids in planning for marine activities, disaster preparedness, and managing coastal resources.
        * **Climate Studies:** Contributes to understanding long-term wave climate changes around islands.
        
        **Limitations/Considerations:**
        * **Resolution:** For islands smaller than the 0.25-degree grid resolution, the model might not capture highly localized phenomena like wave refraction around complex coastlines or very shallow water effects.
        * **Bathymetry:** The model's accuracy might depend on how well its underlying data incorporates local bathymetry for specific near-shore wave dynamics.
        * **Data Availability:** The backend currently downloads data for the selected date, but real-time data ingestion for operational forecasts can be complex and may require access to live ECMWF/WeatherBench2 APIs which might have subscription costs or data latency.
        * **Forecast Frequency (Hourly vs. 6-hourly):** The Aurora model, as per the documentation, works with 6-hourly intervals (00:00, 06:00, 12:00, 18:00 UTC). The predictions will also be at these 6-hour intervals. Any "hourly" analysis would require post-processing interpolation, which may not accurately reflect actual hourly changes.
        """)


    with tab3:
        st.header("Raw Prediction Data")
        st.dataframe(st.session_state.predictions_df, use_container_width=True)

        csv = st.session_state.predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download All Prediction Data as CSV",
            data=csv,
            file_name=f"aurorawave_island_predictions_{selected_location_name.replace(' ', '_').replace(':', '').replace('.', '_')}_{prediction_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download all columns of the raw prediction data."
        )
