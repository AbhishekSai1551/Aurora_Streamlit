import streamlit as st
import plotly.graph_objects as go
import numpy as np
import requests
import datetime
import pandas as pd # For date handling

# --- BACKEND CONFIG ---
BACKEND_URL = "http://127.0.0.1:5000/api/predict" # Ensure this matches your Flask app URL

# --- PLOTTING FUNCTION ---
def create_interactive_map(data, lats, lons, title, unit, colorscale='RdYlBu_r'):
    """Create interactive geographic map with fixed color scaling"""
    # Ensure data is 2D (lat, lon)
    if data.ndim == 3: # If it's (1, lat, lon) or (time, lat, lon) from squeeze, take the first slice
        data = data[0]
    
    # Handle potential NaNs if data contains them
    if np.isnan(data).all():
        st.warning(f"No valid data to display for {title}. Showing empty map.")
        return go.Figure() # Return empty figure or a placeholder

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    
    # Adjust cmin/cmax for specific variables if known ranges are better
    if 'swh' in title.lower(): # Significant Wave Height
        vmin, vmax = 0, 10 # Example range in meters
    elif 'period' in title.lower() or 'pp1d' in title.lower() or 'mwp' in title.lower(): # Wave Period
        vmin, vmax = 0, 20 # Example range in seconds
    elif 'wind' in title.lower(): # Wind
        vmin, vmax = 0, 30 # Example range in m/s

    fig = go.Figure(data=go.Scattermapbox(
        lat=lat_grid.flatten(),
        lon=lon_grid.flatten(),
        mode='markers',
        marker=dict(
            size=8,
            color=data.flatten(),
            colorscale=colorscale,
            showscale=True,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title=unit),
            opacity=0.8
        ),
        text=[f'{val:.2f}' for val in data.flatten()],
        hovertemplate='<b>Latitude: %{lat:.2f}°</b><br>' +
                      '<b>Longitude: %{lon:.2f}°</b><br>' +
                      f'<b>{title}: %{{text}} {unit}</b><extra></extra>'
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=np.mean(lats), lon=np.mean(lons)),
            zoom=6 # Adjust zoom based on the cropped region size
        ),
        title=title,
        height=450,
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    return fig

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Aurora Ocean Wave Prediction",
    layout="wide"
)

# --- LOCATION SELECTOR ---
locations = {
    "Malé": (3.2, 73.2),            # North-central Indian Ocean
    "Port Louis": (-20.16, 57.50),   # Mauritius
    "Chennai": (13.08, 80.27),       # North-east (India)
    "Dar es Salaam": (-6.8, 39.28),  # West (Tanzania)
    "Perth": (-31.95, 115.86),       # East (Australia)
    "Muscat": (23.61, 58.59),        # North-west (Oman)
    "Maputo": (-25.97, 32.58),       # South-west (Mozambique)
    "Jakarta": (-6.21, 106.85),      # East (Indonesia)
    "Phuket": (7.88, 98.39)          # North-east (Thailand)
}
location_names = list(locations.keys())

# --- MAPPING AURORA OUTPUT VARIABLES ---
AURORA_VARIABLES_MAP = {
    "Significant Wave Height (SWH)": "swh", 
    "Primary Wave Mean Period (PP1D)": "pp1d",
    "Mean Wave Direction (MWD)": "mwd",
    "Mean Wave Period (MWP)": "mwp",
    "Wind Speed (Wind)": "wind", # Corresponds to 'wind' in backend
}

# Units for display
VARIABLE_UNITS = {
    "swh": "m", "pp1d": "s", "mwd": "deg", "mwp": "s", "wind": "m/s",
}

# Color scales for display
VARIABLE_COLORSCALES = {
    "swh": 'Blues', "pp1d": 'viridis', "mwd": 'twilight', "mwp": 'viridis', "wind": 'tempo',
}

# --- HEADER ROW ---
st.markdown("## Aurora Ocean Wave and Atmospheric Prediction")
col_map, col_selector = st.columns([2, 1])

with col_map:
    st.markdown("### Location Map")
    world_lats = [v[0] for v in locations.values()]
    world_lons = [v[1] for v in locations.values()]
    world_names = list(locations.keys())
    world_fig = go.Figure(go.Scattermapbox(
        lat=world_lats,
        lon=world_lons,
        mode='markers+text',
        marker=dict(size=12, color='red'),
        text=world_names,
        textposition="top right"
    ))
    world_fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=0, lon=0),
            zoom=1.2
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        height=350
    )
    st.plotly_chart(world_fig, use_container_width=True)

with col_selector:
    st.markdown("### Prediction Parameters")
    selected_location_name = st.selectbox("Select a location:", location_names, key="location_selector")
    
    center_lat, center_lon = locations[selected_location_name]
    
    region_half_width_lat = 2.5
    region_half_width_lon = 2.5
    
    lat_bounds = (max(-90.0, center_lat - region_half_width_lat), min(90.0, center_lat + region_half_width_lat))
    lon_bounds = (center_lon - region_half_width_lon, center_lon + region_half_width_lon)

    st.write(f"Region will be around Lat: ({lat_bounds[0]:.2f}, {lat_bounds[1]:.2f}), Lon: ({lon_bounds[0]:.2f}, {lon_bounds[1]:.2f})")

    # --- UPDATED DATE RANGE TO 2024-2025 ---
    # Be aware that WeatherBench2 HRES_T0 might not have data past 2022.
    # The ECMWF Open Data GCS (wave data) should be up-to-date.
    min_date = datetime.date(2024, 1, 1)
    max_date = datetime.date(2025, 6, 17) # Current date
    
    selected_date = st.date_input("Select Target Date (YYYY-MM-DD):", 
                                  value=datetime.date(2024, 1, 1), # Default to a date in new range
                                  min_value=min_date, max_value=max_date,
                                  key="date_selector")

    run_prediction_button = st.button("Run Prediction", key="run_button")

# --- RESULTS DISPLAY ---
st.markdown("---")
st.markdown("### Prediction Results")

# Initialize session state for predictions if not already present
if "predictions_data" not in st.session_state:
    st.session_state.predictions_data = None
if "prediction_lats" not in st.session_state:
    st.session_state.prediction_lats = None
if "prediction_lons" not in st.session_state:
    st.session_state.prediction_lons = None
if "forecast_times" not in st.session_state:
    st.session_state.forecast_times = []


if run_prediction_button:
    if selected_date:
        target_date_str = selected_date.strftime("%Y%m%d")
        
        request_body = {
            "target_date": target_date_str,
            "lat_bounds": lat_bounds,
            "lon_bounds": lon_bounds
        }
        
        st.info("Sending request to backend... This may take a moment.")
        with st.spinner("Fetching and processing data..."):
            try:
                response = requests.post(BACKEND_URL, json=request_body)
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                result = response.json()

                if result.get("status") == "success":
                    st.session_state.predictions_data = result["predictions"]
                    st.session_state.prediction_lats = np.array(result["lats"])
                    st.session_state.prediction_lons = np.array(result["lons"])
                    st.session_state.forecast_times = [
                        pd.to_datetime(t) for t in result["forecast_times"]
                    ]
                    st.success(f"Prediction successful! Data Prep: {result['data_prep_time']}, Model Rollout: {result['model_rollout_time']}")
                else:
                    st.error(f"Backend processing error: {result.get('error', 'Unknown error')}")
                    st.session_state.predictions_data = None # Clear previous results
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend server. Is it running? (http://127.0.0.1:5000)")
                st.session_state.predictions_data = None
            except requests.exceptions.Timeout:
                st.error("The request to the backend timed out.")
                st.session_state.predictions_data = None
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred during the request to the backend: {e}")
                st.session_state.predictions_data = None
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.predictions_data = None
    else:
        st.warning("Please select a target date.")

# Display results if available
if st.session_state.predictions_data:
    forecast_step_options = [f"{i}: {t.strftime('%Y-%m-%d %H:%M UTC')}" 
                             for i, t in enumerate(st.session_state.forecast_times)]
    
    selected_forecast_step_idx = st.selectbox(
        "Select Forecast Step (0 is initial analysis, 1 is ~6hr forecast, etc.):",
        options=list(range(len(st.session_state.forecast_times))),
        format_func=lambda x: forecast_step_options[x]
    )
    
    st.markdown(f"#### Predictions for {selected_location_name} at {st.session_state.forecast_times[selected_forecast_step_idx].strftime('%Y-%m-%d %H:%M UTC')}")

    # --- ONLY THESE 5 VARIABLES WILL BE DISPLAYED ---
    display_variable_keys = [
        "swh", "pp1d", "mwd", "mwp", "wind" 
    ]
    
    # Filter for variables that actually exist in the prediction data
    available_vars_for_display = [
        var for var in display_variable_keys 
        if var in st.session_state.predictions_data
    ]
    
    if not available_vars_for_display:
        st.warning("No requested variables available to display in the predictions for the selected date. "
                   "Check backend logs and BigQuery data availability for 'swh', 'pp1d', 'mwd', 'mwp', 'wind'.")
    else:
        # Displaying 5 variables: 3 in the first row, 2 in the second
        cols_per_row = 3
        
        # First row (3 plots)
        cols_row1 = st.columns(cols_per_row)
        for j in range(min(len(available_vars_for_display), cols_per_row)):
            var_key = available_vars_for_display[j]
            display_name = next((name for name, key in AURORA_VARIABLES_MAP.items() if key == var_key), var_key)
            
            with cols_row1[j]:
                data_for_map = np.array(st.session_state.predictions_data[var_key][selected_forecast_step_idx])
                fig = create_interactive_map(
                    data_for_map,
                    st.session_state.prediction_lats,
                    st.session_state.prediction_lons,
                    display_name,
                    VARIABLE_UNITS.get(var_key, "unit"),
                    VARIABLE_COLORSCALES.get(var_key, 'RdYlBu_r')
                )
                st.plotly_chart(fig, use_container_width=True)

        # Second row (remaining 2 plots)
        if len(available_vars_for_display) > cols_per_row:
            cols_row2 = st.columns(cols_per_row) # Create 3 columns, only use 2
            for j in range(cols_per_row, len(available_vars_for_display)):
                var_key = available_vars_for_display[j]
                display_name = next((name for name, key in AURORA_VARIABLES_MAP.items() if key == var_key), var_key)
                
                with cols_row2[j - cols_per_row]: # Use column 0 and 1 of the second row
                    data_for_map = np.array(st.session_state.predictions_data[var_key][selected_forecast_step_idx])
                    fig = create_interactive_map(
                        data_for_map,
                        st.session_state.prediction_lats,
                        st.session_state.prediction_lons,
                        display_name,
                        VARIABLE_UNITS.get(var_key, "unit"),
                        VARIABLE_COLORSCALES.get(var_key, 'RdYlBu_r')
                    )
                    st.plotly_chart(fig, use_container_width=True)
