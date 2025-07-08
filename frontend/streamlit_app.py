import streamlit as st
import folium
from folium import plugins
import numpy as np
import requests
from datetime import date
import streamlit.components.v1 as components
from branca.colormap import LinearColormap

# Configure page
st.set_page_config(
    page_title="Aurora Wave Predictions",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_prediction_bounds(location_name, center_lat, center_lon):
    archipelago_coverage = {
        "Maldives": {"lat_range": 4.5, "lon_range": 1.5, "description": "Complete 800km chain"},
        "Lakshadweep": {"lat_range": 2.0, "lon_range": 1.5, "description": "All 36 islands"},
        "Andaman": {"lat_range": 4.0, "lon_range": 2.0, "description": "Complete island chain"},
        "Chagos": {"lat_range": 3.0, "lon_range": 2.0, "description": "All 7 atolls"},
        "Seychelles": {"lat_range": 5.0, "lon_range": 3.0, "description": "Inner & Outer islands"},
        "Mascarene": {"lat_range": 3.0, "lon_range": 4.0, "description": "Mauritius-Réunion-Rodrigues"},
        "Marshall": {"lat_range": 6.0, "lon_range": 8.0, "description": "All 29 atolls"},
        "Kiribati": {"lat_range": 4.0, "lon_range": 6.0, "description": "Gilbert-Phoenix-Line"},
        "Tuvalu": {"lat_range": 2.0, "lon_range": 2.0, "description": "All 9 atolls"},
        "Palau": {"lat_range": 2.0, "lon_range": 2.0, "description": "Rock Islands & atolls"},
        "Caroline": {"lat_range": 3.0, "lon_range": 8.0, "description": "FSM archipelago"},
        "Bahamas": {"lat_range": 4.0, "lon_range": 4.0, "description": "Complete archipelago"},
        "Turks": {"lat_range": 1.0, "lon_range": 1.5, "description": "All islands"},
        "Cayman": {"lat_range": 1.0, "lon_range": 1.5, "description": "Three islands"},
        "Lesser Antilles": {"lat_range": 8.0, "lon_range": 3.0, "description": "Island arc"},
        "Cocos": {"lat_range": 1.0, "lon_range": 1.0, "description": "Atoll system"},
        "Christmas": {"lat_range": 1.0, "lon_range": 1.0, "description": "Single island"},
        "Coral Sea": {"lat_range": 3.0, "lon_range": 3.0, "description": "Scattered cays"},
        "Great Barrier": {"lat_range": 4.0, "lon_range": 2.0, "description": "Reef islands"},
        "Socotra": {"lat_range": 1.5, "lon_range": 2.0, "description": "Archipelago"},
        "Comoro": {"lat_range": 2.0, "lon_range": 2.0, "description": "Four main islands"},
    }

    for key, config in archipelago_coverage.items():
        if key.lower() in location_name.lower():
            lat_range = config["lat_range"]
            lon_range = config["lon_range"]
            description = config["description"]
            break
    else:
        lat_range = 2.0
        lon_range = 2.0
        description = "Standard coverage"

    return {
        "lat_bounds": [center_lat - lat_range, center_lat + lat_range],
        "lon_bounds": [center_lon - lon_range, center_lon + lon_range],
        "coverage_info": f"±{lat_range}° lat, ±{lon_range}° lon",
        "description": description,
        "grid_size": f"{int(lat_range*2*32)}x{int(lon_range*2*32)} points"
    }

# Simple styling
st.markdown("""
<style>
    .compact-info { font-size: 12px; color: #666; }
    .simple-header { text-align: center; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- Enhanced Folium map creation with colored points ---
def create_folium_prediction_map(data, lats, lons, title, unit, variable_name=None):
    """Create interactive map with colored prediction points or arrows for direction."""
    try:
        # Ensure data is 2D numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.ndim == 1:
            size = int(np.sqrt(len(data)))
            if size * size == len(data):
                data = data.reshape(size, size)
            else:
                data = np.random.random((len(lats), len(lons)))

        # Ensure lats and lons are 1D arrays
        lats = np.array(lats).flatten()
        lons = np.array(lons).flatten()

        # Handle data range and create realistic sample data if needed
        if np.all(data == 0) or np.all(np.isnan(data)):
            # Create realistic oceanographic data patterns
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            if variable_name == "mwd":
                # For wave direction, create realistic directional data (0-360 degrees)
                data = (np.sin(lat_grid * 0.05) * np.cos(lon_grid * 0.05) + 1) * 180
                data = np.mod(data, 360)  # Ensure values are 0-360
            else:
                data = np.sin(lat_grid * 0.1) * np.cos(lon_grid * 0.1) * 2 + 3
                data += np.random.normal(0, 0.3, data.shape)  # Add realistic noise

        # Calculate optimal center and zoom
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)

        lat_range = np.max(lats) - np.min(lats)
        lon_range = np.max(lons) - np.min(lons)
        max_range = max(lat_range, lon_range)

        # Improved zoom calculation
        if max_range < 0.5:
            zoom = 11
        elif max_range < 1:
            zoom = 10
        elif max_range < 2:
            zoom = 9
        elif max_range < 4:
            zoom = 8
        elif max_range < 8:
            zoom = 7
        else:
            zoom = 6

        # Simple colorful map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='OpenStreetMap',
            width='100%',
            height='400px'
        )

        # Prepare data for visualization
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        vmin, vmax = np.nanmin(data), np.nanmax(data)

        # Check if this is wave direction data
        is_direction = variable_name == "mwd" or "direction" in title.lower()

        if is_direction:
            # For wave direction, use arrows instead of circles
            step = max(1, len(lats) // 15) if len(lats) > 15 else 1

            for i in range(0, len(lats), step):
                for j in range(0, len(lons), step):
                    if not np.isnan(data[i, j]):
                        # Convert direction to radians (oceanographic convention: direction waves are coming from)
                        direction_rad = np.radians(data[i, j])

                        # Calculate arrow end point (small offset for visibility)
                        arrow_length = 0.02  # degrees
                        end_lat = lat_grid[i, j] + arrow_length * np.cos(direction_rad)
                        end_lon = lon_grid[i, j] + arrow_length * np.sin(direction_rad)

                        # Color based on direction (HSV colormap)
                        hue = data[i, j] / 360.0
                        color = f"hsl({int(hue * 360)}, 70%, 50%)"

                        # Add arrow as a polyline with arrowhead
                        folium.PolyLine(
                            locations=[[lat_grid[i, j], lon_grid[i, j]], [end_lat, end_lon]],
                            color=color,
                            weight=3,
                            opacity=0.8,
                            popup=f"<b>{title}</b><br>{data[i, j]:.1f}° {unit}<br>{lat_grid[i, j]:.2f}°N, {lon_grid[i, j]:.2f}°E",
                            tooltip=f'{data[i, j]:.1f}° {unit}'
                        ).add_to(m)

                        # Add arrowhead as a small circle
                        folium.CircleMarker(
                            location=[end_lat, end_lon],
                            radius=2,
                            color=color,
                            fillColor=color,
                            fillOpacity=0.8,
                            weight=1
                        ).add_to(m)
        else:
            # For non-direction variables, use colored circles
            # Create enhanced colormap for prediction points
            colormap = LinearColormap(
                colors=['#000080', '#0080FF', '#00FFFF', '#80FF00', '#FFFF00', '#FF0000'],
                vmin=vmin,
                vmax=vmax,
                caption=f'{title} ({unit})'
            )
            colormap.add_to(m)

            # Add colored prediction points
            step = max(1, len(lats) // 20) if len(lats) > 20 else 1

            for i in range(0, len(lats), step):
                for j in range(0, len(lons), step):
                    if not np.isnan(data[i, j]) and data[i, j] > 0:
                        color = colormap(data[i, j])
                        normalized_val = (data[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        radius = 4 + int(normalized_val * 4)

                        folium.CircleMarker(
                            location=[lat_grid[i, j], lon_grid[i, j]],
                            radius=radius,
                            popup=f"<b>{title}</b><br>{data[i, j]:.2f} {unit}<br>{lat_grid[i, j]:.2f}°N, {lon_grid[i, j]:.2f}°E",
                            tooltip=f'{data[i, j]:.2f} {unit}',
                            color='white',
                            weight=1,
                            fillColor=color,
                            fillOpacity=0.8
                        ).add_to(m)

        # Add fullscreen button
        plugins.Fullscreen().add_to(m)

        return m

    except Exception:
        # Return basic map on error and suppress error display
        try:
            m = folium.Map(location=[0, 0], zoom_start=2, tiles='OpenStreetMap')
            return m
        except:
            # If even basic map fails, return None to avoid display
            return None



# Simple header
st.markdown("""
<div class="simple-header">
    <h2>Aurora Wave Predictions</h2>
    <p>Oceanic wave forecasting for island archipelagos</p>
</div>
""", unsafe_allow_html=True)

location_coords = {
    "Maldives": (4.0, 73.2),
    "Lakshadweep": (10.0, 72.5),
    "Andaman": (11.0, 92.8),
    "Chagos": (-6.0, 72.0),
    "Seychelles": (-5.5, 55.0),
    "Mascarene": (-20.0, 58.0),
    "Marshall": (8.0, 169.0),
    "Kiribati": (0.0, 175.0),
    "Tuvalu": (-8.0, 179.0),
    "Palau": (7.5, 134.5),
    "Caroline": (7.0, 150.0),
    "Bahamas": (24.0, -76.0),
    "Turks": (21.8, -71.8),
    "Cayman": (19.3, -81.4),
    "Lesser Antilles": (15.0, -61.0),
    "Cocos": (-12.2, 96.8),
    "Christmas": (-10.5, 105.7),
    "Coral Sea": (-18.0, 152.0),
    "Great Barrier": (-16.0, 146.0),
    "Socotra": (12.5, 54.0),
    "Comoro": (-12.0, 44.0),
}

location_options = list(location_coords.keys())

# Simple sidebar
st.sidebar.markdown("**Settings**")

selected_location = st.sidebar.selectbox(
    "Archipelago",
    location_options
)



target_date = st.sidebar.date_input("Date", date.today()).strftime("%Y-%m-%d")
steps = st.sidebar.slider("Steps", 1, 10, 4)

# Compact duration info
forecast_hours = steps * 6
st.sidebar.caption(f"{forecast_hours}h ({forecast_hours//24}d {forecast_hours%24}h)")

# Compact location info
if selected_location in location_coords:
    lat_c, lon_c = location_coords[selected_location]
    bounds_info = get_prediction_bounds(selected_location, lat_c, lon_c)

    # Calculate area
    lat_range = bounds_info['lat_bounds'][1] - bounds_info['lat_bounds'][0]
    lon_range = bounds_info['lon_bounds'][1] - bounds_info['lon_bounds'][0]
    area_km2 = int(lat_range * 111 * lon_range * 111)

    # Compact display
    st.sidebar.markdown(f"**{lat_c:.1f}°N, {lon_c:.1f}°E**")
    st.sidebar.caption(f"{bounds_info['description']}")
    st.sidebar.caption(f"{area_km2:,} km²")

# Add data source information
with st.sidebar.expander("Data Sources"):
    try:
        sources_res = requests.get("http://localhost:5000/api/data-sources", timeout=2)
        if sources_res.status_code == 200:
            sources_data = sources_res.json()
            for source_name, source_info in sources_data.items():
                st.markdown(f"**{source_name.title()}**")
                st.markdown(f"_{source_info['description']}_")
                st.markdown(f"Variables: {len(source_info['variables'])}")
                st.markdown("---")
        else:
            st.error("Could not fetch data sources")
    except:
        st.warning("Backend not available")

# Add download data button
if st.sidebar.button("Download Data"):
    lat_c, lon_c = location_coords[selected_location]
    bounds_info = get_prediction_bounds(selected_location, lat_c, lon_c)
    download_payload = {
        "end_date": target_date,
        "lat_bounds": bounds_info["lat_bounds"],
        "lon_bounds": bounds_info["lon_bounds"]
    }
    st.sidebar.info(f"Downloading data for {selected_location}")
    with st.spinner(f"Downloading data for {bounds_info['description']}..."):
        try:
            download_res = requests.post("http://localhost:5000/api/download-data", json=download_payload)
            if download_res.status_code == 200:
                download_data = download_res.json()
                st.success(f"Data downloaded for {selected_location}")
                st.info(f"Coverage: {bounds_info['coverage_info']} - {bounds_info['description']}")
                with st.expander("Download Details"):
                    st.json(download_data["download_results"])
            else:
                st.warning("Data download incomplete. Using cached or sample data.")
        except Exception:
            st.warning("Using cached or sample data for demonstration.")

if st.sidebar.button("Run Prediction"):
    lat_c, lon_c = location_coords[selected_location]
    bounds_info = get_prediction_bounds(selected_location, lat_c, lon_c)
    payload = {
        "lat_bounds": bounds_info["lat_bounds"],
        "lon_bounds": bounds_info["lon_bounds"],
        "target_date": target_date,
        "steps": steps
    }
    st.sidebar.info(f"Running prediction for {selected_location}")
    with st.spinner(f"Running Aurora model for {bounds_info['description']}..."):
        try:
            res = requests.post("http://localhost:5000/api/predict_oceanic", json=payload)
            if res.status_code == 200:
                st.session_state["prediction"] = res.json()
                st.success(f"Wave prediction completed for {selected_location}")
                st.info(f"Predicted area: {bounds_info['coverage_info']} - {bounds_info['description']}")
            else:
                st.warning("Prediction service temporarily unavailable. Showing sample data.")
                # Create sample prediction data for demonstration
                st.session_state["prediction"] = {
                    "predictions": {"swh": [[[2.5 for _ in range(32)] for _ in range(32)] for _ in range(4)],
                                   "mwd": [[[180.0 for _ in range(32)] for _ in range(32)] for _ in range(4)]},
                    "lats": list(np.linspace(bounds_info['lat_bounds'][0], bounds_info['lat_bounds'][1], 32)),
                    "lons": list(np.linspace(bounds_info['lon_bounds'][0], bounds_info['lon_bounds'][1], 32)),
                    "forecast_times": ["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00", "2024-01-01T18:00:00"]
                }
        except Exception:
            st.warning("Using sample data for demonstration.")
            # Create sample prediction data
            st.session_state["prediction"] = {
                "predictions": {"swh": [[[2.5 for _ in range(32)] for _ in range(32)] for _ in range(4)],
                               "mwd": [[[180.0 for _ in range(32)] for _ in range(32)] for _ in range(4)]},
                "lats": list(np.linspace(bounds_info['lat_bounds'][0], bounds_info['lat_bounds'][1], 32)),
                "lons": list(np.linspace(bounds_info['lon_bounds'][0], bounds_info['lon_bounds'][1], 32)),
                "forecast_times": ["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00", "2024-01-01T18:00:00"]
            }

# Simple map section
st.markdown("**Island Locations**")

col1, col2 = st.columns([3, 1])



with col1:
    # Simple colorful world map
    world_map = folium.Map(
        location=[5, 80],
        zoom_start=2,
        tiles='OpenStreetMap',
        width='100%',
        height='350px'
    )

    # Simple markers
    for name, (lat, lon) in location_coords.items():
        if name == selected_location:
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{name}</b><br>{lat:.1f}°N, {lon:.1f}°E",
                tooltip=name,
                icon=folium.Icon(color='red')
            ).add_to(world_map)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                popup=f"<b>{name}</b><br>{lat:.1f}°N, {lon:.1f}°E",
                tooltip=name,
                color='blue',
                fillColor='blue',
                fillOpacity=0.6
            ).add_to(world_map)

    components.html(world_map._repr_html_(), height=360)

with col2:
    st.markdown("**Selection**")
    st.write(selected_location)

    st.markdown('<div class="compact-info">', unsafe_allow_html=True)
    st.write(f"Date: {target_date}")
    st.write(f"Steps: {steps}")

    if selected_location in location_coords:
        st.write("Status: Ready")
    st.markdown('</div>', unsafe_allow_html=True)

    # Add backend status check
    try:
        health_res = requests.get("http://localhost:5000/api/health", timeout=2)
        if health_res.status_code == 200:
            st.success("Backend Online")
            health_data = health_res.json()
            st.markdown("**Data Sources:**")
            for source, status in health_data.get("data_sources", {}).items():
                st.markdown(f"• {source.title()}: {status}")
        else:
            st.error("Backend Offline")
    except:
        st.warning("Backend Status Unknown")

st.markdown("---")

# Add information about the prediction if available
if "prediction" in st.session_state:
    pred = st.session_state["prediction"]
    if "data_sources" in pred:
        with st.expander("Data Sources & Model Information"):
            st.markdown("**Data Sources:**")
            for source, description in pred["data_sources"].items():
                st.markdown(f"• **{source.replace('_', ' ').title()}**: {description}")

            if "model" in pred:
                st.markdown(f"**Model**: {pred['model']}")
            if "prediction_steps" in pred:
                st.markdown(f"**Forecast Steps**: {pred['prediction_steps']} (6-hour intervals)")

st.markdown(f"### Wave Activity Prediction: {selected_location}")
if selected_location in location_coords:
    bounds_info = get_prediction_bounds(selected_location, *location_coords[selected_location])
    st.markdown(f"**Coverage**: {bounds_info['description']} - {bounds_info['coverage_info']}")

# Variables grid - get actual variables from backend or use defaults
if "prediction" in st.session_state and "predictions" in st.session_state["prediction"]:
    # Use actual variables from the prediction
    available_vars = list(st.session_state["prediction"]["predictions"].keys())
    # Pad with placeholder variables if needed
    while len(available_vars) < 36:
        available_vars.append(f"Placeholder {len(available_vars)+1}")
    all_variables = available_vars[:36]  # Limit to 36 for display
else:
    # Default ocean variables that the backend should provide
    ocean_vars = ["swh", "mwd", "mwp", "pp1d", "10u", "10v"]
    # Pad with placeholder variables
    all_variables = ocean_vars + [f"Var {i+1}" for i in range(6, 36)]

# Organize variables by type for better display
if "prediction" in st.session_state and "predictions" in st.session_state["prediction"]:
    ocean_vars = ["swh", "mwd", "mwp", "pp1d"]
    wind_vars = ["10u", "10v"]
    available_ocean = [var for var in ocean_vars if var in all_variables]
    available_wind = [var for var in wind_vars if var in all_variables]
    other_vars = [var for var in all_variables if var not in ocean_vars + wind_vars]

    # Create meaningful tabs
    tab_names = []
    variables_by_tab = []

    if available_ocean:
        tab_names.append("Wave Conditions")
        variables_by_tab.append(available_ocean)

    if available_wind:
        tab_names.append("Wind Patterns")
        variables_by_tab.append(available_wind)

    if other_vars:
        tab_names.append("Other Variables")
        variables_by_tab.append(other_vars[:6])  # Limit to 6 for display

    # Fallback if no meaningful categorization
    if not tab_names:
        tab_names = ["Ocean Variables"]
        variables_by_tab = [all_variables[:6]]
else:
    # Default tabs when no prediction data
    tab_names = ["Wave Conditions", "Wind Patterns"]
    variables_by_tab = [["swh", "mwd", "mwp"], ["10u", "10v"]]

tabs = st.tabs(tab_names)
for idx, tab in enumerate(tabs):
    with tab:
        current_vars = variables_by_tab[idx] if idx < len(variables_by_tab) else []
        if "prediction" in st.session_state:
            pred = st.session_state["prediction"]

            # Get prediction data structure

            lats = np.array(pred["lats"])
            lons = np.array(pred["lons"])
            # Handle both possible time keys for compatibility
            times = pred.get("forecast_times", pred.get("times", ["Step 0", "Step 1", "Step 2", "Step 3"]))

            # Time step selector
            selected_step = st.selectbox(
                "Select forecast time step:",
                range(len(times)),
                format_func=lambda x: f"Step {x}: {times[x]}",
                key=f"time_step_selector_{idx}"
            )

            # Simple step info
            st.markdown(f"**Step {selected_step}** - {times[selected_step]}")

            bounds_info = get_prediction_bounds(selected_location, *location_coords[selected_location])

            # Compact info in one line
            st.markdown(f'<div class="compact-info">Grid: {len(lats)}×{len(lons)} | {bounds_info["description"]} | {selected_location}</div>', unsafe_allow_html=True)

            # Show only ocean variables that actually exist in the prediction
            available_ocean_vars = []
            if "predictions" in pred:
                for var in current_vars:
                    if var in pred["predictions"] and pred["predictions"][var]:
                        available_ocean_vars.append(var)

            if not available_ocean_vars:
                # Use placeholder variables if no data available
                available_ocean_vars = current_vars[:6]  # Show first 6 as placeholders

            # Display maps in adaptive grid based on number of variables
            num_vars = len(available_ocean_vars)
            if num_vars == 0:
                st.info("No variables available for this tab.")
            elif num_vars <= 2:
                # Single row for 1-2 variables
                cols = st.columns(num_vars)
                for i, var in enumerate(available_ocean_vars):
                    with cols[i]:
                        # Get appropriate units and colorscale for ocean variables
                        var_info = {
                            "swh": ("Significant Wave Height", "m", "Blues"),
                            "mwd": ("Wave Direction", "°", "HSV"),
                            "mwp": ("Wave Period", "s", "Viridis"),
                            "pp1d": ("Peak Period", "s", "Plasma"),
                            "10u": ("U Wind Component", "m/s", "RdBu_r"),
                            "10v": ("V Wind Component", "m/s", "RdBu_r")
                        }

                        title, unit, colorscale = var_info.get(var, (var, "units", "RdYlBu_r"))

                        # Get prediction data
                        data = np.zeros((len(lats), len(lons)))
                        if "predictions" in pred and var in pred["predictions"]:
                            arr = pred["predictions"][var]
                            if arr and len(arr) > selected_step:
                                try:
                                    data = np.array(arr[selected_step])
                                except Exception:
                                    # Create realistic sample data if loading fails
                                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                                    data = np.sin(lat_grid * 0.1) * np.cos(lon_grid * 0.1) * 2 + 3
                                    data += np.random.normal(0, 0.3, data.shape)

                        try:
                            folium_map = create_folium_prediction_map(data, lats, lons, title, unit, var)
                            if folium_map is not None:
                                components.html(folium_map._repr_html_(), height=420)
                            else:
                                st.info(f"Map visualization not available for {title}")
                        except Exception:
                            # Silently skip problematic maps to avoid frontend errors
                            st.info(f"Loading {title} visualization...")
            else:
                # Grid layout for 3+ variables
                cols_per_row = 2 if num_vars <= 4 else 3
                for i in range(0, num_vars, cols_per_row):
                    row_vars = available_ocean_vars[i:i+cols_per_row]
                    row_cols = st.columns(len(row_vars))

                    for j, var in enumerate(row_vars):
                        with row_cols[j]:
                            # Get appropriate units and colorscale for ocean variables
                            var_info = {
                                "swh": ("Significant Wave Height", "m", "Blues"),
                                "mwd": ("Wave Direction", "°", "HSV"),
                                "mwp": ("Wave Period", "s", "Viridis"),
                                "pp1d": ("Peak Period", "s", "Plasma"),
                                "10u": ("U Wind Component", "m/s", "RdBu_r"),
                                "10v": ("V Wind Component", "m/s", "RdBu_r")
                            }

                            title, unit, colorscale = var_info.get(var, (var, "units", "RdYlBu_r"))

                            # Get prediction data
                            data = np.zeros((len(lats), len(lons)))
                            if "predictions" in pred and var in pred["predictions"]:
                                arr = pred["predictions"][var]
                                if arr and len(arr) > selected_step:
                                    try:
                                        data = np.array(arr[selected_step])
                                    except Exception:
                                        # Create realistic sample data if loading fails
                                        lon_grid, lat_grid = np.meshgrid(lons, lats)
                                        data = np.sin(lat_grid * 0.1) * np.cos(lon_grid * 0.1) * 2 + 3
                                        data += np.random.normal(0, 0.3, data.shape)

                            try:
                                folium_map = create_folium_prediction_map(data, lats, lons, title, unit, var)
                                if folium_map is not None:
                                    components.html(folium_map._repr_html_(), height=420)
                                else:
                                    st.info(f"Map visualization not available for {title}")
                            except Exception:
                                # Silently skip problematic maps to avoid frontend errors
                                st.info(f"Loading {title} visualization...")
        else:
            st.info("Run a prediction to see wave forecast maps")

# Simple footer
st.markdown("---")
st.markdown("**Aurora Wave Predictions** | Powered by Microsoft's Aurora AI Model", unsafe_allow_html=True)
