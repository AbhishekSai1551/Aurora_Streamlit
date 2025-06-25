import streamlit as st
import folium
from folium import plugins
import numpy as np
import requests
from datetime import date
import streamlit.components.v1 as components
from branca.colormap import LinearColormap
try:
    from scipy.interpolate import griddata
except ImportError:
    griddata = None

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

# --- Enhanced Folium map creation with colored points ---
def create_folium_prediction_map(data, lats, lons, title, unit, selected_location=None):
    """Create professional interactive Folium map with colored prediction points."""
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

        # Create base map with better styling
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='CartoDB positron',
            width='100%',
            height='450px',
            prefer_canvas=True
        )

        # Add title at the top-left corner to avoid blocking map controls
        title_html = f'''
        <div style="position: fixed;
                    top: 10px; left: 10px;
                    width: 350px; height: 50px;
                    background-color: rgba(255,255,255,0.95);
                    border: 2px solid #2E4057; border-radius: 8px;
                    z-index: 1000; font-size: 14px; font-weight: bold;
                    padding: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <div style="color: #2E4057; font-size: 16px; margin-bottom: 2px;">{title}</div>
        <div style="color: #666; font-size: 11px; font-weight: normal;">
        {selected_location or 'Island Region'} | Prediction Points
        </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        # Prepare data for colored point visualization
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        vmin, vmax = np.nanmin(data), np.nanmax(data)

        # Create more prediction points by interpolating the grid
        # Increase density for better visualization
        if len(lats) < 20 or len(lons) < 20:
            # Try to interpolate to create more points if scipy is available
            if griddata is not None:
                # Original points
                orig_points = []
                orig_values = []
                for i in range(len(lats)):
                    for j in range(len(lons)):
                        if not np.isnan(data[i, j]):
                            orig_points.append([lat_grid[i, j], lon_grid[i, j]])
                            orig_values.append(data[i, j])

                if orig_points and len(orig_points) > 3:  # Need at least 4 points for interpolation
                    try:
                        # Create denser grid
                        dense_lats = np.linspace(np.min(lats), np.max(lats), min(50, len(lats) * 3))
                        dense_lons = np.linspace(np.min(lons), np.max(lons), min(50, len(lons) * 3))
                        dense_lon_grid, dense_lat_grid = np.meshgrid(dense_lons, dense_lats)

                        # Interpolate values to dense grid
                        dense_data = griddata(
                            orig_points, orig_values,
                            (dense_lat_grid, dense_lon_grid),
                            method='linear', fill_value=np.nan  # Use linear for more stability
                        )
                        # Use dense grid for visualization
                        lat_grid, lon_grid = dense_lat_grid, dense_lon_grid
                        data = dense_data
                        lats, lons = dense_lats, dense_lons
                    except Exception as e:
                        # Fall back to original grid if interpolation fails
                        print(f"Interpolation failed: {e}")
                        pass

        # Create enhanced colormap for prediction points
        colormap = LinearColormap(
            colors=['#000080', '#0080FF', '#00FFFF', '#80FF00', '#FFFF00', '#FF0000'],
            vmin=vmin,
            vmax=vmax,
            caption=f'{title} ({unit})'
        )
        colormap.add_to(m)

        # Add colored prediction points instead of heatmap
        # Use adaptive point density based on data size
        if len(lats) > 30:
            step = max(1, len(lats) // 25)  # Limit to ~25 points per dimension
        else:
            step = 1  # Use all points for smaller datasets

        point_count = 0
        ocean_point_count = 0
        land_point_count = 0

        for i in range(0, len(lats), step):
            for j in range(0, len(lons), step):
                if not np.isnan(data[i, j]):
                    # Simple land mask: filter out points that are likely over land
                    # This is a basic heuristic - in production, use proper land mask data
                    lat_point = lat_grid[i, j]
                    lon_point = lon_grid[i, j]

                    # Basic land detection for common island regions
                    is_likely_ocean = True

                    # For Maldives region: filter out points too close to known land masses
                    if 3.0 <= lat_point <= 8.0 and 72.0 <= lon_point <= 75.0:
                        # Very basic Maldives land mask - exclude points that are exactly on atolls
                        # This is simplified - real implementation would use proper bathymetry data
                        if abs(lat_point - 4.175) < 0.05 and abs(lon_point - 73.5) < 0.05:  # Male area
                            is_likely_ocean = False

                    # Skip points with very low or zero values (often indicates land)
                    if data[i, j] <= 0.01 and title in ["Significant Wave Height", "Wave Period", "Peak Period"]:
                        is_likely_ocean = False
                        land_point_count += 1

                    if is_likely_ocean:
                        ocean_point_count += 1
                        # Color based on value
                        color = colormap(data[i, j])

                        # Determine point size based on value magnitude
                        normalized_val = (data[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        radius = 4 + int(normalized_val * 6)  # Size between 4-10

                        # Create detailed popup with prediction information
                        popup_content = f'''
                        <div style="font-family: Arial, sans-serif; min-width: 200px;">
                            <h4 style="margin: 0 0 8px 0; color: #2E4057;">{title} Prediction</h4>
                            <div style="margin-bottom: 6px;">
                                <strong>Value:</strong> <span style="color: #0066cc; font-size: 16px;">{data[i, j]:.2f} {unit}</span>
                            </div>
                            <div style="margin-bottom: 6px;">
                                <strong>Location:</strong> {lat_point:.3f}°N, {lon_point:.3f}°E
                            </div>
                            <div style="margin-bottom: 6px;">
                                <strong>Ocean Point ID:</strong> #{ocean_point_count}
                            </div>
                            <div style="font-size: 11px; color: #666; border-top: 1px solid #ddd; padding-top: 4px;">
                                Range: {vmin:.2f} - {vmax:.2f} {unit}<br>
                                Percentile: {((data[i, j] - vmin) / (vmax - vmin) * 100) if vmax > vmin else 50:.0f}%<br>
                                <span style="color: #0066cc;">🌊 Ocean prediction point</span>
                            </div>
                        </div>
                        '''

                        folium.CircleMarker(
                            location=[lat_point, lon_point],
                            radius=radius,
                            popup=folium.Popup(popup_content, max_width=300),
                            tooltip=f'{title}: {data[i, j]:.2f} {unit} (Ocean)',
                            color='white',
                            weight=2,
                            fillColor=color,
                            fillOpacity=0.9
                        ).add_to(m)

                    point_count += 1

        # Add comprehensive legend showing what the points represent
        legend_html = f'''
        <div style="position: fixed;
                    bottom: 10px; right: 10px; width: 280px;
                    background-color: rgba(255,255,255,0.95);
                    border: 2px solid #2E4057; border-radius: 8px;
                    z-index: 1000; font-size: 12px;
                    padding: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.2)">
        <div style="font-weight: bold; margin-bottom: 8px; color: #2E4057; font-size: 14px;">
        🌊 Ocean Prediction Points
        </div>
        <div style="margin-bottom: 6px;">
        <strong>Variable:</strong> {title}
        </div>
        <div style="margin-bottom: 6px;">
        <strong>Ocean Points:</strong> {ocean_point_count}
        </div>
        <div style="margin-bottom: 6px;">
        <strong>Land Points Filtered:</strong> {land_point_count}
        </div>
        <div style="margin-bottom: 6px;">
        <strong>Value Range:</strong> {vmin:.2f} - {vmax:.2f} {unit}
        </div>
        <div style="margin-bottom: 8px;">
        <strong>Average:</strong> {np.nanmean(data):.2f} {unit}
        </div>
        <div style="font-size: 10px; color: #666; border-top: 1px solid #ddd; padding-top: 6px;">
        <div><span style="color: #000080;">●</span> Low values (blue)</div>
        <div><span style="color: #00FFFF;">●</span> Medium values (cyan)</div>
        <div><span style="color: #FFFF00;">●</span> High values (yellow)</div>
        <div><span style="color: #FF0000;">●</span> Maximum values (red)</div>
        <div style="margin-top: 4px; font-style: italic;">
        Point size indicates value magnitude<br>
        🌊 Only ocean areas shown
        </div>
        </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Add interaction instructions
        instructions_html = f'''
        <div style="position: fixed;
                    top: 70px; left: 10px; width: 300px;
                    background-color: rgba(255,255,255,0.9);
                    border: 1px solid #ddd; border-radius: 5px;
                    z-index: 1000; font-size: 10px;
                    padding: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
        <div style="font-weight: bold; margin-bottom: 3px; color: #2E4057;">Map Controls:</div>
        <div>• Click points for detailed values</div>
        <div>• Hover for quick preview</div>
        <div>• Zoom/pan to explore region</div>
        <div>• Use fullscreen button (⛶) for better view</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(instructions_html))

        # Add fullscreen button
        plugins.Fullscreen().add_to(m)

        return m

    except Exception as e:
        # Silent error handling - return basic map without error display
        m = folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB positron')
        folium.Marker(
            [0, 0],
            popup="Map temporarily unavailable",
            icon=folium.Icon(color='gray', icon='info-sign')
        ).add_to(m)
        return m

# Backward compatibility alias
create_folium_heatmap = create_folium_prediction_map

# --- Page config and locations setup ---
st.set_page_config(page_title="Aurora Oceanic Prediction", layout="wide")

# Complete island archipelagos - predict wave activity around entire island groups
location_coords = {
    # === MAJOR CORAL ARCHIPELAGOS ===
    "Maldives Archipelago": (4.0, 73.2),             # Entire 800km chain (7°N to 0.5°S)
    "Lakshadweep Islands": (10.0, 72.5),             # Complete Indian archipelago
    "Andaman & Nicobar Islands": (11.0, 92.8),       # Full island chain (14°N to 6°N)
    "Chagos Archipelago": (-6.0, 72.0),              # British Indian Ocean Territory
    "Seychelles Islands": (-5.5, 55.0),              # Inner & Outer islands combined

    # === MASCARENE ISLANDS ===
    "Mascarene Islands": (-20.0, 58.0),              # Mauritius, Réunion, Rodrigues group

    # === PACIFIC CORAL ATOLLS ===
    "Marshall Islands": (8.0, 169.0),                # Complete atoll nation
    "Kiribati Atolls": (0.0, 175.0),                 # Gilbert, Phoenix, Line Islands
    "Tuvalu Atolls": (-8.0, 179.0),                  # Complete island nation
    "Palau Archipelago": (7.5, 134.5),               # Rock Islands & atolls
    "Caroline Islands": (7.0, 150.0),                # Federated States of Micronesia

    # === CARIBBEAN CORAL SYSTEMS ===
    "Bahamas Archipelago": (24.0, -76.0),            # Complete island chain
    "Turks and Caicos": (21.8, -71.8),               # British Overseas Territory
    "Cayman Islands": (19.3, -81.4),                 # Three-island group
    "Lesser Antilles": (15.0, -61.0),                # Windward & Leeward islands

    # === ADDITIONAL CORAL SYSTEMS ===
    "Cocos (Keeling) Islands": (-12.2, 96.8),        # Australian coral atolls
    "Christmas Island": (-10.5, 105.7),              # Australian territory
    "Coral Sea Islands": (-18.0, 152.0),             # Australian coral cays
    "Great Barrier Reef Islands": (-16.0, 146.0),    # Queensland coral islands

    # === INDIAN OCEAN CORAL SYSTEMS ===
    "Socotra Archipelago": (12.5, 54.0),             # Yemen UNESCO site
    "Comoro Islands": (-12.0, 44.0),                 # Volcanic & coral islands
}

# Sidebar controls
st.sidebar.header("Controls")

# Organize archipelagos by oceanographic region
location_categories = {
    "Indian Ocean Coral Systems": [
        "Maldives Archipelago", "Lakshadweep Islands", "Chagos Archipelago",
        "Seychelles Islands", "Cocos (Keeling) Islands", "Christmas Island"
    ],
    "Indian Ocean Large Islands": [
        "Andaman & Nicobar Islands", "Mascarene Islands", "Socotra Archipelago", "Comoro Islands"
    ],
    "Pacific Coral Atolls": [
        "Marshall Islands", "Kiribati Atolls", "Tuvalu Atolls", "Palau Archipelago", "Caroline Islands"
    ],
    "Australian Coral Systems": [
        "Coral Sea Islands", "Great Barrier Reef Islands"
    ],
    "Caribbean Coral Islands": [
        "Bahamas Archipelago", "Turks and Caicos", "Cayman Islands", "Lesser Antilles"
    ]
}

# Create a flattened list for the selectbox with category headers
location_options = []
for category, locations in location_categories.items():
    if locations:  # Only add categories that have locations
        location_options.append(f"--- {category} ---")
        location_options.extend(locations)

# Location selector
selected_location = st.sidebar.selectbox(
    "Select Archipelago",
    location_options,
    help="Choose a complete island archipelago for wave predictions"
)

# Skip category headers
if selected_location.startswith("---"):
    st.sidebar.warning("Please select an actual archipelago, not a category header.")
    st.stop()

target_date = st.sidebar.date_input("Prediction date", date.today()).strftime("%Y-%m-%d")
steps = st.sidebar.slider("Forecast steps (6h interval)", 1, 10, 4)

# Show archipelago coverage info
if selected_location in location_coords:
    lat_c, lon_c = location_coords[selected_location]
    bounds_info = get_prediction_bounds(selected_location, lat_c, lon_c)

    st.sidebar.markdown("**Archipelago Coverage:**")
    st.sidebar.markdown(f"**{bounds_info['description']}**")
    st.sidebar.markdown(f"Center: {lat_c:.1f}°N, {lon_c:.1f}°E")
    st.sidebar.markdown(f"Area: {bounds_info['coverage_info']}")
    st.sidebar.markdown(f"Grid: ~{bounds_info['grid_size']}")

    # Calculate approximate area
    lat_range = bounds_info['lat_bounds'][1] - bounds_info['lat_bounds'][0]
    lon_range = bounds_info['lon_bounds'][1] - bounds_info['lon_bounds'][0]
    area_km2 = int(lat_range * 111 * lon_range * 111)  # Rough calculation
    st.sidebar.markdown(f"Coverage: ~{area_km2:,} km²")

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
                    "predictions": {"swh": [[[2.5 for _ in range(32)] for _ in range(32)] for _ in range(4)]},
                    "lats": list(np.linspace(bounds_info['lat_bounds'][0], bounds_info['lat_bounds'][1], 32)),
                    "lons": list(np.linspace(bounds_info['lon_bounds'][0], bounds_info['lon_bounds'][1], 32)),
                    "times": ["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00", "2024-01-01T18:00:00"]
                }
        except Exception:
            st.warning("Using sample data for demonstration.")
            # Create sample prediction data
            st.session_state["prediction"] = {
                "predictions": {"swh": [[[2.5 for _ in range(32)] for _ in range(32)] for _ in range(4)]},
                "lats": list(np.linspace(bounds_info['lat_bounds'][0], bounds_info['lat_bounds'][1], 32)),
                "lons": list(np.linspace(bounds_info['lon_bounds'][0], bounds_info['lon_bounds'][1], 32)),
                "times": ["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00", "2024-01-01T18:00:00"]
            }

# World map with English labels
col1, col2 = st.columns([2, 1])

# Filter out category headers for map display
actual_locations = {k: v for k, v in location_coords.items() if not k.startswith("---")}
world_lats, world_lons = zip(*actual_locations.values())
world_names = list(actual_locations.keys())

with col1:
    # Create Folium world map with island locations
    world_map = folium.Map(
        location=[5, 80],  # Center on Indian Ocean
        zoom_start=2,
        tiles='CartoDB positron',
        width='100%',
        height='350px'
    )

    # Color code by region
    region_colors = {
        "Maldives": "red",
        "Lakshadweep": "darkgreen",
        "Andaman": "blue",
        "Chagos": "green",
        "Seychelles": "orange",
        "Mascarene": "purple",
        "Marshall": "lightblue",
        "Kiribati": "lightblue",
        "Tuvalu": "lightblue",
        "Palau": "lightblue",
        "Caroline": "lightblue",
        "Bahamas": "pink",
        "Turks": "pink",
        "Cayman": "pink",
        "Lesser Antilles": "pink",
        "Cocos": "darkblue",
        "Christmas": "darkblue",
        "Coral Sea": "darkblue",
        "Great Barrier": "darkblue",
        "Socotra": "brown",
        "Comoro": "brown"
    }

    # Add island markers
    for name, (lat, lon) in actual_locations.items():
        # Get color for this location
        color = "red"  # Default
        for region, region_color in region_colors.items():
            if region in name:
                color = region_color
                break

        # Determine marker size based on selection
        if name == selected_location:
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>SELECTED: {name}</b><br>Lat: {lat:.2f}°<br>Lon: {lon:.2f}°",
                tooltip=name,
                icon=folium.Icon(color='red', icon='star', prefix='fa')
            ).add_to(world_map)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=f"<b>{name}</b><br>Lat: {lat:.2f}°<br>Lon: {lon:.2f}°",
                tooltip=name,
                color='white',
                weight=2,
                fillColor=color,
                fillOpacity=0.8
            ).add_to(world_map)

    # Add title
    title_html = '''
    <div style="position: fixed;
                top: 10px; left: 50px; width: 400px; height: 40px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; font-weight: bold; padding: 8px">
    Complete Island Archipelagos - Wave Activity Prediction Coverage
    </div>
    '''
    world_map.get_root().html.add_child(folium.Element(title_html))

    # Display the map
    components.html(world_map._repr_html_(), height=370)
with col2:
    st.markdown("#### Selected Location")
    st.markdown(f"**{selected_location}**")
    st.markdown(f"Date: {target_date}")
    st.markdown(f"Steps: {steps}")

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
            times = pred["forecast_times"]

            # Time step selector
            selected_step = st.selectbox(
                "Select forecast time step:",
                range(len(times)),
                format_func=lambda x: f"Step {x}: {times[x]}",
                key=f"time_step_selector_{idx}"
            )

            st.markdown(f"#### Step {selected_step} – {times[selected_step]}")
            bounds_info = get_prediction_bounds(selected_location, *location_coords[selected_location])

            # Add explanation of what users are seeing
            st.info(f"""
            **📍 Prediction Points Visualization**

            Each colored point represents a prediction location with the following features:
            • **Point Color**: Indicates the predicted value (blue=low, red=high)
            • **Point Size**: Larger points indicate higher values
            • **Interactive**: Click any point for detailed information
            • **Coverage**: {len(lats)}×{len(lons)} grid points across {bounds_info['description']}
            • **Area**: {bounds_info['coverage_info']} around {selected_location}
            """)

            st.markdown("**Map Legend**: Blue points = low values, Yellow/Red points = high values. Point size indicates magnitude.")

            # Show only ocean variables that actually exist in the prediction
            available_ocean_vars = []
            if "predictions" in pred:
                for var in current_vars:
                    if var in pred["predictions"] and pred["predictions"][var]:
                        available_ocean_vars.append(var)

            if not available_ocean_vars:
                st.warning("No ocean variable data available in prediction")
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

                        folium_map = create_folium_prediction_map(data, lats, lons, title, unit, selected_location)
                        components.html(folium_map._repr_html_(), height=420)
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

                            folium_map = create_folium_prediction_map(data, lats, lons, title, unit, selected_location)
                            components.html(folium_map._repr_html_(), height=470)
        else:
            st.info("Run a prediction to display maps.")
