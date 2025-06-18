import os
import time
import datetime
import pickle
import numpy as np
import torch
import xarray as xr
import fsspec
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from aurora import Aurorawave, rollout, Batch, Metadata
from huggingface_hub import hf_hub_download

# --- New Google Cloud Imports ---
from google.cloud import bigquery
import pandas as pd # For processing BigQuery results

# --- Configuration ---
DOWNLOAD_PATH = Path("./aurora_downloads").expanduser()
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

# ECMWF_WAVE_VARIABLES will be used for mapping/querying.
# Note: For BigQuery, you'll need to map these to actual column names
# or parameter IDs in the BQ tables. The 6-digit number is the ECMWF parameter ID.
ECMWF_WAVE_VARIABLES: dict[str, str] = {
    "swh": "140229", # Significant height of total swell and wind waves
    "pp1d": "140231", # Primary wave mean period
    "mwp": "140232", # Mean wave period
    "mwd": "140230", # Mean wave direction
    "shww": "140234", # Significant height of wind waves
    "mdww": "140235", # Mean direction of wind waves
    "mpww": "140236", # Mean period of wind waves
    "shts": "140237", # Significant height of total swell
    "mdts": "140238", # Mean direction of total swell
    "mpts": "140239", # Mean period of total swell
    "swh1": "140121", # Significant height of first swell partition
    "mwd1": "140122", # Mean wave direction of first swell partition
    "mwp1": "140123", # Mean wave period of first swell partition
    "swh2": "140124", # Significant height of second swell partition
    "mwd2": "140125", # Mean wave direction of second swell partition
    "mwp2": "140126", # Mean wave period of second swell partition
    "dwi": "140249", # Wind wave direction
    "wind": "140245", # Wind speed at 10m (might also be available in WeatherBench2 as 10m_u/v_component_of_wind)
}

WEATHERBENCH2_SURFACE_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]
WEATHERBENCH2_ATMOS_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]
WEATHERBENCH2_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"

STATIC_REPO_ID = "microsoft/aurora"
STATIC_FILENAME = "aurora-0.25-wave-static.pickle"

app = Flask(__name__)
CORS(app)

# --- AuroraModelManager class remains the same ---
class AuroraModelManager:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuroraModelManager, cls).__new__(cls)
            print("Loading AuroraWave model...")
            cls._model = Aurorawave()
            cls._model.load_checkpoint()
            cls._model.eval()
            # Try to move to GPU if available
            if torch.cuda.is_available():
                cls._model.to("cuda")
                print("Model loaded to GPU.")
            else:
                cls._model.to("cpu")
                print("Model loaded to CPU.")
        return cls._instance

    def get_model(self):
        return self._model

# --- Data Preparation Helpers ---
def _prepare_hres(x: np.ndarray) -> torch.Tensor:
    # Ensure input is (time, lat, lon) or (time, level, lat, lon)
    # This prepares it for Aurora, typically taking the initial state.
    # The [None, -1, :] from original implies taking last time step for 2D.
    # Let's adjust to ensure it takes the first (initial) time step for the 2D surface variables.
    if x.ndim == 3: # (time, lat, lon)
        return torch.from_numpy(x[0:1].copy()) # Take first time step, keep time dim (1, lat, lon)
    elif x.ndim == 4: # (time, level, lat, lon)
        return torch.from_numpy(x[0:1].copy()) # Take first time step, keep time dim (1, level, lat, lon)
    else:
        raise ValueError(f"Unexpected array dimensions: {x.ndim}")

def _prepare_wave(x: np.ndarray) -> torch.Tensor:
    # Wave data are usually 2D (time, lat, lon)
    # Prepare for Aurora which might expect (1, lat, lon)
    return torch.from_numpy(x[0:1].copy())


def download_and_prepare_data(target_date_str: str, lat_bounds: tuple, lon_bounds: tuple):
    day = target_date_str # e.g., "20220916"
    target_date_dt = datetime.datetime.strptime(day, "%Y%m%d").date() # For BigQuery date filtering

    # --- 1. Download HRES-WAM data (Wave variables) from BigQuery ---
    wave_nc_file = DOWNLOAD_PATH / f"{day}-wave-gcp.nc" # Save as NetCDF from BigQuery

    if not wave_nc_file.exists():
        print(f"Querying BigQuery for HRES-WAM data for {day}...")
        try:
            client = bigquery.Client()
            
            # --- IMPORTANT: BigQuery Query and Data Reshaping ---
            # You MUST adjust this query based on the exact schema of the ECMWF tables
            # in bigquery-public-data.open_data_ecmwf.
            #
            # The 'grib' table stores raw GRIB messages as bytes, which is complex to use directly.
            # The 'forecast' or 'reanalysis' tables typically have parsed data.
            #
            # This is a *placeholder query*. You need to inspect the schema of
            # `bigquery-public-data.open_data_ecmwf.forecast` or `reanalysis` to get
            # the exact column names for variables like SWH, MWP, MWD, etc.
            # Also confirm the time field name (e.g., 'time', 'valid_time', 'forecast_time').
            #
            # Assuming 'forecast' table for simplicity, but 'reanalysis' might be better for 'an' type.
            # You might need to query for `paramId` and then pivot.
            
            # Example query for common parameters - YOU WILL LIKELY NEED TO ADJUST THIS
            # ECMWF_WAVE_VARIABLES_BQ_MAP will map your desired variable names
            # to their actual column names in BigQuery.
            # You'd typically find these as `paramid_140229`, or `swh`, `mean_wave_period` etc.
            
            # This is a generic way if parameters are in a 'parameterId' column
            # and values in a 'value' column. This might not be exact for ECMWF public data.
            # A more common structure for BigQuery public weather data might be columns like:
            #   (valid_time, latitude, longitude, swh, mwd, pp1d, ...)
            
            # For demonstration, let's assume a structure like:
            # valid_time, latitude, longitude, parameter_id, value
            # and we need to pivot it.
            
            # Recommended approach: find a table in bigquery-public-data.open_data_ecmwf
            # that directly has swh, mwd etc. as columns, or a parsed table.
            
            # For now, let's make a generic query and emphasize user's adjustment.
            # We are trying to get the ANALYSIS ('an') data at 00, 06, 12, 18 UTC for the day.
            
            # Example: Using the `forecast` table, which might include `type='an'`
            # and `step=0` for analysis.
            # YOU MUST CHECK THE ACTUAL COLUMN NAMES AND DATA AVAILABILITY IN BQ.
            
            target_dates_str = [
                f"'{target_date_dt.isoformat()}T00:00:00Z'",
                f"'{target_date_dt.isoformat()}T06:00:00Z'",
                f"'{target_date_dt.isoformat()}T12:00:00Z'",
                f"'{target_date_dt.isoformat()}T18:00:00Z'"
            ]
            
            # Convert ECMWF_WAVE_VARIABLES to parameter IDs for BQ query
            param_ids_to_fetch = list(ECMWF_WAVE_VARIABLES.values())
            param_ids_str = ', '.join([f"'{pid}'" for pid in param_ids_to_fetch])

            # This query attempts to fetch data for specific parameter IDs
            # You MUST verify if 'parameter_id' and 'value' columns exist and if this
            # is how HRES-WAM is stored for analysis (`type='an'`, `step=0`).
            query = f"""
                SELECT
                    valid_time,
                    latitude,
                    longitude,
                    parameter_id,
                    value
                FROM
                    `bigquery-public-data.open_data_ecmwf.forecast` -- or `reanalysis`, or a more specific table
                WHERE
                    valid_time IN ({','.join(target_dates_str)})
                    AND parameter_id IN ({param_ids_str})
                    AND type = 'an' -- 'an' for analysis data
                    AND step = 0    -- 0-hour forecast means it's an analysis
                ORDER BY
                    valid_time, latitude DESC, longitude ASC
            """
            
            job = client.query(query)
            df = job.to_dataframe()

            if df.empty:
                raise ValueError(f"No wave data found in BigQuery for date {day} with specified parameters and analysis type. "
                                 f"Please check the query, table name, parameter IDs, and 'type'/'step' filters.")

            # --- Reshape DataFrame to xarray.Dataset ---
            # Pivot the DataFrame from long format (parameter_id, value) to wide format (columns for each param_id)
            # Then convert to xarray.Dataset
            
            # Map parameter IDs back to common names for xarray
            param_id_to_name = {v: k for k, v in ECMWF_WAVE_VARIABLES.items()}
            df['parameter_name'] = df['parameter_id'].astype(str).map(param_id_to_name)
            
            if df['parameter_name'].isnull().any():
                # This means some parameter_ids from BQ were not in our ECMWF_WAVE_VARIABLES map
                # You might need to adjust ECMWF_WAVE_VARIABLES or the BQ query.
                print("Warning: Some parameter_ids from BigQuery were not mapped to Aurora variable names.")
            
            # Pivot to create columns for each variable
            df_pivot = df.pivot_table(
                index=['valid_time', 'latitude', 'longitude'],
                columns='parameter_name',
                values='value'
            ).reset_index()

            # Ensure 'valid_time' is datetime type
            df_pivot['valid_time'] = pd.to_datetime(df_pivot['valid_time'])

            # Convert to xarray Dataset. Ensure correct dimensions and coordinate order.
            # xarray typically expects lat to be descending (N to S) for slice(max, min)
            # and lon ascending (W to E).
            wave_vars_ds = df_pivot.set_index(['valid_time', 'latitude', 'longitude']).to_xarray()
            
            # Ensure latitudes are sorted correctly (N to S for `sel` later)
            if wave_vars_ds.latitude.values[0] < wave_vars_ds.latitude.values[-1]:
                wave_vars_ds = wave_vars_ds.sel(latitude=slice(None, None, -1))

            # Save the dataset to a local NetCDF file for caching
            wave_vars_ds.to_netcdf(str(wave_nc_file))
            print("HRES-WAM data downloaded from BigQuery and saved locally!")

        except Exception as e:
            # Catch a broader exception if BigQuery setup or query is complex
            raise RuntimeError(f"Failed to query and process HRES-WAM data from BigQuery: {e}. "
                               f"Please ensure you have GCP authentication, correct BigQuery project, "
                               f"and the SQL query matches the dataset schema for wave parameters.")
    else:
        print(f"HRES-WAM data for {day} already exists locally from GCP.")
        wave_vars_ds = xr.open_dataset(str(wave_nc_file), engine="netcdf4")


    # --- 2. Download Meteorological variables from WeatherBench2 (Already on GCP) ---
    surface_nc_file = DOWNLOAD_PATH / f"{day}-surface-level.nc"
    atmos_nc_file = DOWNLOAD_PATH / f"{day}-atmospheric.nc"
    
    if not hasattr(download_and_prepare_data, '_ds_global'):
        print("Opening WeatherBench2 Zarr store on GCS...")
        download_and_prepare_data._ds_global = xr.open_zarr(fsspec.get_mapper(WEATHERBENCH2_URL), chunks=None)
        print("WeatherBench2 Zarr store opened.")
    ds_global = download_and_prepare_data._ds_global

    if not surface_nc_file.exists():
        print(f"Downloading surface-level variables for {day} from WeatherBench2 (GCS)...")
        # Ensure 'time' dimension selection matches 'day' for WeatherBench2 data
        # WeatherBench2's time dimension might be `datetime64[ns]`. Convert target_date_dt.
        ds_surf = ds_global[WEATHERBENCH2_SURFACE_VARS].sel(time=pd.to_datetime(target_date_dt)).compute()
        ds_surf.to_netcdf(str(surface_nc_file))
        print("Surface-level variables downloaded from WeatherBench2!")
    else:
        print(f"Surface-level variables for {day} already exists locally from WeatherBench2.")

    if not atmos_nc_file.exists():
        print(f"Downloading atmospheric variables for {day} from WeatherBench2 (GCS)...")
        ds_atmos = ds_global[WEATHERBENCH2_ATMOS_VARS].sel(time=pd.to_datetime(target_date_dt)).compute()
        ds_atmos.to_netcdf(str(atmos_nc_file))
        print("Atmospheric variables downloaded from WeatherBench2!")
    else:
        print(f"Atmospheric variables for {day} already exists locally from WeatherBench2.")

    # --- 3. Download Static Variables (from Hugging Face) ---
    static_path = hf_hub_download(
        repo_id=STATIC_REPO_ID,
        filename=STATIC_FILENAME,
    )
    print("Static variables downloaded!")

    # --- 4. Prepare Batch (including cropping to region) ---
    with open(static_path, "rb") as f:
        static_vars = pickle.load(f)

    surf_vars_ds = xr.open_dataset(
        DOWNLOAD_PATH / f"{day}-surface-level.nc",
        engine="netcdf4",
        decode_timedelta=True,
    )

    wave_vars_ds = xr.open_dataset(
        str(wave_nc_file), # Use the path to the NetCDF generated from BigQuery
        engine="netcdf4",
        decode_timedelta=True
    )

    atmos_vars_ds = xr.open_dataset(
        DOWNLOAD_PATH / f"{day}-atmospheric.nc",
        engine="netcdf4",
        decode_timedelta=True,
    )

    lat_min, lat_max = min(lat_bounds), max(lat_bounds)
    lon_min, lon_max = min(lon_bounds), max(lon_bounds)

    # Ensure coordinates are float for slicing
    lat_min, lat_max, lon_min, lon_max = float(lat_min), float(lat_max), float(lon_min), float(lon_max)

    # sel(latitude=slice(max, min)) is typical for xarray when lat coordinates run from N to S
    # Ensure your BigQuery data output and xarray reshaping results in latitude in this order.
    # Also adjust longitude to handle 0-360 vs -180-180 if necessary, Aurora expects 0-360.
    
    # WeatherBench2 has longitudes from 0 to 359.75. Your input might be -180 to 180.
    # Convert input longitudes to 0-360 range if they are negative.
    if lon_min < 0:
        lon_min += 360
    if lon_max < 0:
        lon_max += 360

    # Ensure lon_min is less than lon_max for slicing across the 0/360 meridian if needed
    if lon_min > lon_max: # Example: from 350 to 10 (crosses 0/360)
        ds_part1 = ds_global.sel(longitude=slice(lon_min, 360))
        ds_part2 = ds_global.sel(longitude=slice(0, lon_max))
        # This merging is complex and might not work directly, simpler to get broader and crop
        # For initial setup, assume non-meridian crossing bounds or let xarray handle it.
        # For now, let's keep slicing simple and assume the bounds don't cross the meridian.
        # If they do, the user will need to implement more complex slicing/merging.
        # Aurora expects 0-360 longitudes so direct slicing might be tricky if input is -180 to 180.
        pass # Handle this edge case later if it occurs

    surf_vars_ds_cropped = surf_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    wave_vars_ds_cropped = wave_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    atmos_vars_ds_cropped = atmos_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    
    if surf_vars_ds_cropped.sizes['latitude'] == 0 or surf_vars_ds_cropped.sizes['longitude'] == 0:
        raise ValueError(f"No data found for the specified region Lat: {lat_bounds}, Lon: {lon_bounds}. "
                         f"Please check the coordinates. Global data is at 0.25x0.25 resolution. "
                         f"Cropped latitude range: {surf_vars_ds_cropped.latitude.values.min():.2f}-{surf_vars_ds_cropped.latitude.values.max():.2f}, "
                         f"longitude range: {surf_vars_ds_cropped.longitude.values.min():.2f}-{surf_vars_ds_cropped.longitude.values.max():.2f}")

    # Prepare surf_vars_input with both meteorological and wave variables
    surf_vars_input = {
        "2t": _prepare_hres(surf_vars_ds_cropped["2m_temperature"].values),
        "10u": _prepare_hres(surf_vars_ds_cropped["10m_u_component_of_wind"].values),
        "10v": _prepare_hres(surf_vars_ds_cropped["10m_v_component_of_wind"].values),
        "msl": _prepare_hres(surf_vars_ds_cropped["mean_sea_level_pressure"].values),
    }
    
    # Map from the xarray dataset (wave_vars_ds_cropped) to Aurora's expected keys
    for aurora_var_name, ecmwf_param_id in ECMWF_WAVE_VARIABLES.items():
        # Check if the variable exists in the reshaped dataset from BigQuery
        # BigQuery column names might be the `aurora_var_name` (e.g. 'swh')
        # or the `ecmwf_param_id` (e.g. '140229') or something else.
        # Adjust this check based on how your BQ data is structured.
        if aurora_var_name in wave_vars_ds_cropped:
            surf_vars_input[aurora_var_name] = _prepare_wave(wave_vars_ds_cropped[aurora_var_name].values)
        else:
            # Fallback if variable is missing, but crucial to get this right from BQ.
            print(f"Warning: Wave variable '{aurora_var_name}' (ECMWF ID: {ecmwf_param_id}) not found in BigQuery derived data for {day}. Skipping.")

    # Batch creation remains the same
    batch = Batch(
        surf_vars=surf_vars_input,
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars={
            "t": _prepare_hres(atmos_vars_ds_cropped["temperature"].values),
            "u": _prepare_hres(atmos_vars_ds_cropped["u_component_of_wind"].values),
            "v": _prepare_hres(atmos_vars_ds_cropped["v_component_of_wind"].values),
            "q": _prepare_hres(atmos_vars_ds_cropped["specific_humidity"].values),
            "z": _prepare_hres(atmos_vars_ds_cropped["geopotential"].values),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds_cropped.latitude.values[::-1].copy()), # Ensure N-S order for Aurora
            lon=torch.from_numpy(surf_vars_ds_cropped.longitude.values),
            time=(surf_vars_ds_cropped.time.values.astype("datetime64[s]").tolist()[0],), # Initial time of the prediction
            atmos_levels=tuple(int(level) for level in atmos_vars_ds_cropped.level.values),
        ),
    )
    return batch, surf_vars_ds_cropped # Also return original cropped data for ref

# --- Flask API Endpoint ---
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    target_date_str = data.get("target_date")
    lat_bounds = tuple(data.get("lat_bounds"))
    lon_bounds = tuple(data.get("lon_bounds"))
    
    if not all([target_date_str, lat_bounds, lon_bounds]):
        return jsonify({"error": "Missing parameters (target_date, lat_bounds, lon_bounds)"}), 400

    try:
        model_manager = AuroraModelManager()
        model = model_manager.get_model()

        start_time = time.time()
        print(f"Downloading and preparing data for {target_date_str}...")
        batch, initial_ds_cropped = download_and_prepare_data(target_date_str, lat_bounds, lon_bounds)
        data_prep_time = time.time() - start_time
        print(f"Data preparation completed in {data_prep_time:.2f} seconds.")

        start_time = time.time()
        print("Running Aurora model rollout...")
        with torch.inference_mode():
            # The steps parameter determines how many forecast steps (e.g., 6-hour intervals)
            # Aurora makes. Adjust as needed. Default for wave model is often 2 (for 12 and 18hr).
            preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]
        model_rollout_time = time.time() - start_time
        print(f"Model rollout completed in {model_rollout_time:.2f} seconds.")

        # Prepare data for JSON response
        predictions_data = {}
        forecast_times = []

        # Iterate through each forecast step (preds[0] is initial, preds[1] is 6h, preds[2] is 12h, etc.)
        # Aurora returns initial state as preds[0], then preds[1], preds[2] etc. are forecasts.
        # We need to send the forecast steps, typically starting from preds[1].
        
        # Decide which forecast steps to return. Let's return all.
        for i, pred_batch in enumerate(preds):
            current_time = pred_batch.metadata.time[0] # Get timestamp for this step
            forecast_times.append(current_time)

            # Store surface variables
            for var_name, tensor in pred_batch.surf_vars.items():
                if var_name not in predictions_data:
                    predictions_data[var_name] = []
                # Remove batch and time dimensions (if only one initial time was fed in)
                # and convert to numpy array for JSON serialization
                predictions_data[var_name].append(tensor.squeeze().cpu().numpy().tolist())
            
            # Store atmospheric variables (example for first level for simplicity, can expand)
            # Note: Atmospheric vars have a 'level' dimension. You might want to select a specific level.
            # For now, we'll just expose a few top-level atmospheric variables at the first available level.
            for var_name, tensor in pred_batch.atmos_vars.items():
                if var_name not in predictions_data:
                    predictions_data[var_name] = []
                # Squeeze to remove batch and time dims, then select first level (index 0)
                # Ensure it's 2D (lat, lon) for map plotting
                predictions_data[var_name].append(tensor.squeeze()[0].cpu().numpy().tolist()) # Taking first level

        # Include metadata for plotting on frontend
        lats = initial_ds_cropped.latitude.values.tolist()
        lons = initial_ds_cropped.longitude.values.tolist()

        return jsonify({
            "status": "success",
            "message": "Prediction generated successfully.",
            "lats": lats,
            "lons": lons,
            "forecast_times": [t.isoformat() for t in forecast_times], # Convert datetime to ISO string
            "predictions": predictions_data, # Dictionary of variable_name: [list of numpy arrays per step]
            "data_prep_time": f"{data_prep_time:.2f}s",
            "model_rollout_time": f"{model_rollout_time:.2f}s"
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500

if __name__ == "__main__":
    print("Starting Flask AuroraWave Backend...")
    app.run(debug=True, host="0.0.0.0", port=5000)
