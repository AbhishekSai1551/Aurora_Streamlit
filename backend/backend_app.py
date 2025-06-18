import os
import time
import datetime
import pickle
import numpy as np
import torch
import xarray as xr
import fsspec
# Removed google.cloud.bigquery and pandas as they are no longer needed for wave data from GCS
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from aurora import Aurorawave, rollout, Batch, Metadata
from huggingface_hub import hf_hub_download

# --- Configuration ---
DOWNLOAD_PATH = Path("./aurora_downloads").expanduser()
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

# ECMWF API keys are NOT needed for public gs://ecmwf-open-data/ bucket
# The check below is now removed as we don't use ecmwfapi directly
# if not ECMWF_API_KEY or not ECMWF_API_EMAIL:
#     raise RuntimeError(
#         "ECMWF_API_KEY or ECMWF_API_EMAIL environment variables are not set. "
#         "These were previously required for downloading HRES-WAM data via ECMWF API. "
#         "Now using Google Cloud Storage public data."
#     )

# ECMWF_WAVE_VARIABLES are still used for parameter IDs and checking their presence
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

# The specific parameter IDs for ECMWF_OPEN_DATA_WAVE_PARAMETERS are numerical, e.g., '140229'
# We will use these directly in the GRIB request.
ECMWF_OPEN_DATA_WAVE_PARAMETERS = [
    "140229", # swh
    "140231", # pp1d
    "140232", # mwp
    "140230", # mwd
    "140245", # wind (10m_wind speed/gust) - check if this is the correct param for 'wind'
              # Note: 'wind' might be derived from '10u' and '10v' in some models.
              # If your Aurora model expects a specific 'wind' variable, ensure this ID is correct.
              # The PDF example shows 'wind' being mapped to '140245'.
]

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

# For ECMWF Open Data in GCS, you can find the actual stream/type for wave data.
# Based on the naming convention, 'wave' stream and 'fc' (forecast) type for 0-step analysis data.
ECMWF_OPEN_DATA_GCS_BASE = "gs://ecmwf-open-data"
ECMWF_OPEN_DATA_WAVE_STREAM = "wave" # Use 'wave' for ocean wave fields
ECMWF_OPEN_DATA_WAVE_TYPE = "fc"     # Use 'fc' for forecast, even for analysis (step=0)
ECMWF_OPEN_DATA_RESOLUTION = "0p25" # Assuming 0.25 degree resolution for wave data

STATIC_REPO_ID = "microsoft/aurora"
STATIC_FILENAME = "aurora-0.25-wave-static.pickle"

app = Flask(__name__)
# For local testing, CORS(app) is fine. For deployment, restrict to your frontend URL.
CORS(app) 

# --- AuroraWave Model Loader (Singleton pattern) ---
class AuroraModelManager:
    _instance = None
    _model = None
    _device = "cpu"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuroraModelManager, cls).__new__(cls)
            try:
                print("Initializing AuroraWave model...")
                cls._instance._model = Aurorawave()
                cls._instance._model.load_checkpoint()
                cls._instance._model.eval()

                if torch.cuda.is_available():
                    cls._instance._device = "cuda"
                    cls._instance._model = cls._instance._model.to(cls._instance._device)
                    print(f"AuroraWave model loaded successfully and moved to {cls._instance._device}.")
                else:
                    cls._instance._device = "cpu"
                    cls._instance._model = cls._instance._model.to(cls._instance._device)
                    print(f"AuroraWave model loaded successfully on CPU (CUDA not available).")
            except Exception as e:
                print(f"ERROR: Failed to load AuroraWave model: {e}")
                cls._instance._model = None
                raise RuntimeError(f"Failed to load AuroraWave model: {e}") # Raise to stop app if model fails
        return cls._instance

    def get_model(self):
        if self._model is None:
            raise RuntimeError("AuroraWave model failed to load. Check previous logs.")
        return self._model, self._device

# --- Data Preparation Helpers ---
def _prepare_hres(x: np.ndarray) -> torch.Tensor:
    """Prepares HRES data (2m_temperature, 10m_wind, msl) for Aurora input."""
    # This prepares it for Aurora, typically taking the initial state.
    # The [None] adds a batch dimension. x[0] takes the first time step.
    if x.ndim == 3: # (time, lat, lon)
        return torch.from_numpy(x[0:1].copy()) # Take first time step, keep time dim (1, lat, lon)
    elif x.ndim == 4: # (time, level, lat, lon)
        return torch.from_numpy(x[0:1].copy()) # Take first time step, keep time dim (1, level, lat, lon)
    else:
        raise ValueError(f"Unexpected HRES array dimensions: {x.ndim}")

def _prepare_wave(x: np.ndarray) -> torch.Tensor:
    """Prepares wave data for Aurora input."""
    # Wave data are usually (time, lat, lon). Take first time step.
    if x.ndim == 3: # (time, lat, lon)
        return torch.from_numpy(x[0:1].copy())
    else:
        raise ValueError(f"Unexpected wave array dimensions: {x.ndim}")


def download_and_prepare_data(target_date_str: str, lat_bounds: tuple, lon_bounds: tuple):
    day = target_date_str # e.g., "20240101"
    
    # --- 1. Download HRES-WAM data (Wave variables) from ECMWF Open Data GCS ---
    # Need to download for each of the 4 analysis times (00Z, 06Z, 12Z, 18Z)
    analysis_times = ["00", "06", "12", "18"]
    all_wave_grib_files = [] # To store paths of downloaded GRIB files
    
    for hh in analysis_times:
        # Construct the GCS path based on the naming convention
        # gs://ecmwf-open-data/[yyyymmdd]/[HH]z/[resol]/[stream]/[yyyymmdd][HH]0000-[step][U]-[stream]-[type].[format]
        gcs_wave_path = (
            f"{ECMWF_OPEN_DATA_GCS_BASE}/{day}/{hh}z/{ECMWF_OPEN_DATA_RESOLUTION}/"
            f"{ECMWF_OPEN_DATA_WAVE_STREAM}/{day}{hh}0000-0h-{ECMWF_OPEN_DATA_WAVE_STREAM}-{ECMWF_OPEN_DATA_WAVE_TYPE}.grib2"
        )
        local_wave_grib_file = DOWNLOAD_PATH / f"{day}-wave-{hh}.grib2"
        all_wave_grib_files.append(local_wave_grib_file)

        if not local_wave_grib_file.exists():
            print(f"Downloading HRES-WAM data for {day} {hh}Z from GCS: {gcs_wave_path}...")
            try:
                with fsspec.open(gcs_wave_path, "rb") as f_remote:
                    with open(local_wave_grib_file, "wb") as f_local:
                        f_local.write(f_remote.read())
                print(f"HRES-WAM data for {day} {hh}Z downloaded!")
            except FileNotFoundError:
                raise RuntimeError(f"HRES-WAM GRIB file not found on GCS: {gcs_wave_path}. "
                                   f"Data might not be available for this date/time or path is incorrect.")
            except Exception as e:
                raise RuntimeError(f"Failed to download HRES-WAM data from GCS for {day} {hh}Z: {e}. "
                                   f"Check your internet connection or GCP authentication.")
        else:
            print(f"HRES-WAM data for {day} {hh}Z already exists locally.")

    # Open all downloaded GRIB files into a single xarray Dataset
    # cfgrib can open multiple GRIB files at once if they have compatible dimensions
    try:
        wave_vars_ds = xr.open_mfdataset(
            [str(f) for f in all_wave_grib_files],
            engine="cfgrib",
            concat_dim="time", # Assuming 'time' is the dimension to concatenate along
            combine='nested',
            coords="minimal",
            data_vars="minimal",
            compat="override",
            # This is important for ensuring cfgrib works with multiple files
            # For each file opened separately, `backend_kwargs={"indexpath": ""}` might be needed
            # but for open_mfdataset, it might handle indices across files.
            # If errors occur, try opening individual files and then merging.
        )
        # Ensure time dimension is sorted ascending if needed
        wave_vars_ds = wave_vars_ds.sortby('time')
        print("Combined HRES-WAM data loaded into xarray dataset.")

    except Exception as e:
        raise RuntimeError(f"Failed to load HRES-WAM GRIB files into xarray: {e}. "
                           f"Ensure cfgrib is installed and files are valid GRIB2.")


    # --- 2. Download Meteorological variables from WeatherBench2 (Already on GCP) ---
    surface_nc_file = DOWNLOAD_PATH / f"{day}-surface-level.nc"
    atmos_nc_file = DOWNLOAD_PATH / f"{day}-atmospheric.nc"
    
    # Load global Zarr store once and keep it in memory for efficiency
    if not hasattr(download_and_prepare_data, '_ds_global'):
        print("Opening WeatherBench2 Zarr store on GCS...")
        # Note: WeatherBench2 data range is typically 2016-2022.
        # If you need 2024-2025 data, you'll need a different source for HRES-T0.
        # For this refactor, we assume WeatherBench2 is still acceptable for atmos/surface.
        download_and_prepare_data._ds_global = xr.open_zarr(fsspec.get_mapper(WEATHERBENCH2_URL), chunks=None)
        print("WeatherBench2 Zarr store opened.")
    ds_global = download_and_prepare_data._ds_global

    # The target_date_dt from frontend will be a date object
    target_date_dt_obj = datetime.datetime.strptime(day, "%Y%m%d").date()

    if not surface_nc_file.exists():
        print(f"Downloading surface-level variables for {day} from WeatherBench2 (GCS)...")
        # Ensure 'time' dimension selection matches 'day' for WeatherBench2 data
        # WeatherBench2 has hourly data, select by date
        ds_surf = ds_global[WEATHERBENCH2_SURFACE_VARS].sel(time=str(target_date_dt_obj)).compute()
        ds_surf.to_netcdf(str(surface_nc_file))
        print("Surface-level variables downloaded from WeatherBench2!")
    else:
        print(f"Surface-level variables for {day} already exists locally from WeatherBench2.")

    if not atmos_nc_file.exists():
        print(f"Downloading atmospheric variables for {day} from WeatherBench2 (GCS)...")
        ds_atmos = ds_global[WEATHERBENCH2_ATMOS_VARS].sel(time=str(target_date_dt_obj)).compute()
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

    atmos_vars_ds = xr.open_dataset(
        DOWNLOAD_PATH / f"{day}-atmospheric.nc",
        engine="netcdf4",
        decode_timedelta=True,
    )

    lat_min, lat_max = min(lat_bounds), max(lat_bounds)
    lon_min, lon_max = min(lon_bounds), max(lon_bounds)

    # Ensure coordinates are float for slicing
    lat_min, lat_max, lon_min, lon_max = float(lat_min), float(lat_max), float(lon_min), float(lon_max)

    # WeatherBench2 and ECMWF Open Data longitudes are typically 0 to 359.75.
    # If frontend sends -180 to 180, convert to 0-360 range for slicing.
    # Aurora itself expects 0-360 for lon.
    lon_min_360 = lon_min % 360
    lon_max_360 = lon_max % 360

    # Handle wrapping around the 0/360 meridian if min > max after conversion
    if lon_min_360 > lon_max_360:
        # Slice in two parts and concatenate
        surf_vars_ds_part1 = surf_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_360, 360))
        surf_vars_ds_part2 = surf_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(0, lon_max_360))
        surf_vars_ds_cropped = xr.concat([surf_vars_ds_part1, surf_vars_ds_part2], dim='longitude').sortby('longitude')

        wave_vars_ds_part1 = wave_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_360, 360))
        wave_vars_ds_part2 = wave_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(0, lon_max_360))
        wave_vars_ds_cropped = xr.concat([wave_vars_ds_part1, wave_vars_ds_part2], dim='longitude').sortby('longitude')
        
        atmos_vars_ds_part1 = atmos_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_360, 360))
        atmos_vars_ds_part2 = atmos_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(0, lon_max_360))
        atmos_vars_ds_cropped = xr.concat([atmos_vars_ds_part1, atmos_vars_ds_part2], dim='longitude').sortby('longitude')

    else:
        surf_vars_ds_cropped = surf_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_360, lon_max_360))
        wave_vars_ds_cropped = wave_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_360, lon_max_360))
        atmos_vars_ds_cropped = atmos_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_360, lon_max_360))
    
    if surf_vars_ds_cropped.sizes['latitude'] == 0 or surf_vars_ds_cropped.sizes['longitude'] == 0:
        raise ValueError(f"No data found for the specified region Lat: {lat_bounds}, Lon: {lon_bounds}. "
                         f"Please check the coordinates. Global data is at {ECMWF_OPEN_DATA_RESOLUTION} resolution. "
                         f"Cropped latitude range: {surf_vars_ds_cropped.latitude.values.min():.2f}-{surf_vars_ds_cropped.latitude.values.max():.2f}, "
                         f"longitude range: {surf_vars_ds_cropped.longitude.values.min():.2f}-{surf_vars_ds_cropped.longitude.values.max():.2f}")

    # Prepare surf_vars_input with both meteorological and wave variables
    # The _prepare_hres and _prepare_wave functions take `(time, [level], lat, lon)` numpy arrays
    # and return `(1, [level], lat, lon)` torch tensors (taking first time step).
    surf_vars_input = {
        "2t": _prepare_hres(surf_vars_ds_cropped["2m_temperature"].values),
        "10u": _prepare_hres(surf_vars_ds_cropped["10m_u_component_of_wind"].values),
        "10v": _prepare_hres(surf_vars_ds_cropped["10m_v_component_of_wind"].values),
        "msl": _prepare_hres(surf_vars_ds_cropped["mean_sea_level_pressure"].values),
    }
    
    # Map from the xarray dataset (wave_vars_ds_cropped) to Aurora's expected keys
    # Use the keys from ECMWF_WAVE_VARIABLES and check if they exist in wave_vars_ds_cropped
    # The 'param' from cfgrib will often be the numeric ID (e.g., '140229') or a short name ('swh')
    # You might need to check both if cfgrib uses short names or numeric.
    for aurora_var_name in ["swh", "pp1d", "mwd", "mwp", "shww", "mdww", "mpww", "shts", "mdts", "mpts",
                            "swh1", "mwd1", "mwp1", "swh2", "mwd2", "mwp2", "dwi", "wind"]:
        if aurora_var_name in wave_vars_ds_cropped:
            surf_vars_input[aurora_var_name] = _prepare_wave(wave_vars_ds_cropped[aurora_var_name].values)
        elif str(ECMWF_WAVE_VARIABLES.get(aurora_var_name)) in wave_vars_ds_cropped: # Check by param ID if named as such
             surf_vars_input[aurora_var_name] = _prepare_wave(wave_vars_ds_cropped[str(ECMWF_WAVE_VARIABLES.get(aurora_var_name))].values)
        else:
            print(f"Warning: Wave variable '{aurora_var_name}' (or its ID) not found in downloaded wave data. Skipping.")

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
            # Latitude needs to be N-S for Aurora
            lat=torch.from_numpy(surf_vars_ds_cropped.latitude.values[::-1].copy()),
            lon=torch.from_numpy(surf_vars_ds_cropped.longitude.values),
            # The time stamp for Aurora's input batch corresponds to one of the initial analysis times.
            # Use the first time point (00Z) from the combined wave data for initial_ds_cropped for consistency
            time=(wave_vars_ds_cropped.time.values.astype("datetime64[s]").tolist()[0],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds_cropped.level.values),
        ),
    )
    return batch, wave_vars_ds_cropped # Return wave_vars_ds_cropped for its time dimension (all 4 steps)


# --- Flask App Endpoints ---
# Initialize AuroraModelManager once at app startup
try:
    aurora_model_manager = AuroraModelManager()
except RuntimeError as e:
    print(f"FATAL: {e}. Exiting Flask app setup.")
    exit(1)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    n_lat = float(data.get('lat_bounds')[1]) # North Lat (max)
    s_lat = float(data.get('lat_bounds')[0]) # South Lat (min)
    w_lon = float(data.get('lon_bounds')[0]) # West Lon (min)
    e_lon = float(data.get('lon_bounds')[1]) # East Lon (max)
    prediction_date_str = data.get('target_date')

    # Steps parameter is no longer explicitly sent by frontend but can be defaulted
    steps = request.args.get('steps', type=int, default=4) # Default to 4 for 00, 06, 12, 18Z + 4 forecast steps

    if not all([prediction_date_str is not None, n_lat is not None, w_lon is not None, s_lat is not None, e_lon is not None]):
        return jsonify({"error": "Missing one or more required parameters (target_date, lat_bounds, lon_bounds)."}), 400
    
    # Note: Aurora rollout steps typically mean number of 6-hour forecast steps.
    # The default 4 in frontend will produce initial + 4 steps (0, 6, 12, 18, 24h).
    # You might want to adjust `steps` parameter based on desired forecast horizon.
    # For now, keeping it flexible.

    try:
        model, device = aurora_model_manager.get_model()

        start_data_prep = time.time()
        batch, initial_wave_ds_cropped = download_and_prepare_data(
            prediction_date_str,
            (s_lat, n_lat), # download_and_prepare_data expects (min_lat, max_lat)
            (w_lon, e_lon)
        )
        data_prep_time = time.time() - start_data_prep
        print(f"Data preparation completed in {data_prep_time:.2f} seconds.")
        
        batch = batch.to(device)

        start_rollout = time.time()
        print(f"Running rollout for {steps} steps for region: N:{n_lat}, W:{w_lon}, S:{s_lat}, E:{e_lon} on date {prediction_date_str}")
        with torch.inference_mode():
            # Rollout produces `steps` number of forecast outputs (0-indexed).
            # If `steps=2`, it gives `preds[0]` (initial state) and `preds[1]` (6hr forecast).
            preds_batches = [pred.to("cpu") for pred in rollout(model, batch, steps=steps)]
        model_rollout_time = time.time() - start_rollout
        print(f"Model rollout completed in {model_rollout_time:.2f} seconds.")

        output_data = {}
        forecast_times = []

        # Extract latitude and longitude from the initial cropped dataset
        lats_output = initial_wave_ds_cropped.latitude.values.tolist()
        lons_output = initial_wave_ds_cropped.longitude.values.tolist()

        # The initial_wave_ds_cropped contains the 00, 06, 12, 18Z analysis data.
        # Aurora model's `rollout` starts from the input batch's time (e.g., 00Z or 06Z depending on your Batch Metadata time selection).
        # Its outputs `preds_batches[i]` will correspond to subsequent 6-hour steps.
        # The first element `preds_batches[0]` is often the initial state from the input batch.
        
        # Determine actual forecast times based on the initial input time and rollout steps
        initial_input_time_dt64 = batch.metadata.time[0] # Get initial input time (datetime64)
        initial_input_timestamp = pd.to_datetime(initial_input_time_dt64).timestamp() # Convert to UNIX timestamp

        for i, pred_batch in enumerate(preds_batches):
            # Calculate forecast time for each step. Aurora steps are usually 6-hourly.
            # If i=0, it's initial state. If i=1, it's initial + 6h.
            current_forecast_time_unix = initial_input_timestamp + (i * 6 * 3600)
            forecast_times.append(datetime.datetime.fromtimestamp(current_forecast_time_unix).isoformat())

            # Populate predictions_data for frontend
            for var_name, tensor_data in pred_batch.surf_vars.items():
                if var_name not in output_data:
                    output_data[var_name] = []
                # Squeeze to remove batch dimension, then convert to list
                output_data[var_name].append(tensor_data.squeeze().cpu().numpy().tolist())
            
            # Atmospheric variables (if you want to expose them, otherwise remove)
            for var_name, tensor_data in pred_batch.atmos_vars.items():
                # Ensure we handle levels if needed for display
                if var_name not in output_data:
                    output_data[var_name] = []
                # Example: taking the first level (index 0) if it's a 4D tensor (time, level, lat, lon)
                output_data[var_name].append(tensor_data.squeeze()[0].cpu().numpy().tolist())


        return jsonify({
            "status": "success",
            "message": "Prediction generated successfully.",
            "lats": lats_output,
            "lons": lons_output,
            "forecast_times": forecast_times, # ISO formatted strings
            "predictions": output_data, # Dictionary of variable_name: [list of 2D numpy arrays (converted to list) per step]
            "data_prep_time": f"{data_prep_time:.2f}s",
            "model_rollout_time": f"{model_rollout_time:.2f}s"
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        # Catch specific RuntimeErrors from download_and_prepare_data or model loading
        return jsonify({"error": f"Backend processing error: {re}"}), 500
    except Exception as e:
        # Catch any other unexpected errors
        app.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred: {e}. Please check backend logs."}), 500

if __name__ == '__main__':
    print("Starting Flask AuroraWave Backend...")
    app.run(debug=True, host="0.0.0.0", port=5000)
