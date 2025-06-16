import os
import time
import datetime
import pickle
import numpy as np
import torch
import xarray as xr
import fsspec
import ecmwfapi
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from aurora import Aurorawave, rollout, Batch, Metadata
from huggingface_hub import hf_hub_download

# --- Configuration ---
# Data will be downloaded to a local cache directory relative to the backend_app.py
# Make sure this path is writable by the user running the Flask app
DOWNLOAD_PATH = Path("./aurora_downloads").expanduser()
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

# Retrieve ECMWF credentials from environment variables for production deployment.
# For local testing, these variables must be set in your environment
# (e.g., export ECMWF_API_KEY="your_key" in your shell, or via a .env file and `python-dotenv`).
# For deployment, cloud platforms provide ways to set these securely.
ECMWF_API_KEY = os.environ.get('ECMWF_API_KEY')
ECMWF_API_EMAIL = os.environ.get('ECMWF_API_EMAIL')
ECMWF_API_URL = os.environ.get('ECMWF_API_URL', 'https://api.ecmwf.int/v1') # Default URL

# Ensure keys are present if we intend to use them explicitly
if not ECMWF_API_KEY or not ECMWF_API_EMAIL:
    # Changed to raise a RuntimeError to prevent startup if critical keys are missing
    # for data download, making it explicit.
    raise RuntimeError(
        "ECMWF_API_KEY or ECMWF_API_EMAIL environment variables are not set. "
        "These are required for downloading HRES-WAM data. "
        "Please set them in your environment or deployment platform."
    )

ECMWF_WAVE_VARIABLES: dict[str, str] = {
    "swh": "140229",
    "pp1d": "140231",
    "mwp": "140232",
    "mwd": "140230",
    "shww": "140234",
    "mdww": "140235",
    "mpww": "140236",
    "shts": "140237",
    "mdts": "140238",
    "mpts": "140239",
    "swh1": "140121",
    "mwd1": "140122",
    "mwp1": "140123",
    "swh2": "140124",
    "mwd2": "140125",
    "mwp2": "140126",
    "dwi": "140249",
    "wind": "140245",
}
for k, v in ECMWF_WAVE_VARIABLES.items():
    assert len(v) == 6
    ECMWF_WAVE_VARIABLES[k] = v[3:6] + "." + v[0:3]

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

# --- AuroraWave Model Loader (Singleton pattern) ---
class AuroraModelManager:
    _instance = None
    _model = None
    _device = "cpu"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuroraModelManager, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        try:
            print("Initializing AuroraWave model...")
            self._model = Aurorawave()
            self._model.load_checkpoint()
            self._model.eval()

            if torch.cuda.is_available():
                self._device = "cuda"
                self._model = self._model.to(self._device)
                print(f"AuroraWave model loaded successfully and moved to {self._device}.")
            else:
                self._device = "cpu"
                self._model = self._model.to(self._device)
                print(f"AuroraWave model loaded successfully on CPU (CUDA not available).")

        except Exception as e:
            print(f"ERROR: Failed to load AuroraWave model: {e}")
            self._model = None
            raise

    def get_model(self):
        if self._model is None:
            raise RuntimeError("AuroraWave model failed to load.")
        return self._model, self._device

# --- Data Preparation Helpers ---
def _prepare_hres(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x[:2][None, -1, :].copy())

def _prepare_wave(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x[:2][None])

def download_and_prepare_data(target_date_str: str, lat_bounds: tuple, lon_bounds: tuple):
    day = target_date_str

    wave_grib_file = DOWNLOAD_PATH / f"{day}-wave.grib"
    if not wave_grib_file.exists():
        print(f"Downloading HRES-WAM data for {day}...")
        try:
            # CORRECTED: Using the variables populated from environment variables
            c = ecmwfapi.ECMWFService("mars", url=ECMWF_API_URL, key=ECMWF_API_KEY, email=ECMWF_API_EMAIL)
            c.execute(
                f"""
            request,
                class=od,
                date={day}/to/{day},
                domain=g,
                expver=1,
                param={'/'.join(ECMWF_WAVE_VARIABLES.values())},
                stream=wave,
                time=00:00:00/06:00:00/12:00:00/18:00:00,
                grid=0.25/0.25,
                type=an,
                target="{day}-wave.grib"
            """,
                str(wave_grib_file),
            )
            print("HRES-WAM data downloaded!")
        except Exception as e:
            raise RuntimeError(f"Failed to download HRES-WAM data from ECMWF: {e}. "
                               f"Ensure ECMWF_API_KEY and ECMWF_API_EMAIL environment variables are set correctly, "
                               f"and you have accepted the necessary data licenses on the ECMWF website.")
    else:
        print(f"HRES-WAM data for {day} already exists.")

    # --- 2. Download Meteorological variables from WeatherBench2 ---
    surface_nc_file = DOWNLOAD_PATH / f"{day}-surface-level.nc"
    atmos_nc_file = DOWNLOAD_PATH / f"{day}-atmospheric.nc"
    
    # Load global Zarr store once and keep it in memory for efficiency
    if not hasattr(download_and_prepare_data, '_ds_global'):
        print("Opening WeatherBench2 Zarr store...")
        download_and_prepare_data._ds_global = xr.open_zarr(fsspec.get_mapper(WEATHERBENCH2_URL), chunks=None)
        print("WeatherBench2 Zarr store opened.")
    ds_global = download_and_prepare_data._ds_global

    if not surface_nc_file.exists():
        print(f"Downloading surface-level variables for {day}...")
        ds_surf = ds_global[WEATHERBENCH2_SURFACE_VARS].sel(time=day).compute()
        ds_surf.to_netcdf(str(surface_nc_file))
        print("Surface-level variables downloaded!")
    else:
        print(f"Surface-level variables for {day} already exists.")

    if not atmos_nc_file.exists():
        print(f"Downloading atmospheric variables for {day}...")
        ds_atmos = ds_global[WEATHERBENCH2_ATMOS_VARS].sel(time=day).compute()
        ds_atmos.to_netcdf(str(atmos_nc_file))
        print("Atmospheric variables downloaded!")
    else:
        print(f"Atmospheric variables for {day} already exists.")

    # --- 3. Download Static Variables ---
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
        DOWNLOAD_PATH / f"{day}-wave.grib",
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )

    atmos_vars_ds = xr.open_dataset(
        DOWNLOAD_PATH / f"{day}-atmospheric.nc",
        engine="netcdf4",
        decode_timedelta=True,
    )

    lat_min, lat_max = min(lat_bounds), max(lat_bounds)
    lon_min, lon_max = min(lon_bounds), max(lon_bounds)

    # sel(latitude=slice(max, min)) is typical for xarray when lat coordinates run from N to S
    surf_vars_ds_cropped = surf_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    wave_vars_ds_cropped = wave_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    atmos_vars_ds_cropped = atmos_vars_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    
    if surf_vars_ds_cropped.sizes['latitude'] == 0 or surf_vars_ds_cropped.sizes['longitude'] == 0:
        raise ValueError(f"No data found for the specified region Lat: {lat_bounds}, Lon: {lon_bounds}. "
                         f"Please check the coordinates. Global data is at 0.25x0.25 resolution. "
                         f"Cropped latitude range: {surf_vars_ds_cropped.latitude.values.min():.2f}-{surf_vars_ds_cropped.latitude.values.max():.2f}, "
                         f"longitude range: {surf_vars_ds_cropped.longitude.values.min():.2f}-{surf_vars_ds_cropped.longitude.values.max():.2f}")


    surf_vars_input = {
        "2t": _prepare_hres(surf_vars_ds_cropped["2m_temperature"].values),
        "10u": _prepare_hres(surf_vars_ds_cropped["10m_u_component_of_wind"].values),
        "10v": _prepare_hres(surf_vars_ds_cropped["10m_v_component_of_wind"].values),
        "msl": _prepare_hres(surf_vars_ds_cropped["mean_sea_level_pressure"].values),
    }
    
    for aurora_var, ecmwf_var in {
        "swh": "swh", "mwd": "mwd", "mwp": "mwp", "pp1d": "pp1d", "shww": "shww",
        "mdww": "mdww", "mpww": "mpww", "shts": "shts", "mdts": "mdts", "mpts": "mpts",
        "swh1": "swh1", "mwd1": "mwd1", "mwp1": "mwp1", "swh2": "swh2",
        "mwd2": "mwd2", "mwp2": "mwp2", "wind": "wind", "dwi": "dwi"
    }.items():
        if ecmwf_var in wave_vars_ds_cropped:
            surf_vars_input[aurora_var] = _prepare_wave(wave_vars_ds_cropped[ecmwf_var].values)
        else:
            print(f"Warning: Wave variable '{ecmwf_var}' not found in downloaded data for {day}. Skipping.")


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
            lat=torch.from_numpy(surf_vars_ds_cropped.latitude.values[::-1].copy()),
            lon=torch.from_numpy(surf_vars_ds_cropped.longitude.values),
            time=(surf_vars_ds_cropped.time.values.astype("datetime64[s]").tolist()[1],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds_cropped.level.values),
        ),
    )
    return batch, surf_vars_ds_cropped

# --- Flask App Endpoints ---
try:
    aurora_model_manager = AuroraModelManager()
except RuntimeError as e:
    print(f"FATAL: {e}. Exiting Flask app setup.")
    exit(1)

@app.route('/api/predict', methods=['GET'])
def predict():
    n_lat = float(request.args.get('n_lat'))
    w_lon = float(request.args.get('w_lon'))
    s_lat = float(request.args.get('s_lat'))
    e_lon = float(request.args.get('e_lon'))
    steps = request.args.get('steps', type=int, default=2)
    prediction_date_str = request.args.get('prediction_date', type=str)

    if not all([n_lat, w_lon, s_lat, e_lon, prediction_date_str]):
        return jsonify({"error": "Missing one or more required parameters (n_lat, w_lon, s_lat, e_lon, prediction_date)."}), 400
    if not 1 <= steps <= 12:
        return jsonify({"error": "Steps must be between 1 and 12."}), 400

    try:
        model, device = aurora_model_manager.get_model()

        batch, original_cropped_ds = download_and_prepare_data(
            prediction_date_str,
            (n_lat, s_lat),
            (w_lon, e_lon)
        )
        
        batch = batch.to(device)

        print(f"Running rollout for {steps} steps for region: N:{n_lat}, W:{w_lon}, S:{s_lat}, E:{e_lon} on date {prediction_date_str}")
        with torch.inference_mode():
            preds_batches = [pred.to("cpu") for pred in rollout(model, batch, steps=steps)]

        output_data = []
        target_variables = ["swh1", "pp1d", "wind", "mwp", "shts"] 

        initial_unix_time = original_cropped_ds.time.values.astype("datetime64[s]").tolist()[1].timestamp()
        
        for i, pred_batch in enumerate(preds_batches):
            forecast_time = initial_unix_time + ((i + 1) * 6 * 3600)

            lats = original_cropped_ds.latitude.values
            lons = original_cropped_ds.longitude.values

            region_data = {
                "forecast_time_unix": int(forecast_time),
                "forecast_time_iso": datetime.datetime.fromtimestamp(forecast_time).isoformat(),
                "region_lat_min": float(min(lats)),
                "region_lat_max": float(max(lats)),
                "region_lon_min": float(min(lons)),
                "region_lon_max": float(max(lons)),
                "spatial_resolution": 0.25
            }
            
            for var_name in target_variables:
                if var_name in pred_batch.surf_vars:
                    data_points = pred_batch.surf_vars[var_name].cpu().numpy().flatten()
                    region_data[f"{var_name}_mean"] = float(np.mean(data_points))
                    region_data[f"{var_name}_min"] = float(np.min(data_points))
                    region_data[f"{var_name}_max"] = float(np.max(data_points))
                    region_data[f"{var_name}_std"] = float(np.std(data_points))
                else:
                    region_data[f"{var_name}_mean"] = None
                    region_data[f"{var_name}_min"] = None
                    region_data[f"{var_name}_max"] = None
                    region_data[f"{var_name}_std"] = None
                    print(f"Warning: Variable '{var_name}' not found in surf_vars from Aurora output.")
            
            output_data.append(region_data)

        return jsonify(output_data), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        return jsonify({"error": f"Backend processing error: {re}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    print("Starting Flask AuroraWave Backend...")
    app.run(debug=True, port=5000, use_reloader=False)
