"""Aurora Wave Prediction Backend"""

import datetime
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import DOWNLOAD_PATH, API_CONFIG, DEFAULT_BOUNDS
from aurora_data_sources import HuggingFaceDataSource, ECMWFDataSource, ECMWFWaveDataSource
from aurora_interface import AuroraModelInterface

app = Flask(__name__)
CORS(app)

# Initialize data sources and model
hf_source = HuggingFaceDataSource(DOWNLOAD_PATH)
ecmwf_source = ECMWFDataSource(DOWNLOAD_PATH)
ecmwf_wave_source = ECMWFWaveDataSource(DOWNLOAD_PATH)  # Replace NOAA with ECMWF wave data
aurora_model = AuroraModelInterface(model_type="wave")

def _get_or_download_wave_data(wave_source, date_range, lat_bounds, lon_bounds):
    """Get existing wave data file or download if not available.

    This prevents duplicate downloads when prediction endpoint is called
    after the download-data endpoint.
    """
    from pathlib import Path
    import os
    from datetime import datetime, timedelta

    # Generate the expected cache filename
    cache_filename = wave_source.get_cache_filename(
        date_range, lat_bounds, lon_bounds, "wave.grib2"
    )
    cache_file = wave_source.download_path / cache_filename

    # Check for both .grib2 and .nc versions
    grib_file = cache_file
    nc_file = cache_file.with_suffix('.nc')

    # Check if files exist and are recent (within last 6 hours)
    current_time = datetime.now()
    max_age = timedelta(hours=6)

    for file_path in [grib_file, nc_file]:
        if file_path.exists():
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age < max_age:
                print(f"Using existing wave data file: {file_path}")
                return str(file_path)
            else:
                print(f"Wave data file exists but is too old ({file_age}), re-downloading...")

    # No suitable existing file found, download new data
    print("No recent wave data found, downloading...")
    return wave_source.download_data(date_range, lat_bounds, lon_bounds)

# Load static variables
try:
    static_variables = hf_source.download_and_load()
    print("Static variables loaded")
except Exception as e:
    print(f"Warning: Could not load static variables: {e}")
    static_variables = {}


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "OK",
        "message": "Aurora backend is running",
        "data_sources": {"huggingface": "Available", "ecmwf": "Available", "noaa": "Available"},
        "model": "Aurora Wave loaded"
    })


@app.route("/api/download-data", methods=["POST"])
def download_data():
    try:
        req = request.json
        end_date = req.get("end_date", datetime.datetime.now().strftime("%Y-%m-%d"))
        lat_bounds = req.get("lat_bounds", DEFAULT_BOUNDS["maldives"]["lat_bounds"])
        lon_bounds = req.get("lon_bounds", DEFAULT_BOUNDS["maldives"]["lon_bounds"])

        results = {}
        try:
            surface_file, atmospheric_file = ecmwf_source.download_monthly_data(end_date, lat_bounds, lon_bounds)
            results["ecmwf"] = {"surface_file": surface_file, "atmospheric_file": atmospheric_file, "status": "success"}
        except Exception as e:
            results["ecmwf"] = {"status": "error", "error": str(e)}

        try:
            ocean_file = ecmwf_wave_source.download_monthly_data(end_date, lat_bounds, lon_bounds)
            results["ecmwf_wave"] = {"ocean_file": ocean_file, "status": "success"}
        except Exception as e:
            results["ecmwf_wave"] = {"status": "error", "error": str(e)}

        results["huggingface"] = {"static_variables": len(static_variables), "status": "success" if static_variables else "error"}

        return jsonify({"status": "OK", "download_results": results, "message": "Data download completed"})

    except Exception as e:
        return jsonify(status="ERROR", error=str(e)), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        req = request.json
        target_date = req.get("target_date", datetime.datetime.now().strftime("%Y-%m-%d"))
        lat_bounds = req.get("lat_bounds", DEFAULT_BOUNDS["maldives"]["lat_bounds"])
        lon_bounds = req.get("lon_bounds", DEFAULT_BOUNDS["maldives"]["lon_bounds"])
        steps = int(req.get("steps", 4))

        end_dt = datetime.datetime.strptime(target_date, "%Y-%m-%d")
        start_dt = end_dt - datetime.timedelta(days=30)
        date_range = (start_dt.strftime("%Y-%m-%d"), target_date)

        # Convert longitude bounds for Aurora compatibility
        aurora_lon_bounds = [(lon + 360) % 360 for lon in lon_bounds]

        # Load data (check for existing files first to avoid duplicate downloads)
        surface_file = ecmwf_source.download_data(date_range, lat_bounds, aurora_lon_bounds, "surface")
        atmospheric_file = ecmwf_source.download_data(date_range, lat_bounds, aurora_lon_bounds, "atmospheric")

        # Check if wave data was already downloaded by the download-data endpoint
        ocean_file = _get_or_download_wave_data(ecmwf_wave_source, date_range, lat_bounds, aurora_lon_bounds)

        surface_ds = ecmwf_source.load_data(surface_file)
        atmospheric_ds = ecmwf_source.load_data(atmospheric_file)
        ocean_ds = ecmwf_wave_source.load_data(ocean_file)

        # Prepare variables and run prediction
        surf_vars = aurora_model.prepare_surface_variables(surface_ds, ocean_ds)
        atmos_vars = aurora_model.prepare_atmospheric_variables(atmospheric_ds)

        metadata_info = {
            "lat": surface_ds.latitude.values,
            "lon": surface_ds.longitude.values,
            "time": (datetime.datetime.strptime(target_date, "%Y-%m-%d"),),
            "atmos_levels": atmospheric_ds.level.values if "level" in atmospheric_ds.coords else (1000,)
        }

        batch = aurora_model.create_batch(surf_vars, atmos_vars, static_variables, metadata_info)
        predictions = aurora_model.predict(batch, steps)
        extracted_predictions = aurora_model.extract_predictions(predictions, ecmwf_wave_source.get_available_variables(), enhance_resolution=True)

        # Generate response coordinates
        base_time = datetime.datetime.strptime(target_date, "%Y-%m-%d")
        forecast_times = [(base_time + datetime.timedelta(hours=i*6)).isoformat() for i in range(steps)]

        original_lats = surface_ds.latitude.values
        original_lons = surface_ds.longitude.values

        if original_lats[0] < original_lats[-1]:
            original_lats = original_lats[::-1]

        original_lons_geo = np.where(original_lons > 180, original_lons - 360, original_lons)

        if extracted_predictions:
            sample_var = next(iter(extracted_predictions.values()))
            if sample_var:
                pred_shape = np.array(sample_var[0]).shape
                if pred_shape != (len(original_lats), len(original_lons)):
                    enhanced_lats = np.linspace(lat_bounds[1], lat_bounds[0], pred_shape[0])
                    enhanced_lons = np.linspace(lon_bounds[0], lon_bounds[1], pred_shape[1])
                    response_lats = enhanced_lats.tolist()
                    response_lons = enhanced_lons.tolist()
                else:
                    response_lats = original_lats.tolist()
                    response_lons = original_lons_geo.tolist()
            else:
                response_lats = original_lats.tolist()
                response_lons = original_lons_geo.tolist()
        else:
            response_lats = original_lats.tolist()
            response_lons = original_lons_geo.tolist()

        return jsonify({
            "status": "OK",
            "target_date": target_date,
            "forecast_times": forecast_times,
            "lats": response_lats,
            "lons": response_lons,
            "predictions": extracted_predictions,
            "model": "Aurora Wave",
            "prediction_steps": steps
        })

    except Exception as e:
        return jsonify(status="ERROR", error=str(e)), 500

@app.route("/api/predict_oceanic", methods=["POST"])
def predict_oceanic():
    return predict()

if __name__ == "__main__":
    print("Starting Aurora Wave Prediction Backend...")
    app.run(host=API_CONFIG["host"], port=API_CONFIG["port"], debug=API_CONFIG["debug"])