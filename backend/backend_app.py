"""Aurora Wave Prediction Backend"""

import datetime
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import DOWNLOAD_PATH, API_CONFIG, DEFAULT_BOUNDS
from aurora_data_sources import HuggingFaceDataSource, ECMWFDataSource, NOAADataSource
from aurora_interface import AuroraModelInterface

app = Flask(__name__)
CORS(app)

# Initialize data sources and model
hf_source = HuggingFaceDataSource(DOWNLOAD_PATH)
ecmwf_source = ECMWFDataSource(DOWNLOAD_PATH)
noaa_source = NOAADataSource(DOWNLOAD_PATH)
aurora_model = AuroraModelInterface(model_type="wave")

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
            ocean_file = noaa_source.download_monthly_data(end_date, lat_bounds, lon_bounds)
            results["noaa"] = {"ocean_file": ocean_file, "status": "success"}
        except Exception as e:
            results["noaa"] = {"status": "error", "error": str(e)}

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

        # Load data
        surface_file = ecmwf_source.download_data(date_range, lat_bounds, aurora_lon_bounds, "surface")
        atmospheric_file = ecmwf_source.download_data(date_range, lat_bounds, aurora_lon_bounds, "atmospheric")
        ocean_file = noaa_source.download_data(date_range, lat_bounds, aurora_lon_bounds)

        surface_ds = ecmwf_source.load_data(surface_file)
        atmospheric_ds = ecmwf_source.load_data(atmospheric_file)
        ocean_ds = noaa_source.load_data(ocean_file)

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
        extracted_predictions = aurora_model.extract_predictions(predictions, noaa_source.get_available_variables(), enhance_resolution=True)

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