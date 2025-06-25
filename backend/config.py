"""Configuration settings for Aurora backend application."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# --- Paths ---
DOWNLOAD_PATH = Path("./downloads")
DOWNLOAD_PATH.mkdir(exist_ok=True)

# --- Aurora Model Configuration ---
AURORA_MODEL_CONFIGS = {
    "wave": {
        "model_class": "AuroraWave",
        "checkpoint_repo": "microsoft/aurora",
        "checkpoint_name": "aurora-0.25-wave-static.pickle"
    },
    "standard": {
        "model_class": "Aurora",
        "checkpoint_repo": "microsoft/aurora", 
        "checkpoint_name": "aurora-0.25-finetuned.ckpt"
    }
}

# --- HuggingFace Configuration ---
HUGGINGFACE_CONFIG = {
    "static_variables": {
        "repo_id": "microsoft/aurora",
        "filename": "aurora-0.25-wave-static.pickle"
    }
}

# --- ECMWF Open Data Configuration ---
ECMWF_CONFIG = {
    "base_url": "https://data.ecmwf.int/forecasts",
    "dataset": "ifs-hres",
    "atmospheric_variables": [
        "temperature",
        "u_component_of_wind", 
        "v_component_of_wind",
        "specific_humidity",
        "geopotential"
    ],
    "surface_variables": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind", 
        "2m_temperature",
        "mean_sea_level_pressure"
    ],
    "pressure_levels": [1000, 925, 850, 700, 500, 300, 250, 200, 150, 100, 70, 50, 30],
    "time_range_days": 30,
    "resolution": "0.25"
}

# --- NOAA Weather Watch Configuration ---
NOAA_CONFIG = {
    "base_url": "https://nomads.ncep.noaa.gov/dods/wave/nww3",
    "wavewatch_resolution": "glo_30m",
    "ocean_variables": [
        "htsgwsfc",  # Significant wave height -> swh
        "dirpwsfc",  # Wave direction -> mwd  
        "perpwsfc",  # Wave period -> mwp
        "ugrdsfc",   # U wind component -> 10u
        "vgrdsfc"    # V wind component -> 10v
    ],
    "aurora_variable_mapping": {
        "htsgwsfc": "swh",
        "dirpwsfc": "mwd", 
        "perpwsfc": "mwp",
        "ugrdsfc": "10u",
        "vgrdsfc": "10v"
    }
}

# --- Aurora Variable Requirements ---
AURORA_VARIABLES = {
    "surface": {
        "required": ["2t", "10u", "10v", "msl"],
        "optional": ["swh", "mwd", "mwp", "pp1d"],
        "wave_specific": ["shww", "mdww", "mpww", "shts", "mdts", "mpts",
                         "swh1", "mwd1", "mwp1", "swh2", "mwd2", "mwp2", "wind", "dwi"]
    },
    "atmospheric": {
        "required": ["t", "u", "v"],
        "optional": ["q", "z"]
    },
    "static": {
        "required": ["z", "slt", "lsm"],
        "optional": []
    }
}

# --- Data Processing Configuration ---
PROCESSING_CONFIG = {
    "batch_size": 1,
    "time_steps": 2,
    "spatial_resolution": {
        "height": 8,
        "width": 8
    },
    "coordinate_system": {
        "latitude_range": (-90, 90),
        "longitude_range": (0, 360)  # Aurora expects 0-360 range
    }
}

# --- API Configuration ---
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "cors_enabled": True
}

# --- Default Geographic Bounds ---
DEFAULT_BOUNDS = {
    "maldives": {
        "lat_bounds": [2.2, 4.2],
        "lon_bounds": [72.2, 74.2]
    },
    "global": {
        "lat_bounds": [-90, 90],
        "lon_bounds": [0, 360]
    }
}
