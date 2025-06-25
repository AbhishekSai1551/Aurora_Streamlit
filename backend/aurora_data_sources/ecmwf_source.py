"""ECMWF open data source for atmospheric and surface variables."""

import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
import xarray as xr
import numpy as np

from .base import BaseDataSource
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ECMWF_CONFIG


class ECMWFDataSource(BaseDataSource):
    """Data source for ECMWF open data atmospheric and surface variables."""
    
    def __init__(self, download_path: Path):
        """Initialize ECMWF data source."""
        super().__init__(download_path)
        self.config = ECMWF_CONFIG
        
    def download_data(self, 
                     date_range: Tuple[str, str],
                     lat_bounds: List[float],
                     lon_bounds: List[float],
                     variable_type: str = "both",
                     **kwargs) -> str:
        """Download ECMWF data for specified parameters.
        
        Args:
            date_range: Start and end dates (YYYY-MM-DD format)
            lat_bounds: [min_lat, max_lat]
            lon_bounds: [min_lon, max_lon]
            variable_type: "atmospheric", "surface", or "both"
            
        Returns:
            Path to downloaded data file
        """
        if not self.validate_bounds(lat_bounds, lon_bounds):
            raise ValueError("Invalid geographic bounds")
            
        cache_file = self.download_path / self.get_cache_filename(
            date_range, lat_bounds, lon_bounds, f"{variable_type}.nc"
        )
        
        if cache_file.exists():
            print(f"Using cached ECMWF data: {cache_file}")
            return str(cache_file)
            
        print(f"Downloading ECMWF {variable_type} data...")
        
        try:
            # For now, create synthetic data that matches ECMWF structure
            # In production, this would use the actual ECMWF API
            ds = self._create_synthetic_ecmwf_data(date_range, lat_bounds, lon_bounds, variable_type)
            ds.to_netcdf(str(cache_file))
            print(f"ECMWF {variable_type} data saved to: {cache_file}")
            return str(cache_file)
            
        except Exception as e:
            print(f"Error downloading ECMWF data: {e}")
            raise
    
    def _create_synthetic_ecmwf_data(self, 
                                   date_range: Tuple[str, str],
                                   lat_bounds: List[float],
                                   lon_bounds: List[float],
                                   variable_type: str) -> xr.Dataset:
        """Create synthetic ECMWF data for testing.
        
        In production, this would be replaced with actual ECMWF API calls.
        """
        start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
        end_date = datetime.strptime(date_range[1], "%Y-%m-%d")
        
        # Create time series (6-hourly data)
        time_delta = timedelta(hours=6)
        times = []
        current_time = start_date
        while current_time <= end_date:
            times.append(current_time)
            current_time += time_delta
            
        # Create spatial grid - Aurora requires minimum resolution
        # Use at least 32x32 grid to meet Aurora's minimum requirements
        grid_size = max(32, 8)  # Minimum 32x32 for Aurora

        # Create consistent coordinate grids (north to south for lat, west to east for lon)
        lats = np.linspace(lat_bounds[1], lat_bounds[0], grid_size)  # North to south
        lons = np.linspace(lon_bounds[0], lon_bounds[1], grid_size)  # West to east

        # Store original longitude range for reference
        original_lon_range = (lon_bounds[0], lon_bounds[1])

        # Convert to [0, 360) range for Aurora compatibility only for internal processing
        lons_aurora = (lons + 360) % 360

        print(f"ECMWF grid: {grid_size}x{grid_size}")
        print(f"Geographic coordinates: lat {lats.max():.3f} to {lats.min():.3f}, lon {lons.min():.3f} to {lons.max():.3f}")
        print(f"Aurora coordinates: lat {lats.max():.3f} to {lats.min():.3f}, lon {lons_aurora.min():.3f} to {lons_aurora.max():.3f}")
        
        data_vars = {}
        
        if variable_type in ["surface", "both"]:
            # Surface variables
            for var in self.config["surface_variables"]:
                if var == "2m_temperature":
                    data = 273.15 + 25 + 5 * np.random.randn(len(times), len(lats), len(lons))
                elif var == "mean_sea_level_pressure":
                    data = 101325 + 1000 * np.random.randn(len(times), len(lats), len(lons))
                else:  # Wind components
                    data = 5 * np.random.randn(len(times), len(lats), len(lons))
                    
                data_vars[var] = (["time", "latitude", "longitude"], data)
        
        if variable_type in ["atmospheric", "both"]:
            # Atmospheric variables with pressure levels
            levels = self.config["pressure_levels"][:5]  # Use first 5 levels
            
            for var in self.config["atmospheric_variables"][:3]:  # Use first 3 variables
                if var == "temperature":
                    data = 273.15 + 20 + 10 * np.random.randn(len(times), len(levels), len(lats), len(lons))
                elif var in ["u_component_of_wind", "v_component_of_wind"]:
                    data = 10 * np.random.randn(len(times), len(levels), len(lats), len(lons))
                else:
                    data = np.random.randn(len(times), len(levels), len(lats), len(lons))
                    
                data_vars[var] = (["time", "level", "latitude", "longitude"], data)
        
        coords = {
            "time": times,
            "latitude": lats,
            "longitude": lons_aurora  # Use Aurora-compatible coordinates for internal processing
        }
        
        if variable_type in ["atmospheric", "both"]:
            coords["level"] = levels[:5]
            
        return xr.Dataset(data_vars, coords=coords)
    
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load ECMWF data from NetCDF file.
        
        Args:
            file_path: Path to NetCDF file
            
        Returns:
            Loaded dataset
        """
        try:
            ds = xr.open_dataset(file_path)
            print(f"Loaded ECMWF data with variables: {list(ds.data_vars.keys())}")
            return ds
            
        except Exception as e:
            print(f"Error loading ECMWF data: {e}")
            raise
    
    def get_available_variables(self) -> List[str]:
        """Get list of available ECMWF variables.
        
        Returns:
            List of variable names
        """
        return self.config["atmospheric_variables"] + self.config["surface_variables"]
    
    def download_monthly_data(self, 
                            end_date: str,
                            lat_bounds: List[float],
                            lon_bounds: List[float]) -> Tuple[str, str]:
        """Download 1 month of ECMWF data ending on specified date.
        
        Args:
            end_date: End date (YYYY-MM-DD format)
            lat_bounds: [min_lat, max_lat]
            lon_bounds: [min_lon, max_lon]
            
        Returns:
            Tuple of (surface_file_path, atmospheric_file_path)
        """
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=30)
        date_range = (start_dt.strftime("%Y-%m-%d"), end_date)
        
        surface_file = self.download_data(date_range, lat_bounds, lon_bounds, "surface")
        atmospheric_file = self.download_data(date_range, lat_bounds, lon_bounds, "atmospheric")
        
        return surface_file, atmospheric_file
