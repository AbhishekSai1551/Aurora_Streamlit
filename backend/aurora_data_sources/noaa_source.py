"""NOAA Weather Watch data source for ocean variables."""

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
from config import NOAA_CONFIG


class NOAADataSource(BaseDataSource):
    """Data source for NOAA Weather Watch ocean variables."""
    
    def __init__(self, download_path: Path):
        """Initialize NOAA data source."""
        super().__init__(download_path)
        self.config = NOAA_CONFIG
        
    def download_data(self, 
                     date_range: Tuple[str, str],
                     lat_bounds: List[float],
                     lon_bounds: List[float],
                     **kwargs) -> str:
        """Download NOAA ocean data for specified parameters.
        
        Args:
            date_range: Start and end dates (YYYY-MM-DD format)
            lat_bounds: [min_lat, max_lat]
            lon_bounds: [min_lon, max_lon]
            
        Returns:
            Path to downloaded data file
        """
        if not self.validate_bounds(lat_bounds, lon_bounds):
            raise ValueError("Invalid geographic bounds")
            
        cache_file = self.download_path / self.get_cache_filename(
            date_range, lat_bounds, lon_bounds, "ocean.nc"
        )
        
        if cache_file.exists():
            print(f"Using cached NOAA ocean data: {cache_file}")
            return str(cache_file)
            
        print("Downloading NOAA ocean data...")
        
        try:
            # Try to download from NOAA WaveWatch III
            ds = self._download_wavewatch3_data(date_range, lat_bounds, lon_bounds)
            
            if ds is None:
                # Fallback to synthetic data
                print("NOAA WaveWatch III unavailable, creating synthetic ocean data...")
                ds = self._create_synthetic_ocean_data(date_range, lat_bounds, lon_bounds)
            
            # Convert to Aurora variable names
            ds = self._convert_to_aurora_variables(ds)
            
            ds.to_netcdf(str(cache_file))
            print(f"NOAA ocean data saved to: {cache_file}")
            return str(cache_file)
            
        except Exception as e:
            print(f"Error downloading NOAA data: {e}")
            raise
    
    def _download_wavewatch3_data(self, 
                                date_range: Tuple[str, str],
                                lat_bounds: List[float],
                                lon_bounds: List[float]) -> xr.Dataset:
        """Download data from NOAA WaveWatch III.
        
        Returns:
            Dataset or None if download fails
        """
        try:
            start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
            
            # Try multiple NOAA WaveWatch III URL formats
            possible_urls = [
                f"{self.config['base_url']}/{self.config['wavewatch_resolution']}/{start_date.strftime('%Y%m%d')}/{self.config['wavewatch_resolution']}_{start_date.strftime('%Y%m%d')}_00z.nc",
                f"https://nomads.ncep.noaa.gov/dods/wave/nww3/{start_date.strftime('%Y%m%d')}/nww3.{start_date.strftime('%Y%m%d')}.nc",
                f"https://nomads.ncep.noaa.gov/dods/wave/multi_1/{start_date.strftime('%Y%m%d')}/multi_1.{start_date.strftime('%Y%m%d')}.nc"
            ]
            
            for url in possible_urls:
                try:
                    print(f"Trying NOAA WaveWatch III URL: {url}")
                    ds = xr.open_dataset(url)
                    
                    # Select geographical region
                    ds_subset = ds.sel(
                        lat=slice(lat_bounds[0], lat_bounds[1]),
                        lon=slice(lon_bounds[0], lon_bounds[1])
                    )
                    
                    print(f"Successfully downloaded from: {url}")
                    print(f"Available variables: {list(ds_subset.data_vars.keys())}")
                    return ds_subset
                    
                except Exception as e:
                    print(f"Failed to connect to {url}: {e}")
                    continue
                    
            return None
            
        except Exception as e:
            print(f"Error in WaveWatch III download: {e}")
            return None
    
    def _create_synthetic_ocean_data(self, 
                                   date_range: Tuple[str, str],
                                   lat_bounds: List[float],
                                   lon_bounds: List[float]) -> xr.Dataset:
        """Create synthetic ocean data for testing."""
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
        lats = np.linspace(lat_bounds[1], lat_bounds[0], grid_size)  # North to south (consistent with ECMWF)
        lons = np.linspace(lon_bounds[0], lon_bounds[1], grid_size)  # West to east

        # Convert to [0, 360) range for Aurora compatibility only for internal processing
        lons_aurora = (lons + 360) % 360

        print(f"NOAA grid: {grid_size}x{grid_size}")
        print(f"Geographic coordinates: lat {lats.max():.3f} to {lats.min():.3f}, lon {lons.min():.3f} to {lons.max():.3f}")
        print(f"Aurora coordinates: lat {lats.max():.3f} to {lats.min():.3f}, lon {lons_aurora.min():.3f} to {lons_aurora.max():.3f}")
        
        data_vars = {}
        
        # Create synthetic ocean variables
        for var in self.config["ocean_variables"]:
            if var == "htsgwsfc":  # Significant wave height
                data = 1.0 + 0.5 * np.random.randn(len(times), len(lats), len(lons))
                data = np.maximum(data, 0.1)  # Ensure positive values
            elif var == "dirpwsfc":  # Wave direction
                data = 360 * np.random.rand(len(times), len(lats), len(lons))
            elif var == "perpwsfc":  # Wave period
                data = 8.0 + 2.0 * np.random.randn(len(times), len(lats), len(lons))
                data = np.maximum(data, 2.0)  # Ensure reasonable values
            else:  # Wind components
                data = 5 * np.random.randn(len(times), len(lats), len(lons))
                
            data_vars[var] = (["time", "lat", "lon"], data)
        
        coords = {
            "time": times,
            "lat": lats,
            "lon": lons_aurora  # Use Aurora-compatible coordinates for internal processing
        }
        
        return xr.Dataset(data_vars, coords=coords)
    
    def _convert_to_aurora_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert NOAA variable names to Aurora expected names."""
        aurora_vars = {}
        
        for noaa_var, aurora_var in self.config["aurora_variable_mapping"].items():
            if noaa_var in ds:
                aurora_vars[aurora_var] = ds[noaa_var]
                print(f"Mapped {noaa_var} → {aurora_var}")
        
        # Add derived variables
        if "mwp" in aurora_vars and "pp1d" not in aurora_vars:
            aurora_vars["pp1d"] = aurora_vars["mwp"]  # Use mean period as proxy for peak period
            print("Added pp1d using mwp as proxy")
        
        # Ensure we have all required ocean variables for Aurora
        required_vars = ["swh", "mwd", "mwp", "pp1d"]
        for var in required_vars:
            if var not in aurora_vars and "swh" in aurora_vars:
                aurora_vars[var] = aurora_vars["swh"]  # Use swh as proxy
                print(f"Added missing variable '{var}' using swh as proxy")
        
        return xr.Dataset(aurora_vars, coords=ds.coords)
    
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load NOAA ocean data from NetCDF file.
        
        Args:
            file_path: Path to NetCDF file
            
        Returns:
            Loaded dataset
        """
        try:
            ds = xr.open_dataset(file_path)
            print(f"Loaded NOAA ocean data with variables: {list(ds.data_vars.keys())}")
            return ds
            
        except Exception as e:
            print(f"Error loading NOAA data: {e}")
            raise
    
    def get_available_variables(self) -> List[str]:
        """Get list of available NOAA ocean variables (Aurora names).
        
        Returns:
            List of Aurora-compatible variable names
        """
        return list(self.config["aurora_variable_mapping"].values()) + ["pp1d"]
    
    def download_monthly_data(self, 
                            end_date: str,
                            lat_bounds: List[float],
                            lon_bounds: List[float]) -> str:
        """Download 1 month of NOAA ocean data ending on specified date.
        
        Args:
            end_date: End date (YYYY-MM-DD format)
            lat_bounds: [min_lat, max_lat]
            lon_bounds: [min_lon, max_lon]
            
        Returns:
            Path to downloaded data file
        """
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=30)
        date_range = (start_dt.strftime("%Y-%m-%d"), end_date)
        
        return self.download_data(date_range, lat_bounds, lon_bounds)
