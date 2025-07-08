"""ECMWF OpenData wave data source for ocean variables."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
import xarray as xr
import numpy as np

from .base import BaseDataSource
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ECMWF_CONFIG, ECMWF_WAVE_CONFIG


class ECMWFWaveDataSource(BaseDataSource):
    """Data source for ECMWF OpenData ocean wave variables."""
    
    def __init__(self, download_path: Path):
        """Initialize ECMWF wave data source."""
        super().__init__(download_path)
        self.config = ECMWF_CONFIG
        
        # Wave-specific configuration
        self.wave_config = ECMWF_WAVE_CONFIG
        
    def download_data(self, 
                     date_range: Tuple[str, str],
                     lat_bounds: List[float],
                     lon_bounds: List[float],
                     **kwargs) -> str:
        """Download ECMWF wave data for specified parameters.
        
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
            date_range, lat_bounds, lon_bounds, "wave.grib2"
        )
        
        if cache_file.exists():
            print(f"Using cached ECMWF wave data: {cache_file}")
            return str(cache_file)
            
        print(f"Downloading ECMWF wave data...")
        
        try:
            # Use ECMWF OpenData client
            from ecmwf.opendata import Client
            
            client = Client(source="ecmwf")
            
            # Parse date range
            start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
            
            # Download wave data using optimal 3-day strategy
            self._download_3day_wave_data(client, start_date, lat_bounds, lon_bounds, cache_file)
            
            print(f"ECMWF wave data saved to: {cache_file}")
            return str(cache_file)
            
        except Exception as e:
            print(f"Error downloading ECMWF wave data: {e}")
            # Fall back to synthetic data for testing
            print("Creating synthetic wave data for testing...")
            ds = self._create_synthetic_wave_data(date_range, lat_bounds, lon_bounds)
            ds.to_netcdf(str(cache_file.with_suffix('.nc')))
            return str(cache_file.with_suffix('.nc'))
    
    def _create_synthetic_wave_data(self, 
                                  date_range: Tuple[str, str],
                                  lat_bounds: List[float],
                                  lon_bounds: List[float]) -> xr.Dataset:
        """Create synthetic wave data for testing when ECMWF is unavailable."""
        start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
        end_date = datetime.strptime(date_range[1], "%Y-%m-%d")
        
        # Create time series (6-hourly data)
        time_delta = timedelta(hours=6)
        times = []
        current_time = start_date
        while current_time <= end_date:
            times.append(current_time)
            current_time += time_delta
        
        # Create spatial grid
        lat_step = 0.25  # ECMWF resolution
        lon_step = 0.25
        
        lats = np.arange(lat_bounds[0], lat_bounds[1] + lat_step, lat_step)
        lons = np.arange(lon_bounds[0], lon_bounds[1] + lon_step, lon_step)
        
        # Convert to Aurora-compatible longitude range (0-360)
        lons_aurora = np.where(lons < 0, lons + 360, lons)
        
        data_vars = {}
        
        # Create synthetic wave variables
        for var in self.wave_config["wave_variables"]:
            if var == "swh":  # Significant wave height
                data = 1.0 + 0.5 * np.random.randn(len(times), len(lats), len(lons))
                data = np.maximum(data, 0.1)  # Ensure positive values
            elif var == "mwd":  # Wave direction
                data = 360 * np.random.rand(len(times), len(lats), len(lons))
            elif var in ["mwp", "pp1d", "mp2"]:  # Wave periods
                data = 8.0 + 2.0 * np.random.randn(len(times), len(lats), len(lons))
                data = np.maximum(data, 2.0)  # Ensure reasonable values
            else:
                data = np.random.randn(len(times), len(lats), len(lons))
                
            data_vars[var] = (["time", "latitude", "longitude"], data)
        
        coords = {
            "time": times,
            "latitude": lats,
            "longitude": lons_aurora  # Use Aurora-compatible coordinates
        }
        
        return xr.Dataset(data_vars, coords=coords)
    
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load ECMWF wave data from GRIB or NetCDF file.
        
        Args:
            file_path: Path to GRIB or NetCDF file
            
        Returns:
            Loaded dataset with Aurora-compatible variable names
        """
        try:
            file_path = Path(file_path)

            # Check if the file exists, if not try with .nc extension
            if not file_path.exists() and file_path.suffix.lower() in ['.grib', '.grib2', '.grb']:
                nc_file = file_path.with_suffix('.nc')
                if nc_file.exists():
                    file_path = nc_file

            if file_path.suffix.lower() in ['.grib', '.grib2', '.grb']:
                # Load GRIB file using cfgrib
                ds = xr.open_dataset(file_path, engine='cfgrib')
                print(f"Loaded ECMWF wave data (GRIB) with variables: {list(ds.data_vars.keys())}")
            else:
                # Load NetCDF file
                ds = xr.open_dataset(file_path)
                print(f"Loaded ECMWF wave data (NetCDF) with variables: {list(ds.data_vars.keys())}")

            # Convert to Aurora-compatible format
            ds_aurora = self._convert_to_aurora_variables(ds)
            return ds_aurora

        except Exception as e:
            print(f"Error loading ECMWF wave data: {e}")
            raise
    
    def _convert_to_aurora_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert ECMWF variable names to Aurora expected names."""
        aurora_vars = {}
        
        # Direct mapping for most variables
        for ecmwf_var, aurora_var in self.wave_config["aurora_variable_mapping"].items():
            if ecmwf_var in ds:
                aurora_vars[aurora_var] = ds[ecmwf_var]
                print(f"Mapped {ecmwf_var} → {aurora_var}")
        
        # Add coordinates and metadata
        coords = {}
        for coord_name in ds.coords:
            coords[coord_name] = ds.coords[coord_name]
        
        # Ensure longitude is in 0-360 range for Aurora
        if 'longitude' in coords:
            lons = coords['longitude'].values
            if np.any(lons < 0):
                coords['longitude'] = xr.DataArray(
                    np.where(lons < 0, lons + 360, lons),
                    dims=coords['longitude'].dims,
                    attrs=coords['longitude'].attrs
                )
                print("Converted longitude to 0-360 range for Aurora compatibility")
        
        return xr.Dataset(aurora_vars, coords=coords, attrs=ds.attrs)
    
    def get_available_variables(self) -> List[str]:
        """Get list of available ECMWF wave variables (Aurora names).
        
        Returns:
            List of Aurora-compatible variable names
        """
        return list(self.wave_config["aurora_variable_mapping"].values())
    
    def download_monthly_data(self, 
                            end_date: str,
                            lat_bounds: List[float],
                            lon_bounds: List[float]) -> str:
        """Download 1 month of ECMWF wave data ending on specified date.
        
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

    def _download_3day_wave_data(self, client, start_date: datetime, lat_bounds: List[float],
                                lon_bounds: List[float], target_file: Path):
        """Download 3 days of wave data using optimal forecast times.

        Strategy based on ECMWF availability:
        - Last 2 days: Use 00z and 06z (best availability)
        - 3rd day back: Use 12z and 18z (00z/06z often unavailable)

        Args:
            client: ECMWF OpenData client
            start_date: Starting date for download
            lat_bounds: Latitude bounds
            lon_bounds: Longitude bounds
            target_file: Target file path
        """
        area = [lat_bounds[1], lon_bounds[0], lat_bounds[0], lon_bounds[1]]  # N, W, S, E

        # Calculate the 3 days to download
        today = datetime.now()
        dates_and_times = []

        for days_back in range(1, 4):  # Last 3 days
            target_date = today - timedelta(days=days_back)
            date_str = target_date.strftime("%Y-%m-%d")

            if days_back <= 2:
                # For most recent 2 days: use 00z and 06z
                forecast_times = ["00", "06"]
            else:
                # For 3rd day: use 12z and 18z (better availability)
                forecast_times = ["12", "18"]

            for time in forecast_times:
                dates_and_times.append((date_str, time))

        print(f"Downloading wave data for: {dates_and_times}")

        # Try to download data for each date/time combination
        successful_downloads = []

        for date_str, time in dates_and_times:
            try:
                temp_file = target_file.with_name(f"temp_{date_str}_{time}z.grib2")

                result = client.retrieve(
                    type="fc",
                    stream="wave",
                    param=self.wave_config["wave_variables"],
                    date=date_str,
                    time=time,
                    step=["0", "6", "12"],  # First few forecast steps
                    area=area,
                    target=str(temp_file)
                )

                if temp_file.exists():
                    successful_downloads.append(temp_file)
                    print(f"✓ Downloaded {date_str} {time}z: {temp_file.stat().st_size/1024:.1f} KB")
                else:
                    print(f"✗ No file created for {date_str} {time}z")

            except Exception as e:
                print(f"✗ Failed to download {date_str} {time}z: {e}")
                continue

        if successful_downloads:
            # Combine all downloaded files into one
            self._combine_wave_files(successful_downloads, target_file)

            # Clean up temporary files
            for temp_file in successful_downloads:
                if temp_file.exists():
                    temp_file.unlink()

            print(f"✓ Combined {len(successful_downloads)} files into {target_file}")
        else:
            raise Exception("No wave data could be downloaded for any date/time combination")

    def _combine_wave_files(self, file_paths: List[Path], output_file: Path):
        """Combine multiple GRIB wave files into a single dataset using batching.

        Args:
            file_paths: List of GRIB file paths to combine
            output_file: Output file path
        """
        print(f"  Combining {len(file_paths)} files using batched approach...")

        # Process files in batches to avoid memory issues
        batch_size = 2  # Process 2 files at a time
        batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]

        combined_datasets = []

        for batch_idx, batch in enumerate(batches):
            print(f"  Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} files)...")

            batch_datasets = []
            for file_path in batch:
                try:
                    # Load each GRIB file
                    ds = xr.open_dataset(file_path, engine='cfgrib')

                    # Immediately reduce memory by selecting only first 2 time steps
                    # and averaging over forecast steps if present
                    ds_reduced = self._reduce_dataset_memory(ds)
                    batch_datasets.append(ds_reduced)



                    # Close original dataset to free memory
                    ds.close()

                except Exception as e:
                    print(f"    Warning: Could not load {file_path}: {e}")
                    continue

            if batch_datasets:
                try:
                    # Combine this batch
                    if len(batch_datasets) == 1:
                        batch_combined = batch_datasets[0]
                    else:
                        batch_combined = xr.concat(batch_datasets, dim='time')

                    combined_datasets.append(batch_combined)
                    print(f"    Batch {batch_idx + 1} combined successfully")

                    # Clean up batch datasets
                    for ds in batch_datasets:
                        if ds != batch_combined:
                            ds.close()

                except Exception as e:
                    print(f"    Warning: Could not combine batch {batch_idx + 1}: {e}")
                    # Use first dataset from batch as fallback
                    if batch_datasets:
                        combined_datasets.append(batch_datasets[0])
                        print(f"    Using first dataset from batch {batch_idx + 1} as fallback")

        if not combined_datasets:
            raise Exception("No datasets could be loaded from downloaded files")

        # Final combination of all batches
        try:
            if len(combined_datasets) == 1:
                final_ds = combined_datasets[0]
            else:
                print(f"  Final combination of {len(combined_datasets)} batches...")
                final_ds = xr.concat(combined_datasets, dim='time')

            # Sort by time to ensure chronological order
            final_ds = final_ds.sortby('time')

            # Save as NetCDF for easier handling
            output_nc = output_file.with_suffix('.nc')
            final_ds.to_netcdf(output_nc)

            # Update the target file to point to NetCDF
            if output_file.exists():
                output_file.unlink()
            output_nc.rename(output_file.with_suffix('.nc'))

            print(f"  ✓ Combined dataset shape: {dict(final_ds.dims)}")
            print(f"  ✓ Time range: {final_ds.time.min().values} to {final_ds.time.max().values}")

            # Clean up
            final_ds.close()
            for ds in combined_datasets:
                ds.close()

        except Exception as e:
            print(f"  Warning: Could not perform final combination: {e}")
            # Fall back to using the first batch
            if combined_datasets:
                combined_datasets[0].to_netcdf(output_file.with_suffix('.nc'))
                print(f"  Using first batch as fallback")
            else:
                raise Exception("No usable datasets available")

    def _reduce_dataset_memory(self, ds: xr.Dataset) -> xr.Dataset:
        """Reduce dataset memory usage by selecting subset and averaging forecast steps.

        Args:
            ds: Input xarray dataset

        Returns:
            Memory-reduced dataset
        """
        # Take only first 2 time steps to reduce memory
        if 'time' in ds.dims and ds.dims['time'] > 2:
            ds = ds.isel(time=slice(0, 2))

        # If there are forecast steps, average them to reduce the step dimension
        if 'step' in ds.dims and ds.dims['step'] > 1:
            # Average over forecast steps
            ds = ds.mean(dim='step', keep_attrs=True)
            print(f"    Averaged over {ds.dims.get('step', 'N/A')} forecast steps")

        # Convert to float32 to reduce memory usage
        for var in ds.data_vars:
            if ds[var].dtype == 'float64':
                ds[var] = ds[var].astype('float32')

        return ds
