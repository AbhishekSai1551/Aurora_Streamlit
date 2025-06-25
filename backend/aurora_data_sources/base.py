"""Base class for data sources."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import xarray as xr


class BaseDataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, download_path: Path):
        """Initialize data source.
        
        Args:
            download_path: Path to store downloaded data
        """
        self.download_path = download_path
        self.download_path.mkdir(exist_ok=True)
    
    @abstractmethod
    def download_data(self, 
                     date_range: Tuple[str, str],
                     lat_bounds: List[float],
                     lon_bounds: List[float],
                     **kwargs) -> str:
        """Download data for specified parameters.
        
        Args:
            date_range: Start and end dates (YYYY-MM-DD format)
            lat_bounds: [min_lat, max_lat]
            lon_bounds: [min_lon, max_lon]
            **kwargs: Additional parameters specific to data source
            
        Returns:
            Path to downloaded data file
        """
        pass
    
    @abstractmethod
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load data from file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Loaded dataset
        """
        pass
    
    @abstractmethod
    def get_available_variables(self) -> List[str]:
        """Get list of available variables from this data source.
        
        Returns:
            List of variable names
        """
        pass
    
    def validate_bounds(self, lat_bounds: List[float], lon_bounds: List[float]) -> bool:
        """Validate geographic bounds.
        
        Args:
            lat_bounds: [min_lat, max_lat]
            lon_bounds: [min_lon, max_lon]
            
        Returns:
            True if bounds are valid
        """
        if not (-90 <= lat_bounds[0] <= lat_bounds[1] <= 90):
            return False
        if not (0 <= lon_bounds[0] <= lon_bounds[1] <= 360):
            return False
        return True
    
    def get_cache_filename(self, 
                          date_range: Tuple[str, str],
                          lat_bounds: List[float], 
                          lon_bounds: List[float],
                          suffix: str = "nc") -> str:
        """Generate cache filename for given parameters.
        
        Args:
            date_range: Start and end dates
            lat_bounds: Latitude bounds
            lon_bounds: Longitude bounds
            suffix: File extension
            
        Returns:
            Cache filename
        """
        start_date, end_date = date_range
        lat_str = f"{lat_bounds[0]:.1f}-{lat_bounds[1]:.1f}"
        lon_str = f"{lon_bounds[0]:.1f}-{lon_bounds[1]:.1f}"
        return f"{self.__class__.__name__.lower()}_{start_date}_{end_date}_{lat_str}_{lon_str}.{suffix}"
