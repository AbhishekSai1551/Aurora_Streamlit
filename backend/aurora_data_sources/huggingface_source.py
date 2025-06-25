"""HuggingFace data source for Aurora static variables."""

import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import xarray as xr
from huggingface_hub import hf_hub_download

from .base import BaseDataSource
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUGGINGFACE_CONFIG


class HuggingFaceDataSource(BaseDataSource):
    """Data source for HuggingFace Aurora static variables."""
    
    def __init__(self, download_path: Path):
        """Initialize HuggingFace data source."""
        super().__init__(download_path)
        self.config = HUGGINGFACE_CONFIG
        
    def download_data(self, 
                     date_range: Tuple[str, str] = None,
                     lat_bounds: List[float] = None,
                     lon_bounds: List[float] = None,
                     **kwargs) -> str:
        """Download static variables from HuggingFace.
        
        Note: Static variables don't depend on date/location parameters.
        
        Returns:
            Path to downloaded static variables file
        """
        try:
            static_path = hf_hub_download(
                repo_id=self.config["static_variables"]["repo_id"],
                filename=self.config["static_variables"]["filename"],
                cache_dir=str(self.download_path)
            )
            print(f"Downloaded static variables from HuggingFace: {static_path}")
            return static_path
            
        except Exception as e:
            print(f"Error downloading from HuggingFace: {e}")
            raise
    
    def load_data(self, file_path: str) -> Dict[str, Any]:
        """Load static variables from pickle file.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Dictionary of static variables
        """
        try:
            with open(file_path, 'rb') as f:
                static_vars = pickle.load(f)
            print(f"Loaded static variables: {list(static_vars.keys())}")
            return static_vars
            
        except Exception as e:
            print(f"Error loading static variables: {e}")
            return {}
    
    def get_available_variables(self) -> List[str]:
        """Get list of available static variables.
        
        Returns:
            List of static variable names
        """
        # Standard Aurora static variables
        return ["z", "slt", "lsm"]  # geopotential, soil type, land-sea mask
    
    def download_and_load(self) -> Dict[str, Any]:
        """Download and load static variables in one step.
        
        Returns:
            Dictionary of loaded static variables
        """
        file_path = self.download_data()
        return self.load_data(file_path)
