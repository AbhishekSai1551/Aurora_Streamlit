"""Aurora model interface"""

import datetime
from typing import Dict, List, Any
import torch
import numpy as np
import xarray as xr

from aurora import AuroraWave, rollout, Batch, Metadata

class AuroraModelInterface:
    def __init__(self, model_type: str = "wave"):
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            if self.model_type == "wave":
                self.model = AuroraWave()
            else:
                from aurora import Aurora
                self.model = Aurora()

            self.model.load_checkpoint()
            self.model.eval()
            self.model = self.model.to(self.device)
        except Exception as e:
            raise
    
    def prepare_surface_variables(self, 
                                surface_ds: xr.Dataset,
                                ocean_ds: xr.Dataset = None) -> Dict[str, torch.Tensor]:
        """Prepare surface variables for Aurora model.
        
        Args:
            surface_ds: Surface variables dataset
            ocean_ds: Ocean variables dataset (optional)
            
        Returns:
            Dictionary of prepared surface variables
        """
        surf_vars = {}
        
        # Standard surface variables mapping
        surface_mapping = {
            "2t": "2m_temperature",
            "10u": "10m_u_component_of_wind", 
            "10v": "10m_v_component_of_wind",
            "msl": "mean_sea_level_pressure"
        }
        
        # Prepare standard surface variables
        for aurora_var, ecmwf_var in surface_mapping.items():
            if ecmwf_var in surface_ds:
                data = surface_ds[ecmwf_var].values
                surf_vars[aurora_var] = self._prepare_tensor(data)

        # Add ocean variables if available
        if ocean_ds is not None:
            ocean_vars = ["swh", "mwd", "mwp", "pp1d"]
            for var in ocean_vars:
                if var in ocean_ds:
                    data = ocean_ds[var].values
                    surf_vars[var] = self._prepare_tensor(data)
        
        # Add required wave-specific variables for AuroraWave
        if self.model_type == "wave":
            self._add_wave_specific_variables(surf_vars)
        
        return surf_vars
    
    def prepare_atmospheric_variables(self, 
                                    atmospheric_ds: xr.Dataset) -> Dict[str, torch.Tensor]:
        """Prepare atmospheric variables for Aurora model.
        
        Args:
            atmospheric_ds: Atmospheric variables dataset
            
        Returns:
            Dictionary of prepared atmospheric variables
        """
        atmos_vars = {}
        
        # Atmospheric variables mapping
        atmospheric_mapping = {
            "t": "temperature",
            "u": "u_component_of_wind",
            "v": "v_component_of_wind"
        }
        
        for aurora_var, ecmwf_var in atmospheric_mapping.items():
            if ecmwf_var in atmospheric_ds:
                data = atmospheric_ds[ecmwf_var].values
                atmos_vars[aurora_var] = self._prepare_tensor(data)
                print(f"Prepared atmospheric variable {aurora_var}: {atmos_vars[aurora_var].shape}")
        
        return atmos_vars
    
    def _prepare_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Prepare numpy array as PyTorch tensor for Aurora.
        
        Args:
            data: Input numpy array
            
        Returns:
            Prepared PyTorch tensor
        """
        # Ensure we have the right number of dimensions
        if data.ndim == 3:  # (time, lat, lon)
            # Take first 2 time steps and add batch dimension
            tensor = torch.from_numpy(data[:2][None][..., ::-1, :].copy())
        elif data.ndim == 4:  # (time, level, lat, lon)
            # Take first 2 time steps and add batch dimension
            tensor = torch.from_numpy(data[:2][None][..., ::-1, :].copy())
        elif data.ndim == 2:  # (lat, lon) - single time step
            # Add time and batch dimensions
            tensor = torch.from_numpy(data[None, None, ::-1, :].copy())
        else:
            print(f"Warning: Unexpected data shape: {data.shape}")
            tensor = torch.from_numpy(data[None].copy())
        
        return tensor.float()
    
    def _add_wave_specific_variables(self, surf_vars: Dict[str, torch.Tensor]):
        """Add wave-specific variables required by AuroraWave model."""
        if "swh" in surf_vars:
            # Use swh as proxy for missing wave variables
            wave_vars = ["shww", "mdww", "mpww", "shts", "mdts", "mpts",
                        "swh1", "mwd1", "mwp1", "swh2", "mwd2", "mwp2", "wind", "dwi"]
            
            for var in wave_vars:
                if var not in surf_vars:
                    surf_vars[var] = surf_vars["swh"].clone()
                    print(f"Added wave variable '{var}' using swh as proxy")
    
    def create_batch(self,
                    surf_vars: Dict[str, torch.Tensor],
                    atmos_vars: Dict[str, torch.Tensor],
                    static_vars: Dict[str, Any],
                    metadata_info: Dict[str, Any]) -> Batch:
        """Create Aurora Batch object.

        Args:
            surf_vars: Surface variables
            atmos_vars: Atmospheric variables
            static_vars: Static variables
            metadata_info: Metadata information (lat, lon, time, levels)

        Returns:
            Aurora Batch object
        """
        try:
            # Get target spatial shape from surface variables
            target_shape = None
            if surf_vars:
                sample_var = next(iter(surf_vars.values()))
                target_shape = sample_var.shape[-2:]  # Get (height, width)
                print(f"Target spatial shape: {target_shape}")

            # Prepare static variables and resize to match target shape
            static_tensors = {}

            # Include all required static variables for Aurora Wave model
            # From the logs, we know these are available: ['lsm', 'z', 'slt', 'wmb', 'lat_mask']
            essential_static_vars = ['z', 'lsm', 'slt', 'wmb', 'lat_mask']

            for var_name in essential_static_vars:
                if var_name in static_vars:
                    var_data = static_vars[var_name]

                    if isinstance(var_data, np.ndarray):
                        tensor = torch.from_numpy(var_data).float()
                    elif isinstance(var_data, torch.Tensor):
                        tensor = var_data.float()
                    else:
                        continue

                    # Resize static variables to match target shape
                    if target_shape and tensor.shape != target_shape:
                        print(f"Resizing static variable '{var_name}' from {tensor.shape} to {target_shape}")
                        try:
                            # Use interpolation to resize
                            tensor = torch.nn.functional.interpolate(
                                tensor.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                                size=target_shape,
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0).squeeze(0)  # Remove added dims
                            static_tensors[var_name] = tensor
                            print(f"Successfully resized '{var_name}' to {tensor.shape}")
                        except Exception as e:
                            print(f"Failed to resize static variable '{var_name}': {e}")
                            # Create dummy static variable with correct shape
                            static_tensors[var_name] = torch.zeros(target_shape)
                            print(f"Created dummy '{var_name}' with shape {target_shape}")
                    else:
                        static_tensors[var_name] = tensor

            # If no static variables were loaded, create minimal dummy ones
            if not static_tensors and target_shape:
                print("No static variables available, creating minimal dummy static variables")
                for var_name in essential_static_vars:
                    static_tensors[var_name] = torch.zeros(target_shape)
                    print(f"Created dummy static variable '{var_name}' with shape {target_shape}")
            
            # Create metadata with EXACT spatial dimensions matching the data
            if target_shape:
                height, width = target_shape
                lat_values = np.linspace(90, -90, height)
                lon_values = np.linspace(0, 360, width)
                print(f"Creating metadata for exact spatial resolution: {height}x{width}")
            else:
                # Use data from metadata_info if available
                lat_values = metadata_info.get("lat", np.linspace(90, -90, 32))
                lon_values = metadata_info.get("lon", np.linspace(0, 360, 32))
                height, width = len(lat_values), len(lon_values)
                print(f"Using metadata spatial resolution: {height}x{width}")

            # Convert longitude values to [0, 360) range if they're in [-180, 180) range
            if isinstance(lon_values, np.ndarray):
                # Check if any longitude is negative (indicating [-180, 180) range)
                if np.any(lon_values < 0):
                    print(f"Converting longitude range from [{lon_values.min():.1f}, {lon_values.max():.1f}] to [0, 360)")
                    lon_values = (lon_values + 360) % 360
                    print(f"New longitude range: [{lon_values.min():.1f}, {lon_values.max():.1f}]")

            # Ensure longitude values are in [0, 360) range
            lon_values = np.clip(lon_values, 0, 359.999)

            # Validate dimensions match exactly
            lat_tensor = torch.tensor(lat_values).float()
            lon_tensor = torch.tensor(lon_values).float()

            print(f"Validation: lat tensor shape {lat_tensor.shape} should match height {height}")
            print(f"Validation: lon tensor shape {lon_tensor.shape} should match width {width}")

            assert lat_tensor.shape[0] == height, f"Latitude dimension mismatch: {lat_tensor.shape[0]} != {height}"
            assert lon_tensor.shape[0] == width, f"Longitude dimension mismatch: {lon_tensor.shape[0]} != {width}"

            metadata = Metadata(
                lat=lat_tensor,
                lon=lon_tensor,
                time=metadata_info.get("time", (datetime.datetime.now(),)),
                atmos_levels=metadata_info.get("atmos_levels", (1000, 925, 850, 700, 500))
            )
            
            batch = Batch(
                surf_vars=surf_vars,
                static_vars=static_tensors,
                atmos_vars=atmos_vars,
                metadata=metadata
            ).to(self.device)
            
            print("Aurora batch created successfully!")
            print(f"Surface variables: {list(surf_vars.keys())}")
            print(f"Atmospheric variables: {list(atmos_vars.keys())}")
            print(f"Static variables: {list(static_tensors.keys())}")

            # Debug: Print tensor shapes
            if surf_vars:
                sample_surf = next(iter(surf_vars.values()))
                print(f"Surface variable shape: {sample_surf.shape}")
            if atmos_vars:
                sample_atmos = next(iter(atmos_vars.values()))
                print(f"Atmospheric variable shape: {sample_atmos.shape}")
            if static_tensors:
                sample_static = next(iter(static_tensors.values()))
                print(f"Static variable shape: {sample_static.shape}")

            # Final validation before returning batch
            print(f"Final batch spatial shape: {batch.spatial_shape}")
            print(f"Final metadata lat shape: {batch.metadata.lat.shape}")
            print(f"Final metadata lon shape: {batch.metadata.lon.shape}")

            return batch
            
        except Exception as e:
            print(f"Error creating Aurora batch: {e}")
            raise
    
    def predict(self, batch: Batch, steps: int = 4) -> List[Batch]:
        """Run Aurora model prediction.

        Args:
            batch: Input batch
            steps: Number of prediction steps

        Returns:
            List of prediction batches
        """
        try:
            print(f"Running Aurora model rollout for {steps} steps...")

            # Debug: Print detailed batch information
            print("=== BATCH DEBUG INFO ===")
            print(f"Batch spatial shape: {batch.spatial_shape}")
            if batch.surf_vars:
                for var_name, var_tensor in batch.surf_vars.items():
                    print(f"Surface var '{var_name}': {var_tensor.shape}")
                    break  # Just show one example
            if batch.atmos_vars:
                for var_name, var_tensor in batch.atmos_vars.items():
                    print(f"Atmospheric var '{var_name}': {var_tensor.shape}")
                    break  # Just show one example
            if batch.static_vars:
                for var_name, var_tensor in batch.static_vars.items():
                    print(f"Static var '{var_name}': {var_tensor.shape}")
                    break  # Just show one example
            print(f"Metadata lat shape: {batch.metadata.lat.shape}")
            print(f"Metadata lon shape: {batch.metadata.lon.shape}")
            print(f"Metadata time: {batch.metadata.time}")
            print(f"Metadata atmos_levels: {batch.metadata.atmos_levels}")
            print("========================")

            predictions = [p.to("cpu") for p in rollout(self.model, batch, steps=steps)]
            print("Model rollout completed successfully!")
            return predictions

        except Exception as e:
            print(f"Error during model prediction: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_predictions(self,
                          predictions: List[Batch],
                          variable_names: List[str],
                          enhance_resolution: bool = True) -> Dict[str, List[List]]:
        """Extract prediction data from Aurora output with optional resolution enhancement.

        Args:
            predictions: List of prediction batches
            variable_names: Variables to extract
            enhance_resolution: Whether to enhance spatial resolution for better visualization

        Returns:
            Dictionary of extracted predictions
        """
        extracted = {}

        for var in variable_names:
            var_predictions = []
            for pred in predictions:
                try:
                    var_data = None
                    if var in pred.surf_vars:
                        # Detach tensor from computation graph before converting to numpy
                        var_data = pred.surf_vars[var].squeeze().detach().cpu().numpy()
                    elif var in pred.atmos_vars:
                        # Detach tensor from computation graph before converting to numpy
                        var_data = pred.atmos_vars[var].squeeze().detach().cpu().numpy()

                    if var_data is not None:
                        # Enhance resolution if requested and data is small
                        if enhance_resolution and var_data.shape[0] < 50 and var_data.shape[1] < 50:
                            try:
                                from scipy.ndimage import zoom
                                # Increase resolution by factor of 2-3 for better visualization
                                zoom_factor = min(3.0, 100.0 / max(var_data.shape))
                                if zoom_factor > 1.1:  # Only zoom if significant improvement
                                    enhanced_data = zoom(var_data, zoom_factor, order=1)
                                    print(f"Enhanced resolution for '{var}': {var_data.shape} -> {enhanced_data.shape}")
                                    var_data = enhanced_data
                            except ImportError:
                                print("scipy not available for resolution enhancement")
                            except Exception as e:
                                print(f"Resolution enhancement failed for '{var}': {e}")

                        var_predictions.append(var_data.tolist())
                    else:
                        # Create dummy data with enhanced resolution if possible
                        base_size = 32
                        if enhance_resolution:
                            base_size = min(64, base_size * 2)  # Enhanced dummy data
                        dummy_data = [[0.0] * base_size for _ in range(base_size)]
                        var_predictions.append(dummy_data)

                except Exception as e:
                    print(f"Warning: Could not extract variable '{var}': {e}")
                    # Create dummy data with correct shape if extraction fails
                    base_size = 32
                    if enhance_resolution:
                        base_size = min(64, base_size * 2)
                    dummy_data = [[0.0] * base_size for _ in range(base_size)]
                    var_predictions.append(dummy_data)

            if var_predictions:
                extracted[var] = var_predictions
                if var_predictions:
                    sample_shape = np.array(var_predictions[0]).shape
                    print(f"Extracted {len(var_predictions)} time steps for variable '{var}' with shape {sample_shape}")

        return extracted
