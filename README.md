# Aurora Wave Prediction

A production-ready web application for oceanic wave prediction using Microsoft's Aurora model with real-time ECMWF data.

## Features

- **Real-time Wave Prediction**: Generate wave forecasts using the Aurora model
- **Interactive Web Interface**: User-friendly Streamlit frontend
- **ECMWF Data Integration**: Fetch real-time atmospheric and wave data
- **Flexible Geographic Selection**: Choose custom regions for prediction
- **Multiple Time Steps**: Generate multi-step forecasts
- **Memory-Optimized Processing**: Efficient handling of large datasets
- **Robust Data Pipeline**: 3-day wave data strategy with batched processing

## Architecture

```
Frontend (Streamlit) ↔ Backend (Flask) ↔ Aurora Model
                                ↓
                        ECMWF Data Sources
                        (Surface, Atmospheric, Wave)
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Aurora-Wave-Prediction
   ```

2. **Set up Backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Frontend**
   ```bash
   cd ../frontend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Backend**
   ```bash
   cd backend
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   python backend_app.py
   ```
2. **Start the Frontend** (in a new terminal)
   ```bash
   cd frontend
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   streamlit run streamlit_app.py
   ```

3. **Access the Application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:5000

## API Endpoints

### Download Data
```http
POST /api/download-data
Content-Type: application/json

{
  "target_date": "2024-01-15",
  "lat_bounds": [2.2, 4.2],
  "lon_bounds": [72.2, 74.2]
}
```

### Generate Predictions
```http
POST /api/predict_oceanic
Content-Type: application/json

{
  "target_date": "2024-01-15",
  "lat_bounds": [2.2, 4.2],
  "lon_bounds": [72.2, 74.2],
  "steps": 5
}
```

## Data Sources

### ECMWF OpenData (Primary)
- **Surface Variables**: 2m temperature, 10m wind, mean sea level pressure
- **Atmospheric Variables**: Temperature, wind (5 pressure levels: 1000, 925, 850, 700, 500 hPa)
- **Wave Variables**: Significant wave height, wave direction, wave period, peak period
- **Resolution**: 0.25° global grid
- **Update Frequency**: 4 times daily (00z, 06z, 12z, 18z)
- **Data Strategy**: 3-day coverage with optimal forecast time selection

### Wave Data Processing
- **Batched Processing**: Handles large GRIB files (12MB each) in memory-efficient batches
- **Forecast Step Averaging**: Combines multiple forecast steps for robust predictions
- **Automatic Fallback**: Synthetic data generation when ECMWF is unavailable
- **File Caching**: Reuses downloaded data to prevent duplicate downloads

## Model Details

### Aurora Wave Model
- **Type**: Transformer-based weather prediction model
- **Input**: Multi-variable atmospheric and oceanic data
- **Output**: Wave forecasts (height, direction, period)
- **Resolution**: Configurable (default: 32x32, enhanced to 96x96)
- **Forecast Steps**: 1-10 steps (6-hour intervals)
- **Variables**: 22 surface variables, 3 atmospheric variables, 5 static variables

### Technical Improvements
- **Memory Management**: Batched processing prevents allocation errors
- **Tensor Shape Handling**: Proper atmospheric level preservation (5 levels)
- **Data Type Detection**: Automatic distinction between wave and atmospheric data
- **Spatial Resolution**: Automatic resizing and interpolation

## Configuration

Key configuration options in `backend/config.py`:

- **ECMWF Configuration**: Variables, pressure levels, resolution
- **Wave Data Configuration**: Variables, mapping, processing options
- **Aurora Variables**: Required and optional variables for model
- **Processing Configuration**: Batch size, time steps, spatial resolution

## Development

### Project Structure
```
Aurora-Wave-Prediction/
├── backend/
│   ├── aurora_data_sources/     # Data source implementations
│   │   ├── base.py             # Base data source class
│   │   ├── ecmwf_source.py     # ECMWF atmospheric/surface data
│   │   ├── ecmwf_wave_source.py # ECMWF wave data with batching
│   │   ├── huggingface_source.py # Static variables from HuggingFace
│   │   └── __init__.py         # Package initialization
│   ├── aurora_interface.py      # Aurora model interface
│   ├── backend_app.py          # Flask API server
│   ├── config.py               # Configuration settings
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── streamlit_app.py        # Streamlit web interface
│   └── requirements.txt        # Python dependencies
├── downloads/                   # Data cache directory
└── README.md
```

### Key Dependencies
- **ecmwf-opendata**: ECMWF data access
- **cfgrib**: GRIB file processing
- **xarray**: Multi-dimensional data handling
- **torch**: PyTorch for Aurora model
- **aurora**: Microsoft Aurora model package

## Performance Optimizations

### Memory Management
- **Batched Processing**: Process GRIB files in groups of 2 to prevent memory overflow
- **Immediate Reduction**: Reduce dataset memory usage during loading
- **Data Type Optimization**: Use float32 instead of float64
- **Automatic Cleanup**: Close datasets after processing

### Data Processing
- **Smart Caching**: Reuse downloaded files within 6-hour window
- **Forecast Step Averaging**: Combine multiple forecast steps for wave data
- **Pressure Level Preservation**: Maintain 5 atmospheric levels for model compatibility
- **Spatial Interpolation**: Efficient resizing using scipy interpolation

## Troubleshooting

### Common Issues

1. **Memory Allocation Errors**
   - **Fixed**: Batched processing handles large datasets
   - Use smaller geographic regions for very large areas
   - Monitor system memory usage

2. **Tensor Shape Mismatches**
   - **Fixed**: Proper data type detection and processing
   - Atmospheric data preserves 5 pressure levels
   - Wave data averages forecast steps correctly

3. **Data Download Failures**
   - Check internet connectivity
   - Verify ECMWF service status
   - System falls back to synthetic data automatically

4. **Model Loading Errors**
   - Ensure sufficient disk space for model checkpoints
   - Check HuggingFace connectivity
   - Verify CUDA availability for GPU acceleration

### Performance Tips
- Use smaller geographic regions for faster processing
- Enable GPU acceleration when available
- Monitor download cache to prevent disk space issues
- Use recent data (within 3 days) for best ECMWF availability

## Production Deployment

The system is production-ready with:
- ✅ Memory-efficient data processing
- ✅ Robust error handling and fallbacks
- ✅ Efficient data caching and reuse
- ✅ Real ECMWF data integration
- ✅ Scalable architecture

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft Research for the Aurora model
- ECMWF for weather and wave data access
- Streamlit and Flask communities
- Contributors to ecmwf-opendata and cfgrib packages
- GPU acceleration recommended for faster predictions

## Troubleshooting

**Backend won't start**: Check Python version (3.8+) and install requirements
**Frontend connection error**: Ensure backend is running on port 5000
**Slow predictions**: Consider using GPU or reducing forecast steps
**Memory issues**: Close other applications, use smaller prediction areas

## License

Uses Microsoft's Aurora model under their license terms.
