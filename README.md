# Aurora Wave Prediction Application

A streamlined web application for oceanic wave predictions using Microsoft's Aurora model. Provides interactive maps showing wave conditions around island archipelagos worldwide.

## Features

- **Interactive Maps**: Colored prediction points showing wave conditions
- **20+ Island Locations**: Pre-configured archipelagos from Maldives to Caribbean
- **Real-time Predictions**: 6-hour forecast intervals up to 60 hours ahead
- **Ocean Variables**: Wave height, direction, period, and wind components
- **Land Masking**: Ocean-only predictions with automatic land filtering

## Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU optional but recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Aurora_Streamlit.git
cd Aurora_Streamlit
```

2. **Setup Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Setup Frontend**
```bash
cd ../frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application

1. **Start Backend** (Terminal 1)
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python backend_app.py
```
Backend runs on http://localhost:5000

2. **Start Frontend** (Terminal 2)
```bash
cd frontend
source venv/bin/activate  # On Windows: venv\Scripts\activate
streamlit run streamlit_app.py
```
Frontend runs on http://localhost:8501

### Usage

1. Open http://localhost:8501 in your browser
2. Select an island archipelago from the sidebar
3. Choose prediction date and forecast steps
4. Click "Download Data" to cache training data
5. Click "Run Prediction" to generate wave forecasts
6. Explore interactive maps with colored prediction points

## Web UI preview
[![Watch the video](https://github.com/AbhishekSai1551/Aurora-Wave-Prediction/main/assets/Thumbnail1.png)](https://github.com/AbhishekSai1551/Aurora-Wave-Prediction/main/assets/Preview1.mp4)


## Project Structure

```
Aurora_Streamlit/
├── backend/
│   ├── aurora_data_sources/    # Data source modules
│   ├── aurora_interface.py     # Aurora model interface
│   ├── backend_app.py         # Flask API server
│   ├── config.py              # Configuration
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py       # Streamlit web app
│   └── requirements.txt
└── README.md
```

## API Endpoints

- `GET /api/health` - Check backend status
- `POST /api/download-data` - Download training data
- `POST /api/predict_oceanic` - Generate wave predictions

## Predicted Variables

- **SWH**: Significant Wave Height (m)
- **MWD**: Mean Wave Direction (°)
- **MWP**: Mean Wave Period (s)
- **PP1D**: Peak Period (s)
- **10U/10V**: Surface Wind Components (m/s)

## Data Sources

- **ECMWF**: Atmospheric and surface variables
- **NOAA**: Ocean wave variables
- **HuggingFace**: Aurora model static variables

## Performance Notes

- First run: ~2-3 minutes (model download)
- Subsequent runs: ~30-60 seconds
- Data automatically cached to avoid re-downloads
- GPU acceleration recommended for faster predictions

## Troubleshooting

**Backend won't start**: Check Python version (3.8+) and install requirements
**Frontend connection error**: Ensure backend is running on port 5000
**Slow predictions**: Consider using GPU or reducing forecast steps
**Memory issues**: Close other applications, use smaller prediction areas

## License

Uses Microsoft's Aurora model under their license terms.
