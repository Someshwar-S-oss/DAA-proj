# 🏠 Airbnb Voronoi Anomaly Detector - Streamlit App

## Quick Start

### 1. Install Dependencies (in Anaconda Prompt)

```bash
conda activate daa-proj
cd C:\Users\eshwa\Repos\DAA-proj
pip install streamlit streamlit-folium openpyxl
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Features

### 📊 Overview Tab
- Dataset statistics and metrics
- Anomaly distribution charts
- Price distribution visualization
- Summary statistics table

### 🗺️ Maps Tab
- **Marker Map**: Interactive map with clickable markers for each listing
- **Voronoi Overlay Map**: Colored polygons showing Voronoi regions
  - Color by: Anomaly Category, Anomaly Score, or Price

### 📈 Visualizations Tab
- Basic Voronoi Diagram
- Voronoi colored by Cell Area
- Voronoi colored by Price
- Anomaly Detection Map
- Metric Distributions
- Price vs Isolation Analysis

### 🔍 Anomaly Details Tab
- Filter listings by:
  - Anomaly category
  - Minimum anomaly score
  - Price range
- View top 20 most anomalous listings
- Interactive data table

### 📄 Report Tab
- Executive summary with key metrics
- Category breakdown
- Key insights and recommendations
- Spatial and price analysis

### 💾 Download Tab
- Download processed data as CSV, JSON, or Excel
- Download anomalies-only dataset
- Preview data in the browser

## CSV File Requirements

Your CSV file must contain these columns:
- `latitude` (required)
- `longitude` (required)
- `price` (required)
- `name` (optional)
- `neighbourhood` (optional)
- `room_type` (optional)

## Usage Flow

1. **Upload CSV**: Click "Browse files" in the sidebar
2. **Run Analysis**: Click the "🚀 Run Analysis" button
3. **Explore Results**: Navigate through the 6 tabs
4. **Download**: Export your results from the Download tab

## Tips

- Use the **Voronoi Overlay Map** to see spatial patterns clearly
- Filter in the **Anomaly Details** tab to focus on specific listings
- Download results in **Excel format** for further analysis
- The app processes everything automatically - no manual steps needed!

## Troubleshooting

If the app doesn't start:
```bash
# Make sure you're in the right environment
conda activate daa-proj

# Update streamlit
pip install --upgrade streamlit streamlit-folium

# Run again
streamlit run app.py
```

## Performance Notes

- Large datasets (>1000 listings) may take 30-60 seconds to process
- Voronoi overlay maps are limited to 500 regions for performance
- The app caches results in session - no need to re-upload for different views
