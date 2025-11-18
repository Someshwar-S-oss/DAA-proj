# Airbnb Voronoi Anomaly Detector

A geospatial anomaly detection system that identifies outlier Airbnb listings using Voronoi diagrams and geometric proximity analysis.

## 🎯 Overview

This project analyzes Airbnb listings using computational geometry to detect spatial anomalies. It generates Voronoi tessellations and calculates multiple spatial metrics to identify unusual listings based on their location, isolation, density, and price patterns.

## ✨ Features

### Core Analysis
- **Voronoi Diagram Generation**: Creates spatial tessellation of listing locations
- **Multi-Metric Anomaly Detection**: Combines spatial isolation, density, and price factors
- **Rich Visualizations**: Generates multiple plots and interactive maps with colored Voronoi regions
- **Comprehensive Reports**: Detailed analysis with top anomalous listings
- **Automated Pipeline**: End-to-end analysis in command-line or web interface

### Interactive Web Application 🌟
- **Drag & Drop CSV Upload**: No file path configuration needed
- **Interactive Dashboard**: 6 comprehensive tabs for analysis
- **Live Maps**: Both marker-based and Voronoi polygon overlay maps
- **Dynamic Filtering**: Filter listings by category, score, and price
- **Multiple Export Formats**: Download results as CSV, JSON, or Excel
- **Real-time Visualization**: All charts and maps generated on-demand

## 📋 Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Modern web browser (for interactive map viewing)

## 🚀 Quick Start

### Option 1: Web Application (Recommended) 🌐

**Best for**: Interactive exploration, drag-and-drop convenience, and real-time visualization

1. **Install Anaconda** (if not already installed)

2. **Set up environment**:
   ```bash
   # Open Anaconda Prompt
   cd C:\Users\eshwa\Repos\DAA-proj
   
   # Create conda environment
   conda create -n daa-proj 
   conda activate daa-proj
   
   # Install dependencies
   conda install numpy scipy pandas scikit-learn matplotlib seaborn -y
   pip install shapely folium tqdm streamlit streamlit-folium openpyxl
   ```

3. **Launch the app**:
   ```bash
   streamlit run app.py
   ```

4. **Use the app**:
   - Browser opens automatically at `http://localhost:8501`
   - Upload your CSV file via drag-and-drop
   - Click "🚀 Run Analysis"
   - Explore results across 6 interactive tabs!

### Option 2: Command-Line Interface 💻

**Best for**: Batch processing, automation, and scripting

1. **Install dependencies** (same as above, or use venv):
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Place your CSV file** in the project directory as `listings (1).csv`

3. **Run the analysis**:
   ```powershell
   python main.py
   ```

4. **Interactive menu options**:
   - **1-7**: Step-by-step analysis
   - **8**: Run complete analysis
   - **9**: Export results

5. **View the results**:
   - **Figures**: `output/figures/*.png`
   - **Report**: `output/reports/anomaly_report.txt`
   - **Maps**: `output/maps/*.html`
   - **Data**: `output/reports/listings_with_anomalies.csv`

## 📊 Required CSV Format

Your CSV file must contain these columns:
- `latitude` ✅ Required
- `longitude` ✅ Required
- `price` ✅ Required
- `name` (optional)
- `neighbourhood` (optional)
- `room_type` (optional)

## 📁 Project Structure

```
airbnb-voronoi-anomaly-detector/
│
├── src/
│   ├── __init__.py               # Package initialization
│   ├── data_loader.py            # Data loading and cleaning
│   ├── voronoi_generator.py      # Voronoi diagram generation
│   ├── anomaly_detector.py       # Anomaly detection algorithms
│   └── visualizer.py             # Visualization creation
│
├── output/
│   ├── figures/                  # Generated plots (PNG)
│   ├── reports/                  # Analysis reports (TXT, CSV)
│   └── maps/                     # Interactive maps (HTML)
│
├── app.py                        # 🌟 Streamlit web application
├── main.py                       # Command-line interface (interactive menu)
├── requirements.txt              # Python dependencies
├── README.md                     # This file (you're reading it!)
├── STREAMLIT_README.md          # Detailed Streamlit app guide
├── airbnb_voronoi_guide.md      # Detailed implementation guide
└── listings (1).csv             # Your data file (example)
```

## 🔬 How It Works

### 1. Data Processing
- Loads Airbnb listings from CSV
- Cleans data (removes invalid coordinates, handles outliers)
- Extracts geographic coordinates

### 2. Voronoi Analysis
- Generates Voronoi diagram from listing coordinates
- Calculates cell areas (isolation measure)
- Computes nearest neighbor distances
- Measures local density

### 3. Anomaly Detection
- Applies multiple detection methods (Z-score, IQR, Isolation Forest)
- Calculates composite anomaly scores
- Classifies anomalies into categories:
  - **Isolated**: Large Voronoi cells (far from other listings)
  - **Price Anomaly**: Unusual pricing patterns
  - **Low Density**: In sparse areas
  - **Multi-factor Anomaly**: Multiple suspicious factors

### 4. Visualization
- Basic Voronoi diagram
- Colored by cell area and price
- Anomaly highlights
- Metric distributions
- Price vs isolation scatter plot
- Interactive HTML map with markers

### 5. Reporting
- Summary statistics
- Top 20 most anomalous listings
- Category breakdown
- Key insights and recommendations

## 🎨 Streamlit App Features

### Tab 1: 📊 Overview
- Dataset statistics and key metrics
- Anomaly distribution bar charts
- Price distribution histograms
- Summary tables

### Tab 2: 🗺️ Maps
- **Marker Map**: Click markers for listing details
- **Voronoi Overlay Map**: Colored polygons showing spatial regions
  - Color by: Category, Anomaly Score, or Price
  - Interactive tooltips on click

### Tab 3: 📈 Visualizations
- All 6 analysis plots in dropdown menu
- High-resolution matplotlib figures
- Proper geographic scale (4 decimal precision)

### Tab 4: 🔍 Anomaly Details
- **Interactive Filters**:
  - Filter by anomaly category
  - Slider for minimum anomaly score
  - Price range slider
- Top 20 most anomalous listings table
- Sortable data grid

### Tab 5: 📄 Report
- Executive summary with key metrics
- Category breakdown statistics
- Key insights and recommendations
- Spatial and price analysis

### Tab 6: 💾 Download
- Export as CSV, JSON, or Excel
- Download all listings or anomalies only
- Multi-sheet Excel with separate tabs
- Data preview in browser

## 📊 Output Files (CLI Mode)

### Visualizations (`output/figures/`)
1. `01_voronoi_basic.png` - Basic Voronoi diagram
2. `02_voronoi_cell_area.png` - Colored by cell area
3. `03_voronoi_price.png` - Colored by price
4. `04_anomalies_highlighted.png` - Anomalies marked with stars
5. `05_metric_distributions.png` - Histograms of all metrics
6. `06_price_vs_isolation.png` - Price vs isolation analysis

### Interactive Maps (`output/maps/`)
- `interactive_anomaly_map.html` - Marker-based map
- `voronoi_overlay_map.html` - Colored Voronoi regions overlay

### Reports (`output/reports/`)
- `anomaly_report.txt` - Comprehensive text report
- `listings_with_anomalies.csv` - Original data + anomaly scores

## 🎨 Customization

### Adjust Anomaly Detection Threshold

In `main.py`, modify the threshold parameter:
```python
anomaly_categories = detector.classify_anomalies(threshold=0.7)  # Default: 0.7
```

### Change Metric Weights

Adjust the weights in the composite score calculation:
```python
anomaly_scores = detector.composite_score(weights={
    'cell_area': 0.4,         # Spatial isolation
    'nearest_neighbor': 0.3,  # Distance to neighbors
    'density': 0.2,           # Local density
    'price': 0.1              # Price factor
})
```

### Modify Density Radius

Change the radius for density calculation:
```python
density = voronoi.calculate_density(radius=0.01)  # In coordinate units
```

## 🐛 Troubleshooting

### Streamlit App Issues

**App won't start**
```bash
# Make sure streamlit is installed
pip install streamlit streamlit-folium

# Try updating
pip install --upgrade streamlit

# Run from project root
cd C:\Users\eshwa\Repos\DAA-proj
streamlit run app.py
```

**"ModuleNotFoundError"**
- Install missing packages: `pip install streamlit streamlit-folium openpyxl`
- Activate your conda environment: `conda activate daa-proj`

**Maps not displaying**
- Check browser console for errors
- Try different browser (Chrome, Firefox recommended)
- Ensure folium is installed: `pip install folium`

**SSL certificate errors (MSYS2 Python)**
- This project requires Anaconda or official Python
- MSYS2 Python has known SSL/compilation issues
- Solution: Install Anaconda and use conda environment (see Quick Start)

### CLI Issues

**"Data file not found"**
- Ensure CSV is in the project root directory
- Or modify the `data_file` variable in `main.py`

**Import errors**
- Install dependencies: `pip install -r requirements.txt` (or use conda)
- Activate your environment if using one

**"No module named 'src'"**
- Run script from the project root directory
- The script automatically adds `src/` to the Python path

**Memory issues with large datasets**
- Reduce dataset size or increase available RAM
- For Streamlit: Limited to 500 Voronoi cells for performance
- For CLI: Process smaller subsets of data

## 📖 Additional Documentation

- **`STREAMLIT_README.md`**: Detailed Streamlit app guide
  - Installation steps
  - Feature walkthrough
  - Usage tips
  - Performance notes

- **`airbnb_voronoi_guide.md`**: Implementation guide
  - Theoretical background
  - Algorithm details
  - Extension ideas
  - Mathematical foundations

## 🎓 Learning Outcomes

By using this project, you'll understand:
- Voronoi diagrams and their applications in geospatial analysis
- Spatial anomaly detection techniques
- Multi-metric composite scoring systems
- Interactive data visualization with Streamlit
- Geospatial data processing and analysis
- Data visualization best practices

## 🆚 CLI vs Web App Comparison

| Feature | CLI (`main.py`) | Web App (`app.py`) |
|---------|-----------------|-------------------|
| **Interface** | Interactive menu | Browser-based GUI |
| **CSV Upload** | File path in code | Drag & drop |
| **Visualization** | Saved PNG files | Live in browser |
| **Maps** | Open HTML in browser | Embedded interactive maps |
| **Filtering** | Not available | Dynamic filters |
| **Export** | Auto-saved to output/ | Multi-format download |
| **Best For** | Automation, scripting | Exploration, presentation |
| **Speed** | Faster for large datasets | Slight overhead for UI |

## 💡 Tips & Best Practices

### For Web App:
- **Performance**: Large datasets (>1000 listings) take 30-60 seconds
- **Voronoi overlay**: Limited to 500 regions for browser performance
- **Fresh analysis**: Click "Run Analysis" each time to avoid cached data
- **Export early**: Download results before uploading a new file

### For CLI:
- **Batch processing**: Run multiple analyses with different parameters
- **Automation**: Integrate into data pipelines
- **Large datasets**: No performance limits on Voronoi cells
- **Custom weights**: Easily modify anomaly detection weights in code

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to fork, modify, and improve this project for your needs!

## 📧 Questions?

- Web App Guide: See `STREAMLIT_README.md`
- Implementation Details: See `airbnb_voronoi_guide.md`
- Issues: Check the Troubleshooting section above

---

**Happy Analyzing! 🎉🗺️**
