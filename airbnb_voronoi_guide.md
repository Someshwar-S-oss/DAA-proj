# Airbnb Geolocation Anomaly Detector
## Complete Implementation Guide using Voronoi Diagrams

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Project Structure](#project-structure)
4. [Implementation Steps](#implementation-steps)
5. [Code Implementation](#code-implementation)
6. [Analysis & Insights](#analysis--insights)
7. [Deployment & Extensions](#deployment--extensions)

---

## 🎯 Project Overview

### Objective
Build a geospatial anomaly detection system that identifies outlier Airbnb listings using Voronoi diagrams and geometric proximity analysis.

### Key Features
- Voronoi tessellation of listing locations
- Multi-metric anomaly scoring
- Interactive visualizations
- Automated outlier detection
- Business insights generation

### Expected Outcomes
- Visual map of listing distribution with Voronoi cells
- Ranked list of anomalous listings
- Statistical analysis of spatial patterns
- Actionable insights for market analysis

---

## 🛠 Prerequisites & Setup

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- Modern web browser (for interactive visualizations)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas numpy scipy matplotlib seaborn
pip install plotly geopandas shapely scikit-learn
pip install folium jupyter

# Optional but recommended
pip install ipywidgets  # For interactive notebooks
```

### Dataset Requirements

**Expected CSV Format:**
```
id, name, host_id, host_name, neighbourhood_group, neighbourhood, 
latitude, longitude, room_type, price, minimum_nights, 
number_of_reviews, last_review, reviews_per_month, 
calculated_host_listings_count, availability_365, 
number_of_reviews_ltm, license
```

**Data Sources:**
- Inside Airbnb: http://insideairbnb.com/get-the-data/
- Download listings.csv for your target city

---

## 📁 Project Structure

```
airbnb-anomaly-detector/
│
├── data/
│   ├── raw/
│   │   └── listings.csv
│   └── processed/
│       └── listings_cleaned.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── voronoi_generator.py
│   ├── anomaly_detector.py
│   └── visualizer.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_voronoi_analysis.ipynb
│   └── 03_anomaly_detection.ipynb
│
├── output/
│   ├── figures/
│   ├── reports/
│   └── maps/
│
├── requirements.txt
├── README.md
└── main.py
```

---

## 🚀 Implementation Steps

### Phase 1: Data Preparation (Week 1)

#### Step 1.1: Data Loading & Cleaning
```python
# Tasks:
- Load CSV file
- Handle missing values in lat/lon
- Remove invalid coordinates (0,0 or out of bounds)
- Filter extreme price outliers
- Validate data types
```

#### Step 1.2: Exploratory Data Analysis
```python
# Analyze:
- Distribution of listings across neighborhoods
- Price ranges and statistics
- Geographic bounds of the city
- Review patterns
- Room type distribution
```

### Phase 2: Voronoi Implementation (Week 1-2)

#### Step 2.1: Basic Voronoi Diagram
```python
# Implement:
- Extract (lat, lon) coordinates
- Generate Voronoi diagram using scipy
- Handle infinite regions at boundaries
- Clip Voronoi cells to city bounds
```

#### Step 2.2: Geometric Metrics
```python
# Calculate:
- Cell area for each listing
- Cell perimeter
- Distance to nearest neighbors (k=3, 5, 10)
- Local density (listings per km²)
- Perimeter-to-area ratio
```

### Phase 3: Anomaly Detection (Week 2)

#### Step 3.1: Single-Metric Anomalies
```python
# Identify outliers using:
- Z-score (threshold: |z| > 3)
- IQR method (1.5 * IQR)
- Percentile-based (top/bottom 5%)
```

#### Step 3.2: Multi-Metric Anomaly Score
```python
# Combine metrics:
- Spatial isolation score
- Price anomaly score
- Review frequency deviation
- Availability pattern score
- Weighted composite score
```

#### Step 3.3: Classification
```python
# Categorize anomalies:
- Isolated listings (large Voronoi cells)
- Price anomalies in sparse areas
- Suspicious patterns (isolated + no reviews)
- Market gaps (underserved areas)
```

### Phase 4: Visualization (Week 2-3)

#### Step 4.1: Static Visualizations
```python
# Create:
- Voronoi diagram with cell coloring
- Scatter plots with anomaly highlights
- Heatmaps of listing density
- Statistical distribution plots
```

#### Step 4.2: Interactive Maps
```python
# Build:
- Folium/Plotly interactive maps
- Voronoi overlay on city map
- Clickable markers with listing details
- Filter controls (price, room type, etc.)
```

### Phase 5: Analysis & Reporting (Week 3)

#### Step 5.1: Statistical Analysis
```python
# Generate:
- Summary statistics of anomalies
- Correlation analysis
- Neighborhood-wise breakdown
- Temporal patterns
```

#### Step 5.2: Business Insights
```python
# Produce:
- Top 20 anomalous listings report
- Market gap identification
- Pricing strategy recommendations
- Risk assessment for suspicious listings
```

---

## 💻 Code Implementation

### 1. data_loader.py

```python
import pandas as pd
import numpy as np

class AirbnbDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        
    def load_data(self):
        """Load CSV file"""
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df)} listings")
        return self.df
    
    def clean_data(self):
        """Clean and validate data"""
        # Remove missing coordinates
        self.df = self.df.dropna(subset=['latitude', 'longitude'])
        
        # Remove invalid coordinates
        self.df = self.df[
            (self.df['latitude'] != 0) & 
            (self.df['longitude'] != 0)
        ]
        
        # Handle price outliers (optional)
        q1 = self.df['price'].quantile(0.01)
        q99 = self.df['price'].quantile(0.99)
        self.df = self.df[
            (self.df['price'] >= q1) & 
            (self.df['price'] <= q99)
        ]
        
        print(f"Cleaned data: {len(self.df)} listings")
        return self.df
    
    def get_coordinates(self):
        """Extract coordinate arrays"""
        points = self.df[['longitude', 'latitude']].values
        return points
    
    def get_summary(self):
        """Generate data summary"""
        summary = {
            'total_listings': len(self.df),
            'price_range': (self.df['price'].min(), self.df['price'].max()),
            'avg_price': self.df['price'].mean(),
            'neighborhoods': self.df['neighbourhood'].nunique(),
            'room_types': self.df['room_type'].value_counts().to_dict()
        }
        return summary
```

### 2. voronoi_generator.py

```python
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

class VoronoiAnalyzer:
    def __init__(self, points):
        self.points = points
        self.vor = None
        self.cell_areas = None
        
    def generate_voronoi(self):
        """Generate Voronoi diagram"""
        self.vor = Voronoi(self.points)
        return self.vor
    
    def calculate_cell_areas(self, bound_box=None):
        """Calculate area of each Voronoi cell"""
        areas = []
        
        for point_idx in range(len(self.points)):
            region_idx = self.vor.point_region[point_idx]
            region = self.vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                # Infinite region - assign large value
                areas.append(np.inf)
                continue
            
            # Get vertices of the region
            vertices = [self.vor.vertices[i] for i in region]
            polygon = Polygon(vertices)
            
            # Clip to bounding box if provided
            if bound_box:
                bbox = Polygon(bound_box)
                polygon = polygon.intersection(bbox)
            
            areas.append(polygon.area)
        
        self.cell_areas = np.array(areas)
        return self.cell_areas
    
    def calculate_nearest_neighbors(self, k=5):
        """Calculate distance to k nearest neighbors"""
        from scipy.spatial.distance import cdist
        
        distances = cdist(self.points, self.points)
        np.fill_diagonal(distances, np.inf)
        
        # Get k nearest for each point
        nearest_k = np.sort(distances, axis=1)[:, :k]
        avg_distance = nearest_k.mean(axis=1)
        
        return avg_distance, nearest_k
    
    def calculate_density(self, radius=0.01):
        """Calculate local density around each point"""
        from scipy.spatial.distance import cdist
        
        distances = cdist(self.points, self.points)
        within_radius = (distances < radius).sum(axis=1) - 1  # Exclude self
        
        return within_radius
```

### 3. anomaly_detector.py

```python
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, data, metrics):
        self.data = data
        self.metrics = metrics
        self.anomaly_scores = None
        
    def zscore_method(self, values, threshold=3):
        """Detect anomalies using Z-score"""
        z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
        anomalies = z_scores > threshold
        return anomalies, z_scores
    
    def iqr_method(self, values):
        """Detect anomalies using IQR"""
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomalies = (values < lower_bound) | (values > upper_bound)
        return anomalies
    
    def percentile_method(self, values, percentile=95):
        """Detect top percentile as anomalies"""
        threshold = np.percentile(values, percentile)
        anomalies = values > threshold
        return anomalies
    
    def isolation_forest_method(self, features, contamination=0.1):
        """Use Isolation Forest for multi-dimensional anomaly detection"""
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        
        predictions = iso_forest.fit_predict(features)
        anomalies = predictions == -1
        scores = iso_forest.score_samples(features)
        
        return anomalies, scores
    
    def composite_score(self, weights=None):
        """Calculate weighted composite anomaly score"""
        if weights is None:
            weights = {
                'cell_area': 0.4,
                'nearest_neighbor': 0.3,
                'density': 0.2,
                'price': 0.1
            }
        
        # Normalize each metric to 0-1
        normalized = {}
        for key, values in self.metrics.items():
            valid_values = values[~np.isinf(values)]
            min_val = valid_values.min()
            max_val = valid_values.max()
            normalized[key] = (values - min_val) / (max_val - min_val)
        
        # Calculate weighted sum
        score = sum(normalized[key] * weights[key] 
                   for key in weights.keys())
        
        self.anomaly_scores = score
        return score
    
    def classify_anomalies(self, threshold=0.8):
        """Classify listings into anomaly categories"""
        categories = []
        
        for idx, score in enumerate(self.anomaly_scores):
            if score > threshold:
                # Determine specific category
                if self.metrics['cell_area'][idx] > np.percentile(
                    self.metrics['cell_area'], 90
                ):
                    categories.append('Isolated')
                elif self.metrics['price'][idx] > np.percentile(
                    self.metrics['price'], 90
                ):
                    categories.append('Price Anomaly')
                else:
                    categories.append('Multi-factor Anomaly')
            else:
                categories.append('Normal')
        
        return categories
```

### 4. visualizer.py

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import voronoi_plot_2d

class VoronoiVisualizer:
    def __init__(self, vor, data, metrics):
        self.vor = vor
        self.data = data
        self.metrics = metrics
        
    def plot_basic_voronoi(self, figsize=(15, 12)):
        """Plot basic Voronoi diagram"""
        fig, ax = plt.subplots(figsize=figsize)
        voronoi_plot_2d(self.vor, ax=ax, show_vertices=False)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Airbnb Listings - Voronoi Diagram')
        plt.tight_layout()
        return fig
    
    def plot_colored_voronoi(self, color_metric='cell_area', 
                            figsize=(15, 12), cmap='YlOrRd'):
        """Plot Voronoi diagram colored by metric"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot Voronoi edges
        for simplex in self.vor.ridge_vertices:
            if -1 not in simplex:
                plt.plot(self.vor.vertices[simplex, 0], 
                        self.vor.vertices[simplex, 1], 'k-', alpha=0.3)
        
        # Scatter plot with color mapping
        scatter = ax.scatter(
            self.data['longitude'],
            self.data['latitude'],
            c=self.metrics[color_metric],
            cmap=cmap,
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        plt.colorbar(scatter, label=color_metric)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Voronoi Diagram - Colored by {color_metric}')
        plt.tight_layout()
        return fig
    
    def plot_anomalies(self, anomaly_labels, figsize=(15, 12)):
        """Highlight anomalous listings"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normal points
        normal = anomaly_labels == 'Normal'
        ax.scatter(
            self.data.loc[normal, 'longitude'],
            self.data.loc[normal, 'latitude'],
            c='lightblue',
            s=30,
            alpha=0.5,
            label='Normal'
        )
        
        # Anomalies
        anomaly = ~normal
        ax.scatter(
            self.data.loc[anomaly, 'longitude'],
            self.data.loc[anomaly, 'latitude'],
            c='red',
            s=100,
            alpha=0.8,
            marker='*',
            edgecolors='black',
            linewidth=1,
            label='Anomaly'
        )
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Anomalous Listings Detection')
        ax.legend()
        plt.tight_layout()
        return fig
    
    def plot_interactive_map(self, anomaly_scores):
        """Create interactive Plotly map"""
        fig = go.Figure()
        
        # Add all points
        fig.add_trace(go.Scattermapbox(
            lon=self.data['longitude'],
            lat=self.data['latitude'],
            mode='markers',
            marker=dict(
                size=8,
                color=anomaly_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Anomaly Score")
            ),
            text=[
                f"Name: {name}<br>Price: ${price}<br>Room: {room}" 
                for name, price, room in zip(
                    self.data['name'],
                    self.data['price'],
                    self.data['room_type']
                )
            ],
            hoverinfo='text'
        ))
        
        # Set map layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lon=self.data['longitude'].mean(),
                    lat=self.data['latitude'].mean()
                ),
                zoom=11
            ),
            title="Interactive Anomaly Detection Map",
            height=800
        )
        
        return fig
    
    def plot_metric_distributions(self, figsize=(15, 10)):
        """Plot distribution of all metrics"""
        metrics_to_plot = [k for k in self.metrics.keys() 
                          if k != 'cell_area']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot[:4]):
            axes[idx].hist(self.metrics[metric], bins=50, 
                          edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(metric)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {metric}')
        
        plt.tight_layout()
        return fig
```

### 5. main.py (Complete Pipeline)

```python
import pandas as pd
import numpy as np
from src.data_loader import AirbnbDataLoader
from src.voronoi_generator import VoronoiAnalyzer
from src.anomaly_detector import AnomalyDetector
from src.visualizer import VoronoiVisualizer

def main():
    # Step 1: Load and clean data
    print("=== Loading Data ===")
    loader = AirbnbDataLoader('data/raw/listings.csv')
    df = loader.load_data()
    df = loader.clean_data()
    
    summary = loader.get_summary()
    print(f"Summary: {summary}")
    
    # Step 2: Generate Voronoi diagram
    print("\n=== Generating Voronoi Diagram ===")
    points = loader.get_coordinates()
    voronoi = VoronoiAnalyzer(points)
    vor = voronoi.generate_voronoi()
    
    # Step 3: Calculate metrics
    print("\n=== Calculating Metrics ===")
    cell_areas = voronoi.calculate_cell_areas()
    avg_nn_dist, _ = voronoi.calculate_nearest_neighbors(k=5)
    density = voronoi.calculate_density(radius=0.01)
    
    metrics = {
        'cell_area': cell_areas,
        'nearest_neighbor': avg_nn_dist,
        'density': density,
        'price': df['price'].values
    }
    
    # Step 4: Detect anomalies
    print("\n=== Detecting Anomalies ===")
    detector = AnomalyDetector(df, metrics)
    anomaly_scores = detector.composite_score()
    anomaly_categories = detector.classify_anomalies(threshold=0.8)
    
    # Add results to dataframe
    df['anomaly_score'] = anomaly_scores
    df['anomaly_category'] = anomaly_categories
    df['cell_area'] = cell_areas
    
    # Step 5: Generate visualizations
    print("\n=== Creating Visualizations ===")
    viz = VoronoiVisualizer(vor, df, metrics)
    
    # Save plots
    fig1 = viz.plot_basic_voronoi()
    fig1.savefig('output/figures/voronoi_basic.png', dpi=300)
    
    fig2 = viz.plot_colored_voronoi(color_metric='cell_area')
    fig2.savefig('output/figures/voronoi_colored.png', dpi=300)
    
    fig3 = viz.plot_anomalies(df['anomaly_category'])
    fig3.savefig('output/figures/anomalies_highlighted.png', dpi=300)
    
    fig4 = viz.plot_metric_distributions()
    fig4.savefig('output/figures/metric_distributions.png', dpi=300)
    
    # Interactive map
    interactive_fig = viz.plot_interactive_map(anomaly_scores)
    interactive_fig.write_html('output/maps/interactive_anomaly_map.html')
    
    # Step 6: Generate report
    print("\n=== Generating Report ===")
    anomalies = df[df['anomaly_category'] != 'Normal']
    
    report = f"""
    AIRBNB ANOMALY DETECTION REPORT
    ================================
    
    Total Listings Analyzed: {len(df)}
    Anomalies Detected: {len(anomalies)} ({len(anomalies)/len(df)*100:.2f}%)
    
    Top 10 Most Anomalous Listings:
    {anomalies.nlargest(10, 'anomaly_score')[
        ['name', 'neighbourhood', 'price', 'anomaly_score', 'cell_area']
    ].to_string()}
    
    Anomaly Categories:
    {df['anomaly_category'].value_counts().to_string()}
    
    Key Statistics:
    - Average cell area: {cell_areas[~np.isinf(cell_areas)].mean():.6f}
    - Max cell area: {cell_areas[~np.isinf(cell_areas)].max():.6f}
    - Average NN distance: {avg_nn_dist.mean():.6f}
    """
    
    with open('output/reports/anomaly_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    
    # Save processed data
    df.to_csv('data/processed/listings_with_anomalies.csv', index=False)
    
    print("\n=== Analysis Complete ===")
    print("Check output/ folder for results")

if __name__ == "__main__":
    main()
```

---

## 📊 Analysis & Insights

### Key Questions to Answer

1. **Geographic Distribution**
   - Where are listings concentrated?
   - Which areas are underserved?
   - Are there isolated listings far from clusters?

2. **Price vs. Location**
   - Do isolated listings charge premium prices?
   - Are there pricing anomalies in sparse areas?
   - Does distance from city center correlate with price?

3. **Market Opportunities**
   - Which neighborhoods have low listing density?
   - Where could new listings be successful?
   - Are there geographic gaps in coverage?

4. **Suspicious Patterns**
   - Listings with unusual combinations (isolated + expensive + no reviews)
   - Potential fake listings or data errors
   - Hosts with scattered listings in odd locations

### Interpretation Guide

**Large Voronoi Cells (Isolated)**
- May indicate underserved areas
- Could be vacation properties outside city center
- Potential data quality issues

**Small Cells (Dense Clusters)**
- High competition areas
- Tourist hotspots
- City center concentrations

**Price Anomalies in Sparse Areas**
- Premium unique properties
- Overpriced listings with low demand
- Luxury or specialty accommodations

---

## 🚀 Deployment & Extensions

### Extensions to Consider

1. **Temporal Analysis**
   - Track how Voronoi patterns change over time
   - Seasonal variation in listing distribution
   - Market evolution analysis

2. **Multi-city Comparison**
   - Compare spatial patterns across cities
   - Identify universal vs. city-specific patterns
   - Benchmark analysis

3. **Predictive Modeling**
   - Predict listing success based on location
   - Recommend optimal prices for new listings
   - Forecast market saturation

4. **3D Voronoi**
   - Add price as third dimension
   - Volumetric analysis of market segments

5. **Web Dashboard**
   - Flask/Streamlit interactive dashboard
   - Real-time anomaly detection
   - User-driven exploration tools

### Deployment Options

**Local Analysis**
```bash
python main.py
```

**Jupyter Notebook**
- Better for exploratory analysis
- Interactive visualizations
- Step-by-step execution

**Web Application**
```python
# Using Streamlit
streamlit run app.py
```

**Automated Reports**
```bash
# Schedule with cron (Linux/Mac)
0 0 * * 0 /path/to/venv/bin/python /path/to/main.py
```

---

## 📝 Testing & Validation

### Unit Tests

```python
# test_voronoi.py
import pytest
from src.voronoi_generator import VoronoiAnalyzer

def test_voronoi_generation():
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    analyzer = VoronoiAnalyzer(points)
    vor = analyzer.generate_voronoi()
    assert vor is not None
    assert len(vor.points) == 4

def test_area_calculation():
    # Test with known configuration
    pass
```

### Validation Strategies

1. **Visual Inspection**: Manual review of top anomalies
2. **Cross-validation**: Compare with domain knowledge
3. **Sensitivity Analysis**: Test parameter variations
4. **Statistical Tests**: Validate metric distributions

---

## 📚 Resources & References

### Documentation
- SciPy Voronoi: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html
- Shapely: https://shapely.readthedocs.io/
- Plotly: https://plotly.com/python/

### Academic References
- Fortune's Algorithm for Voronoi Diagrams
- Spatial Outlier Detection Methods
- Geographic Information Systems (GIS) techniques

### Datasets
- Inside Airbnb: http://insideairbnb.com/
- Sample cities: New York, Paris, London, Tokyo

---

## ✅ Project Checklist

- [ ] Environment setup complete
- [ ] Dataset downloaded and validated
- [ ] Data cleaning pipeline implemented
- [ ] Voronoi diagram generation working
- [ ] Geometric metrics calculated
- [ ] Anomaly detection algorithms implemented
- [ ] Basic visualizations created
- [ ] Interactive map generated
- [ ] Analysis report produced
- [ ] Code documented and tested
- [ ] Results validated
- [ ] Presentation/demo prepared

---

## 🎓 Learning Outcomes

By completing this project, you will:
- Understand Voronoi diagrams and their applications
- Implement geometric algorithms
- Apply multiple anomaly detection techniques
- Create interactive geospatial visualizations
- Conduct real-world data analysis
- Generate actionable business insights

---

**Good luck with your implementation!**