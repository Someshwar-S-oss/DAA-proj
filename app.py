"""
Streamlit Web Application for Airbnb Voronoi Anomaly Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import sys
import os
from io import BytesIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import AirbnbDataLoader
from src.voronoi_generator import VoronoiAnalyzer
from src.anomaly_detector import AnomalyDetector
from src.visualizer import VoronoiVisualizer

# Page configuration
st.set_page_config(
    page_title="Airbnb Voronoi Anomaly Detector",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5A5F;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF5A5F;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'uploaded_file_id' not in st.session_state:
    st.session_state.uploaded_file_id = None


def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily with unique name."""
    import time
    temp_path = f"temp_{int(time.time())}_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def clear_output_files():
    """Clear any previous output files to ensure fresh analysis."""
    import shutil
    output_dirs = ['output/figures', 'output/maps', 'output/reports']
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)


def run_analysis(file_path):
    """Run the complete Voronoi anomaly detection analysis."""
    
    # Clear previous output files
    clear_output_files()
    
    # Clear any previous session state
    keys_to_clear = ['loader', 'df', 'voronoi', 'vor', 'metrics', 'detector', 'viz']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    with st.spinner("Loading and cleaning data..."):
        loader = AirbnbDataLoader(file_path)
        df = loader.load_data()
        df = loader.clean_data()
        st.session_state.loader = loader
        st.session_state.df = df
    
    with st.spinner("Generating Voronoi diagram..."):
        points = loader.get_coordinates()
        voronoi = VoronoiAnalyzer(points)
        vor = voronoi.generate_voronoi()
        bbox = loader.get_bounding_box(padding=0.02)
        
        cell_areas = voronoi.calculate_cell_areas(bound_box=bbox)
        avg_nn_dist, nn_matrix = voronoi.calculate_nearest_neighbors(k=5)
        density = voronoi.calculate_density(radius=0.01)
        
        metrics = {
            'cell_area': cell_areas,
            'nearest_neighbor': avg_nn_dist,
            'density': density,
            'price': df['price'].values
        }
        
        st.session_state.voronoi = voronoi
        st.session_state.vor = vor
        st.session_state.metrics = metrics
    
    with st.spinner("Detecting anomalies..."):
        detector = AnomalyDetector(df, metrics)
        anomaly_scores = detector.composite_score(weights={
            'cell_area': 0.4,
            'nearest_neighbor': 0.3,
            'density': 0.2,
            'price': 0.1
        })
        anomaly_categories = detector.classify_anomalies(threshold=0.7)
        
        df['anomaly_score'] = anomaly_scores
        df['anomaly_category'] = anomaly_categories
        df['cell_area'] = cell_areas
        df['nearest_neighbor_dist'] = avg_nn_dist
        df['local_density'] = density
        
        st.session_state.detector = detector
        st.session_state.df = df
    
    with st.spinner("Creating visualizations..."):
        viz = VoronoiVisualizer(vor, df, metrics)
        st.session_state.viz = viz
    
    st.session_state.analysis_done = True
    st.success("✅ Analysis completed successfully!")


def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🏠 Airbnb Voronoi Anomaly Detector</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📤 Upload Data")
        uploaded_file = st.file_uploader(
            "Upload your Airbnb listings CSV",
            type=['csv'],
            help="CSV should contain: latitude, longitude, price, and other listing details",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Generate unique file ID
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            # Check if it's a new file
            if st.session_state.uploaded_file_id != current_file_id:
                st.session_state.uploaded_file_id = current_file_id
                st.session_state.analysis_done = False
            
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            if not st.session_state.analysis_done:
                st.warning("⚠️ Click 'Run Analysis' to process this file")
            else:
                st.info("✅ Analysis complete! Upload a new file to run again.")
            
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                # Reset analysis state
                st.session_state.analysis_done = False
                
                temp_path = save_uploaded_file(uploaded_file)
                try:
                    run_analysis(temp_path)
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                finally:
                    # Always clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
        This tool analyzes Airbnb listings using Voronoi diagrams to detect:
        - 🔴 Isolated listings
        - 🟠 Price anomalies
        - 🟣 Low-density areas
        - 🔵 Normal listings
        """)
        
        st.markdown("---")
        st.header("📊 Required Columns")
        st.code("""
- latitude
- longitude
- price
- name (optional)
- neighbourhood (optional)
- room_type (optional)
        """)
    
    # Main content
    if not st.session_state.analysis_done:
        st.info("👈 Upload a CSV file from the sidebar to get started!")
        
        # Show example
        st.subheader("📋 Expected CSV Format")
        example_df = pd.DataFrame({
            'name': ['Cozy Apartment', 'Beach House'],
            'latitude': [40.7128, 40.7580],
            'longitude': [-74.0060, -73.9855],
            'price': [120.0, 350.0],
            'neighbourhood': ['Manhattan', 'Brooklyn'],
            'room_type': ['Entire home/apt', 'Private room']
        })
        st.dataframe(example_df, use_container_width=True)
        
    else:
        df = st.session_state.df
        viz = st.session_state.viz
        detector = st.session_state.detector
        loader = st.session_state.loader
        
        # Create tabs
        tabs = st.tabs([
            "📊 Overview", 
            "🗺️ Maps", 
            "📈 Visualizations", 
            "🔍 Anomaly Details",
            "📄 Report",
            "💾 Download"
        ])
        
        # TAB 1: Overview
        with tabs[0]:
            st.header("📊 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Listings", f"{len(df):,}")
            with col2:
                num_anomalies = (df['anomaly_category'] != 'Normal').sum()
                st.metric("Anomalies Detected", f"{num_anomalies:,}",
                         delta=f"{num_anomalies/len(df)*100:.1f}%")
            with col3:
                st.metric("Average Price", f"${df['price'].mean():.2f}")
            with col4:
                st.metric("Median Price", f"${df['price'].median():.2f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Anomaly Distribution")
                category_counts = df['anomaly_category'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#90caf9', '#f44336', '#ff9800', '#9c27b0', '#b71c1c']
                category_counts.plot(kind='bar', ax=ax, color=colors[:len(category_counts)])
                ax.set_xlabel("Category", fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                ax.set_title("Listings by Anomaly Category", fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("Price Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df['price'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
                ax.set_xlabel("Price ($)", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title("Price Distribution", fontsize=14, fontweight='bold')
                ax.axvline(df['price'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f"Mean: ${df['price'].mean():.2f}")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("---")
            
            st.subheader("Dataset Summary Statistics")
            summary_df = pd.DataFrame({
                'Metric': ['Total Listings', 'Price Range', 'Avg Price', 'Median Price', 
                          'Anomalies', 'Anomaly Rate', 'Unique Neighborhoods'],
                'Value': [
                    f"{len(df):,}",
                    f"${df['price'].min():.2f} - ${df['price'].max():.2f}",
                    f"${df['price'].mean():.2f}",
                    f"${df['price'].median():.2f}",
                    f"{num_anomalies:,}",
                    f"{num_anomalies/len(df)*100:.2f}%",
                    f"{df['neighbourhood'].nunique() if 'neighbourhood' in df.columns else 'N/A'}"
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # TAB 2: Maps
        with tabs[1]:
            st.header("🗺️ Interactive Maps")
            
            map_type = st.radio(
                "Select map type:",
                ["Marker Map", "Voronoi Overlay Map"],
                horizontal=True
            )
            
            if map_type == "Marker Map":
                st.subheader("📍 Marker Map - Click markers for details")
                
                center_lat = df['latitude'].mean()
                center_lon = df['longitude'].mean()
                
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=12,
                    tiles='OpenStreetMap'
                )
                
                color_map = {
                    'Normal': 'blue',
                    'Isolated': 'red',
                    'Price Anomaly': 'orange',
                    'Low Density': 'purple',
                    'Multi-factor Anomaly': 'darkred'
                }
                
                for idx, row in df.iterrows():
                    category = row['anomaly_category']
                    score = row['anomaly_score']
                    
                    popup_text = f"""
                    <b>{row.get('name', 'Unknown')[:50]}</b><br>
                    Price: ${row.get('price', 'N/A')}<br>
                    Room Type: {row.get('room_type', 'N/A')}<br>
                    Neighbourhood: {row.get('neighbourhood', 'N/A')}<br>
                    Anomaly Score: {score:.3f}<br>
                    Category: {category}
                    """
                    
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(
                            color=color_map.get(category, 'gray'),
                            icon='info-sign' if category == 'Normal' else 'warning-sign'
                        )
                    ).add_to(m)
                
                folium_static(m, width=1200, height=600)
            
            else:  # Voronoi Overlay Map
                st.subheader("🔷 Voronoi Regions - Click regions for details")
                
                color_option = st.selectbox(
                    "Color regions by:",
                    ["Anomaly Category", "Anomaly Score", "Price"]
                )
                
                color_by_map = {
                    "Anomaly Category": "category",
                    "Anomaly Score": "score",
                    "Price": "price"
                }
                
                with st.spinner("Generating Voronoi overlay map..."):
                    # Use unique temp file name with timestamp to avoid conflicts
                    import time
                    temp_map_path = f"temp_voronoi_map_{int(time.time())}.html"
                    
                    try:
                        viz.create_voronoi_overlay_map(
                            df['anomaly_score'].values,
                            df['anomaly_category'].values,
                            save_path=temp_map_path,
                            color_by=color_by_map[color_option],
                            max_cells=500
                        )
                        
                        with open(temp_map_path, 'r', encoding='utf-8') as f:
                            map_html = f.read()
                        
                        st.components.v1.html(map_html, height=600, scrolling=True)
                    finally:
                        # Always clean up temp file
                        if os.path.exists(temp_map_path):
                            os.remove(temp_map_path)
        
        # TAB 3: Visualizations
        with tabs[2]:
            st.header("📈 Analysis Visualizations")
            
            viz_option = st.selectbox(
                "Select visualization:",
                [
                    "Basic Voronoi Diagram",
                    "Voronoi by Cell Area",
                    "Voronoi by Price",
                    "Anomaly Detection Map",
                    "Metric Distributions",
                    "Price vs Isolation"
                ]
            )
            
            if viz_option == "Basic Voronoi Diagram":
                fig = viz.plot_basic_voronoi()
                st.pyplot(fig)
            
            elif viz_option == "Voronoi by Cell Area":
                fig = viz.plot_colored_voronoi(color_metric='cell_area')
                st.pyplot(fig)
            
            elif viz_option == "Voronoi by Price":
                fig = viz.plot_colored_voronoi(color_metric='price', cmap='viridis')
                st.pyplot(fig)
            
            elif viz_option == "Anomaly Detection Map":
                fig = viz.plot_anomalies(df['anomaly_category'].values)
                st.pyplot(fig)
            
            elif viz_option == "Metric Distributions":
                fig = viz.plot_metric_distributions()
                st.pyplot(fig)
            
            elif viz_option == "Price vs Isolation":
                fig = viz.plot_price_vs_isolation(df['anomaly_category'].values)
                st.pyplot(fig)
        
        # TAB 4: Anomaly Details
        with tabs[3]:
            st.header("🔍 Anomaly Analysis Details")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Filter Options")
                
                selected_categories = st.multiselect(
                    "Select categories:",
                    options=df['anomaly_category'].unique(),
                    default=df['anomaly_category'].unique()
                )
                
                min_score = st.slider(
                    "Minimum anomaly score:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )
                
                price_range = st.slider(
                    "Price range:",
                    min_value=float(df['price'].min()),
                    max_value=float(df['price'].max()),
                    value=(float(df['price'].min()), float(df['price'].max()))
                )
            
            with col2:
                filtered_df = df[
                    (df['anomaly_category'].isin(selected_categories)) &
                    (df['anomaly_score'] >= min_score) &
                    (df['price'] >= price_range[0]) &
                    (df['price'] <= price_range[1])
                ]
                
                st.subheader(f"Filtered Listings ({len(filtered_df)} results)")
                
                display_columns = ['name', 'neighbourhood', 'price', 'room_type', 
                                 'anomaly_score', 'anomaly_category']
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                
                st.dataframe(
                    filtered_df[available_columns].sort_values('anomaly_score', ascending=False),
                    use_container_width=True,
                    height=400
                )
            
            st.markdown("---")
            
            st.subheader("🏆 Top 20 Most Anomalous Listings")
            top_anomalies = detector.get_top_anomalies(n=20)
            st.dataframe(
                top_anomalies[available_columns],
                use_container_width=True,
                height=400
            )
        
        # TAB 5: Report
        with tabs[4]:
            st.header("📄 Analysis Report")
            
            anomalies = df[df['anomaly_category'] != 'Normal']
            total_listings = len(df)
            num_anomalies = len(anomalies)
            pct_anomalies = (num_anomalies / total_listings) * 100
            
            st.subheader("Executive Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analyzed", f"{total_listings:,}")
            with col2:
                st.metric("Anomalies", f"{num_anomalies:,}")
            with col3:
                st.metric("Anomaly Rate", f"{pct_anomalies:.2f}%")
            
            st.markdown("---")
            
            st.subheader("📊 Category Breakdown")
            category_counts = df['anomaly_category'].value_counts()
            for category, count in category_counts.items():
                pct = (count / total_listings) * 100
                st.write(f"**{category}:** {count:,} listings ({pct:.2f}%)")
            
            st.markdown("---")
            
            st.subheader("💡 Key Insights")
            
            metrics = st.session_state.metrics
            cell_areas = metrics['cell_area']
            density = metrics['density']
            
            st.write(f"""
            **Spatial Distribution:**
            - {(cell_areas == np.inf).sum()} listings are on the boundary with infinite cells
            - {(density < np.percentile(density, 10)).sum()} listings are in low-density areas
            - Average nearest neighbor distance: {metrics['nearest_neighbor'].mean():.6f} degrees
            
            **Price Analysis:**
            - Anomalous listings average price: ${anomalies['price'].mean():.2f}
            - Normal listings average price: ${df[df['anomaly_category'] == 'Normal']['price'].mean():.2f}
            - Price difference: {((anomalies['price'].mean() / df[df['anomaly_category'] == 'Normal']['price'].mean() - 1) * 100):.1f}%
            
            **Anomaly Patterns:**
            - {(df['anomaly_score'] > 0.8).sum()} listings have very high anomaly scores (>0.8)
            - Most common anomaly type: {category_counts.index[category_counts != category_counts.get('Normal', 0)][0] if len(category_counts) > 1 else 'N/A'}
            
            **Recommendations:**
            - ✅ Review 'Isolated' listings for potential data quality issues
            - ✅ Investigate 'Price Anomaly' listings for market opportunities
            - ✅ Monitor 'Multi-factor Anomaly' listings for suspicious patterns
            """)
        
        # TAB 6: Download
        with tabs[5]:
            st.header("💾 Download Results")
            
            st.subheader("Download Processed Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="airbnb_anomaly_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Anomalies only CSV
                anomalies_csv = df[df['anomaly_category'] != 'Normal'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Anomalies Only (CSV)",
                    data=anomalies_csv,
                    file_name="airbnb_anomalies_only.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON download
                json_data = df.to_json(orient='records', indent=2).encode('utf-8')
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name="airbnb_anomaly_results.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                # Excel download
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='All Listings', index=False)
                    df[df['anomaly_category'] != 'Normal'].to_excel(writer, sheet_name='Anomalies', index=False)
                
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer.getvalue(),
                    file_name="airbnb_anomaly_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
