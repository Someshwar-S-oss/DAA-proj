"""
Visualization module for Voronoi diagrams and anomaly detection results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from matplotlib.collections import LineCollection
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10


class VoronoiVisualizer:
    """
    Creates various visualizations for Voronoi analysis and anomaly detection.
    """
    
    def __init__(self, vor, data, metrics):
        """
        Initialize the visualizer.
        
        Args:
            vor (scipy.spatial.Voronoi): Voronoi diagram object
            data (pd.DataFrame): Airbnb listings data
            metrics (dict): Dictionary of calculated metrics
        """
        self.vor = vor
        self.data = data
        self.metrics = metrics
        
    def plot_basic_voronoi(self, figsize=(15, 12), save_path=None):
        """
        Plot basic Voronoi diagram with edges and points.
        
        Args:
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot Voronoi edges
        for simplex in self.vor.ridge_vertices:
            if -1 not in simplex:
                simplex = np.asarray(simplex)
                ax.plot(self.vor.vertices[simplex, 0], 
                       self.vor.vertices[simplex, 1], 
                       'k-', linewidth=0.5, alpha=0.5)
        
        # Plot points
        ax.plot(self.vor.points[:, 0], self.vor.points[:, 1], 
               'b.', markersize=3, alpha=0.6)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Airbnb Listings - Voronoi Diagram', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Format axes to show proper decimal precision
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        
        # Set reasonable axis limits based on data range
        lon_range = self.data['longitude'].max() - self.data['longitude'].min()
        lat_range = self.data['latitude'].max() - self.data['latitude'].min()
        ax.set_xlim(self.data['longitude'].min() - lon_range*0.05, 
                   self.data['longitude'].max() + lon_range*0.05)
        ax.set_ylim(self.data['latitude'].min() - lat_range*0.05, 
                   self.data['latitude'].max() + lat_range*0.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_colored_voronoi(self, color_metric='cell_area', 
                            figsize=(15, 12), cmap='YlOrRd', save_path=None):
        """
        Plot Voronoi diagram with points colored by metric value.
        
        Args:
            color_metric (str): Metric to use for coloring
            figsize (tuple): Figure size
            cmap (str): Colormap name
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot Voronoi edges
        for simplex in self.vor.ridge_vertices:
            if -1 not in simplex:
                ax.plot(self.vor.vertices[simplex, 0], 
                       self.vor.vertices[simplex, 1], 
                       'k-', alpha=0.3, linewidth=0.5)
        
        # Get metric values and handle infinities
        metric_values = self.metrics[color_metric].copy()
        finite_mask = ~np.isinf(metric_values)
        
        # Create color array
        colors = np.zeros_like(metric_values)
        if finite_mask.sum() > 0:
            colors[finite_mask] = metric_values[finite_mask]
            colors[~finite_mask] = metric_values[finite_mask].max() * 1.5
        
        # Scatter plot with color mapping
        scatter = ax.scatter(
            self.data['longitude'],
            self.data['latitude'],
            c=colors,
            cmap=cmap,
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_metric.replace('_', ' ').title(), fontsize=12)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'Voronoi Diagram - Colored by {color_metric.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Format axes to show proper decimal precision
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        
        # Set reasonable axis limits
        lon_range = self.data['longitude'].max() - self.data['longitude'].min()
        lat_range = self.data['latitude'].max() - self.data['latitude'].min()
        ax.set_xlim(self.data['longitude'].min() - lon_range*0.05, 
                   self.data['longitude'].max() + lon_range*0.05)
        ax.set_ylim(self.data['latitude'].min() - lat_range*0.05, 
                   self.data['latitude'].max() + lat_range*0.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_anomalies(self, anomaly_labels, figsize=(15, 12), save_path=None):
        """
        Highlight anomalous listings on the map.
        
        Args:
            anomaly_labels (np.ndarray): Array of anomaly category labels
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot Voronoi edges (lighter)
        for simplex in self.vor.ridge_vertices:
            if -1 not in simplex:
                ax.plot(self.vor.vertices[simplex, 0], 
                       self.vor.vertices[simplex, 1], 
                       'gray', alpha=0.2, linewidth=0.5)
        
        # Define colors for each category
        category_colors = {
            'Normal': 'lightblue',
            'Isolated': 'red',
            'Price Anomaly': 'orange',
            'Low Density': 'purple',
            'Multi-factor Anomaly': 'darkred'
        }
        
        # Plot each category
        for category, color in category_colors.items():
            mask = anomaly_labels == category
            if mask.sum() > 0:
                size = 30 if category == 'Normal' else 100
                marker = 'o' if category == 'Normal' else '*'
                alpha = 0.4 if category == 'Normal' else 0.8
                
                ax.scatter(
                    self.data.loc[mask, 'longitude'],
                    self.data.loc[mask, 'latitude'],
                    c=color,
                    s=size,
                    alpha=alpha,
                    marker=marker,
                    edgecolors='black' if category != 'Normal' else 'none',
                    linewidth=0.5,
                    label=f'{category} ({mask.sum()})'
                )
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Anomalous Listings Detection', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format axes to show proper decimal precision
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        
        # Set reasonable axis limits
        lon_range = self.data['longitude'].max() - self.data['longitude'].min()
        lat_range = self.data['latitude'].max() - self.data['latitude'].min()
        ax.set_xlim(self.data['longitude'].min() - lon_range*0.05, 
                   self.data['longitude'].max() + lon_range*0.05)
        ax.set_ylim(self.data['latitude'].min() - lat_range*0.05, 
                   self.data['latitude'].max() + lat_range*0.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_metric_distributions(self, figsize=(15, 10), save_path=None):
        """
        Plot distribution histograms for all metrics.
        
        Args:
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Select metrics to plot
        metrics_to_plot = []
        for key in ['cell_area', 'nearest_neighbor', 'density', 'price']:
            if key in self.metrics:
                metrics_to_plot.append(key)
        
        n_metrics = len(metrics_to_plot)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            values = self.metrics[metric]
            finite_values = values[~np.isinf(values)]
            
            if len(finite_values) > 0:
                axes[idx].hist(finite_values, bins=50, 
                             edgecolor='black', alpha=0.7, color='steelblue')
                axes[idx].set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
                axes[idx].set_ylabel('Frequency', fontsize=11)
                axes[idx].set_title(f'Distribution of {metric.replace("_", " ").title()}', 
                                  fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = finite_values.mean()
                median_val = np.median(finite_values)
                axes[idx].axvline(mean_val, color='red', linestyle='--', 
                                linewidth=2, label=f'Mean: {mean_val:.2f}')
                axes[idx].axvline(median_val, color='green', linestyle='--', 
                                linewidth=2, label=f'Median: {median_val:.2f}')
                axes[idx].legend(fontsize=9)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_scatter_matrix(self, anomaly_scores, figsize=(12, 10), save_path=None):
        """
        Create scatter plot matrix of metrics colored by anomaly score.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores for coloring
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Prepare data
        import pandas as pd
        
        plot_data = pd.DataFrame()
        for key in ['cell_area', 'nearest_neighbor', 'density', 'price']:
            if key in self.metrics:
                values = self.metrics[key].copy()
                finite_mask = ~np.isinf(values)
                values[~finite_mask] = np.nan
                plot_data[key.replace('_', ' ').title()] = values
        
        plot_data['Anomaly Score'] = anomaly_scores
        
        # Remove rows with NaN
        plot_data_clean = plot_data.dropna()
        
        # Create scatter matrix
        fig = plt.figure(figsize=figsize)
        
        from pandas.plotting import scatter_matrix
        scatter_matrix(plot_data_clean, 
                      c=plot_data_clean['Anomaly Score'], 
                      cmap='YlOrRd',
                      figsize=figsize,
                      alpha=0.6,
                      diagonal='hist',
                      s=20)
        
        plt.suptitle('Metrics Scatter Matrix (colored by Anomaly Score)', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_price_vs_isolation(self, anomaly_labels, figsize=(12, 8), save_path=None):
        """
        Plot price vs isolation metrics to identify suspicious patterns.
        
        Args:
            anomaly_labels (np.ndarray): Anomaly category labels
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get metrics
        cell_areas = self.metrics['cell_area'].copy()
        prices = self.metrics['price']
        
        # Handle infinities
        finite_mask = ~np.isinf(cell_areas)
        cell_areas[~finite_mask] = cell_areas[finite_mask].max() * 1.5
        
        # Define colors
        category_colors = {
            'Normal': 'lightblue',
            'Isolated': 'red',
            'Price Anomaly': 'orange',
            'Low Density': 'purple',
            'Multi-factor Anomaly': 'darkred'
        }
        
        # Plot each category
        for category, color in category_colors.items():
            mask = anomaly_labels == category
            if mask.sum() > 0:
                size = 30 if category == 'Normal' else 100
                alpha = 0.4 if category == 'Normal' else 0.7
                
                ax.scatter(
                    cell_areas[mask],
                    prices[mask],
                    c=color,
                    s=size,
                    alpha=alpha,
                    edgecolors='black' if category != 'Normal' else 'none',
                    linewidth=0.5,
                    label=f'{category} ({mask.sum()})'
                )
        
        ax.set_xlabel('Cell Area (Isolation Measure)', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title('Price vs Isolation Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Use log scale if range is large
        if cell_areas[finite_mask].max() / cell_areas[finite_mask].min() > 100:
            ax.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def create_interactive_map_html(self, anomaly_scores, anomaly_labels, save_path):
        """
        Create an interactive HTML map using folium.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            anomaly_labels (np.ndarray): Anomaly category labels
            save_path (str): Path to save the HTML file
        """
        try:
            import folium
            from folium import plugins
            
            # Calculate map center
            center_lat = self.data['latitude'].mean()
            center_lon = self.data['longitude'].mean()
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            
            # Define marker colors
            color_map = {
                'Normal': 'blue',
                'Isolated': 'red',
                'Price Anomaly': 'orange',
                'Low Density': 'purple',
                'Multi-factor Anomaly': 'darkred'
            }
            
            # Add markers
            for idx, row in self.data.iterrows():
                category = anomaly_labels[idx]
                score = anomaly_scores[idx]
                
                # Create popup text
                popup_text = f"""
                <b>{row.get('name', 'Unknown')}</b><br>
                Price: ${row.get('price', 'N/A')}<br>
                Room Type: {row.get('room_type', 'N/A')}<br>
                Neighbourhood: {row.get('neighbourhood', 'N/A')}<br>
                Anomaly Score: {score:.3f}<br>
                Category: {category}
                """
                
                # Choose icon
                icon = folium.Icon(
                    color=color_map.get(category, 'gray'),
                    icon='info-sign' if category == 'Normal' else 'warning-sign'
                )
                
                # Add marker
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=icon
                ).add_to(m)
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; 
                        top: 10px; right: 10px; width: 200px; height: auto; 
                        background-color: white; z-index:9999; font-size:14px;
                        border:2px solid grey; border-radius: 5px; padding: 10px">
                <p style="margin: 0; font-weight: bold;">Anomaly Categories</p>
                <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:blue"></i> Normal</p>
                <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:red"></i> Isolated</p>
                <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:orange"></i> Price Anomaly</p>
                <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:purple"></i> Low Density</p>
                <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:darkred"></i> Multi-factor</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Save map
            m.save(save_path)
            print(f"Saved interactive map: {save_path}")
            
        except ImportError:
            print("Warning: folium not installed. Skipping interactive map creation.")
            print("Install with: pip install folium")
    
    def create_voronoi_overlay_map(self, anomaly_scores, anomaly_labels, save_path, 
                                   color_by='category', max_cells=500):
        """
        Create an interactive map with Voronoi regions overlaid as colored polygons.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores
            anomaly_labels (np.ndarray): Anomaly category labels
            save_path (str): Path to save the HTML file
            color_by (str): 'category', 'score', or 'price' - what to color regions by
            max_cells (int): Maximum number of cells to render (for performance)
        """
        try:
            import folium
            from shapely.geometry import Polygon
            import matplotlib.colors as mcolors
            import matplotlib.cm as cm
            
            # Calculate map center
            center_lat = self.data['latitude'].mean()
            center_lon = self.data['longitude'].mean()
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='CartoDB positron'
            )
            
            # Define colors for categories
            category_color_map = {
                'Normal': '#90caf9',  # Light blue
                'Isolated': '#f44336',  # Red
                'Price Anomaly': '#ff9800',  # Orange
                'Low Density': '#9c27b0',  # Purple
                'Multi-factor Anomaly': '#b71c1c'  # Dark red
            }
            
            # Calculate bounding box
            lat_range = [self.data['latitude'].min(), self.data['latitude'].max()]
            lon_range = [self.data['longitude'].min(), self.data['longitude'].max()]
            lat_margin = (lat_range[1] - lat_range[0]) * 0.1
            lon_margin = (lon_range[1] - lon_range[0]) * 0.1
            bbox = [lon_range[0] - lon_margin, lat_range[0] - lat_margin,
                   lon_range[1] + lon_margin, lat_range[1] + lat_margin]
            
            # Create Voronoi regions
            from scipy.spatial import voronoi_plot_2d
            import matplotlib.pyplot as plt
            
            # Only process finite regions
            cells_added = 0
            
            for point_idx, region_idx in enumerate(self.vor.point_region):
                if cells_added >= max_cells:
                    break
                    
                region = self.vor.regions[region_idx]
                
                # Skip infinite regions and empty regions
                if not region or -1 in region:
                    continue
                
                # Get vertices for this region
                vertices = [self.vor.vertices[i] for i in region]
                polygon_coords = [(v[1], v[0]) for v in vertices]  # (lat, lon) for folium
                
                # Check if polygon is within reasonable bounds
                lons = [v[0] for v in vertices]
                lats = [v[1] for v in vertices]
                
                if (min(lons) < bbox[0] or max(lons) > bbox[2] or 
                    min(lats) < bbox[1] or max(lats) > bbox[3]):
                    continue
                
                # Determine color based on color_by parameter
                if color_by == 'category':
                    category = anomaly_labels[point_idx]
                    fill_color = category_color_map.get(category, '#gray')
                    fill_opacity = 0.4 if category == 'Normal' else 0.6
                elif color_by == 'score':
                    score = anomaly_scores[point_idx]
                    cmap = cm.get_cmap('YlOrRd')
                    rgba = cmap(score)
                    fill_color = mcolors.rgb2hex(rgba[:3])
                    fill_opacity = 0.5
                elif color_by == 'price':
                    price = self.data.iloc[point_idx]['price']
                    max_price = self.data['price'].quantile(0.95)
                    normalized_price = min(price / max_price, 1.0)
                    cmap = cm.get_cmap('viridis')
                    rgba = cmap(normalized_price)
                    fill_color = mcolors.rgb2hex(rgba[:3])
                    fill_opacity = 0.5
                else:
                    fill_color = '#gray'
                    fill_opacity = 0.3
                
                # Create popup info
                row = self.data.iloc[point_idx]
                popup_text = f"""
                <b>{row.get('name', 'Unknown')[:50]}</b><br>
                Price: ${row.get('price', 'N/A')}<br>
                Room Type: {row.get('room_type', 'N/A')}<br>
                Neighbourhood: {row.get('neighbourhood', 'N/A')}<br>
                Anomaly Score: {anomaly_scores[point_idx]:.3f}<br>
                Category: {anomaly_labels[point_idx]}
                """
                
                # Add polygon to map
                folium.Polygon(
                    locations=polygon_coords,
                    popup=folium.Popup(popup_text, max_width=300),
                    color='black',
                    weight=1,
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity
                ).add_to(m)
                
                cells_added += 1
            
            # Add central point markers (smaller)
            for idx, row in self.data.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=2,
                    color='black',
                    fill=True,
                    fill_color='white',
                    fill_opacity=0.8,
                    weight=1
                ).add_to(m)
            
            # Add legend based on color_by
            if color_by == 'category':
                legend_html = '''
                <div style="position: fixed; 
                            top: 10px; right: 10px; width: 220px; height: auto; 
                            background-color: white; z-index:9999; font-size:13px;
                            border:2px solid grey; border-radius: 5px; padding: 10px">
                    <p style="margin: 0 0 8px 0; font-weight: bold;">Voronoi Regions by Category</p>
                    <p style="margin: 5px 0;"><span style="background-color:#90caf9; padding: 3px 10px; border: 1px solid black;"></span> Normal</p>
                    <p style="margin: 5px 0;"><span style="background-color:#f44336; padding: 3px 10px; border: 1px solid black;"></span> Isolated</p>
                    <p style="margin: 5px 0;"><span style="background-color:#ff9800; padding: 3px 10px; border: 1px solid black;"></span> Price Anomaly</p>
                    <p style="margin: 5px 0;"><span style="background-color:#9c27b0; padding: 3px 10px; border: 1px solid black;"></span> Low Density</p>
                    <p style="margin: 5px 0;"><span style="background-color:#b71c1c; padding: 3px 10px; border: 1px solid black;"></span> Multi-factor</p>
                </div>
                '''
            elif color_by == 'score':
                legend_html = '''
                <div style="position: fixed; 
                            top: 10px; right: 10px; width: 220px; height: auto; 
                            background-color: white; z-index:9999; font-size:13px;
                            border:2px solid grey; border-radius: 5px; padding: 10px">
                    <p style="margin: 0 0 8px 0; font-weight: bold;">Voronoi Regions by Anomaly Score</p>
                    <p style="margin: 5px 0;">Low Score <span style="background: linear-gradient(to right, #ffffcc, #ff0000); padding: 3px 50px; border: 1px solid black;"></span> High Score</p>
                </div>
                '''
            else:  # price
                legend_html = '''
                <div style="position: fixed; 
                            top: 10px; right: 10px; width: 220px; height: auto; 
                            background-color: white; z-index:9999; font-size:13px;
                            border:2px solid grey; border-radius: 5px; padding: 10px">
                    <p style="margin: 0 0 8px 0; font-weight: bold;">Voronoi Regions by Price</p>
                    <p style="margin: 5px 0;">Low Price <span style="background: linear-gradient(to right, #440154, #fde724); padding: 3px 50px; border: 1px solid black;"></span> High Price</p>
                </div>
                '''
            
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add info text
            info_html = f'''
            <div style="position: fixed; 
                        bottom: 10px; left: 10px; width: 250px; 
                        background-color: white; z-index:9999; font-size:11px;
                        border:2px solid grey; border-radius: 5px; padding: 8px">
                <p style="margin: 0;"><b>Voronoi Diagram Overlay</b></p>
                <p style="margin: 3px 0;">Showing {cells_added} regions</p>
                <p style="margin: 3px 0;">Click regions for details</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(info_html))
            
            # Save map
            m.save(save_path)
            print(f"✅ Saved Voronoi overlay map with {cells_added} regions: {save_path}")
            
        except ImportError as e:
            print(f"Warning: Required library not installed: {e}")
            print("Install with: pip install folium shapely")
        except Exception as e:
            print(f"Error creating Voronoi overlay map: {e}")
