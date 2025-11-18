"""
Voronoi diagram generation and spatial metrics calculation module.
"""

import numpy as np
from scipy.spatial import Voronoi, distance
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class VoronoiAnalyzer:
    """
    Generates Voronoi diagrams and calculates spatial metrics for Airbnb listings.
    """
    
    def __init__(self, points):
        """
        Initialize the Voronoi analyzer.
        
        Args:
            points (np.ndarray): Array of [longitude, latitude] coordinates
        """
        self.points = points
        self.vor = None
        self.cell_areas = None
        self.cell_perimeters = None
        
    def generate_voronoi(self):
        """
        Generate Voronoi diagram from points.
        
        Returns:
            scipy.spatial.Voronoi: Voronoi diagram object
        """
        self.vor = Voronoi(self.points)
        print(f"Generated Voronoi diagram with {len(self.vor.regions)} regions")
        return self.vor
    
    def calculate_cell_areas(self, bound_box=None):
        """
        Calculate area of each Voronoi cell.
        
        Args:
            bound_box (list, optional): Bounding box to clip infinite regions
            
        Returns:
            np.ndarray: Array of cell areas
        """
        areas = []
        
        for point_idx in range(len(self.points)):
            region_idx = self.vor.point_region[point_idx]
            region = self.vor.regions[region_idx]
            
            # Handle infinite or empty regions
            if -1 in region or len(region) == 0:
                areas.append(np.inf)
                continue
            
            try:
                # Get vertices of the region
                vertices = [self.vor.vertices[i] for i in region]
                polygon = Polygon(vertices)
                
                # Clip to bounding box if provided
                if bound_box and polygon.is_valid:
                    bbox = Polygon(bound_box)
                    polygon = polygon.intersection(bbox)
                
                if polygon.is_valid and not polygon.is_empty:
                    areas.append(polygon.area)
                else:
                    areas.append(np.inf)
                    
            except Exception as e:
                # Handle any geometric errors
                areas.append(np.inf)
        
        self.cell_areas = np.array(areas)
        finite_areas = self.cell_areas[~np.isinf(self.cell_areas)]
        print(f"Calculated {len(finite_areas)} finite cell areas")
        print(f"  Mean area: {finite_areas.mean():.6f}")
        print(f"  Max area: {finite_areas.max():.6f}")
        return self.cell_areas
    
    def calculate_cell_perimeters(self):
        """
        Calculate perimeter of each Voronoi cell.
        
        Returns:
            np.ndarray: Array of cell perimeters
        """
        perimeters = []
        
        for point_idx in range(len(self.points)):
            region_idx = self.vor.point_region[point_idx]
            region = self.vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                perimeters.append(np.inf)
                continue
            
            try:
                vertices = [self.vor.vertices[i] for i in region]
                polygon = Polygon(vertices)
                
                if polygon.is_valid:
                    perimeters.append(polygon.length)
                else:
                    perimeters.append(np.inf)
            except:
                perimeters.append(np.inf)
        
        self.cell_perimeters = np.array(perimeters)
        return self.cell_perimeters
    
    def calculate_nearest_neighbors(self, k=5):
        """
        Calculate distance to k nearest neighbors for each point.
        
        Args:
            k (int): Number of nearest neighbors to consider
            
        Returns:
            tuple: (average distances, k-nearest distances matrix)
        """
        # Calculate pairwise distances
        distances = distance.cdist(self.points, self.points)
        
        # Set diagonal to infinity to exclude self
        np.fill_diagonal(distances, np.inf)
        
        # Get k nearest for each point
        nearest_k = np.partition(distances, k-1, axis=1)[:, :k]
        avg_distance = nearest_k.mean(axis=1)
        
        print(f"Calculated {k}-nearest neighbor distances")
        print(f"  Mean NN distance: {avg_distance.mean():.6f}")
        print(f"  Max NN distance: {avg_distance.max():.6f}")
        
        return avg_distance, nearest_k
    
    def calculate_density(self, radius=0.01):
        """
        Calculate local density (number of points within radius) for each point.
        
        Args:
            radius (float): Search radius in coordinate units
            
        Returns:
            np.ndarray: Number of neighbors within radius for each point
        """
        distances = distance.cdist(self.points, self.points)
        
        # Count points within radius (excluding self)
        within_radius = np.sum((distances < radius) & (distances > 0), axis=1)
        
        print(f"Calculated local density (radius={radius})")
        print(f"  Mean density: {within_radius.mean():.2f}")
        print(f"  Max density: {within_radius.max()}")
        
        return within_radius
    
    def calculate_perimeter_to_area_ratio(self):
        """
        Calculate the perimeter-to-area ratio for each cell (compactness measure).
        
        Returns:
            np.ndarray: Perimeter-to-area ratios
        """
        if self.cell_areas is None:
            raise ValueError("Cell areas not calculated. Call calculate_cell_areas() first.")
        
        if self.cell_perimeters is None:
            self.calculate_cell_perimeters()
        
        # Avoid division by zero
        ratios = np.zeros_like(self.cell_areas)
        valid_mask = (self.cell_areas > 0) & ~np.isinf(self.cell_areas) & ~np.isinf(self.cell_perimeters)
        ratios[valid_mask] = self.cell_perimeters[valid_mask] / self.cell_areas[valid_mask]
        ratios[~valid_mask] = np.inf
        
        return ratios
    
    def get_voronoi_edges(self):
        """
        Extract Voronoi edges for visualization.
        
        Returns:
            list: List of edge line segments as [(x1, y1), (x2, y2)]
        """
        edges = []
        
        for simplex in self.vor.ridge_vertices:
            if -1 not in simplex:
                # Both vertices are finite
                v1 = self.vor.vertices[simplex[0]]
                v2 = self.vor.vertices[simplex[1]]
                edges.append([v1, v2])
        
        return edges
    
    def get_metrics_summary(self):
        """
        Get summary of all calculated metrics.
        
        Returns:
            dict: Dictionary of metric summaries
        """
        summary = {}
        
        if self.cell_areas is not None:
            finite_areas = self.cell_areas[~np.isinf(self.cell_areas)]
            summary['cell_areas'] = {
                'mean': finite_areas.mean(),
                'median': np.median(finite_areas),
                'std': finite_areas.std(),
                'min': finite_areas.min(),
                'max': finite_areas.max()
            }
        
        return summary
